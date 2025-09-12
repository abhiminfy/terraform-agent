# policy_engine.py - Policy-as-code implementation with tfsec/checkov
# integration
import json
import json as _pe_json
import logging
import subprocess
import tempfile
from io import StringIO as _PE_StringIO
from pathlib import Path
from typing import Any
from typing import Any as _Any
from typing import Dict
from typing import Dict as _Dict
from typing import List
from typing import List as _List

import hcl2 as _pe_hcl2

from src.app.utils.utils import run_cmd_async as _pe_run_cmd_async
from src.app.utils.utils import secure_tempdir as _pe_secure_tempdir

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Policy-as-code engine for Terraform validation with multiple security scanners"""

    def __init__(self):
        self.policy_rules = self._load_default_policies()
        self.enabled_scanners = self._detect_available_scanners()

    def _detect_available_scanners(self) -> Dict[str, bool]:
        """Detect which security scanners are available"""
        scanners = {
            "tfsec": False,
            "checkov": False,
            "terrascan": False,
            "tflint": False,
        }

        for scanner in scanners.keys():
            try:
                result = subprocess.run([scanner, "--version"], capture_output=True, timeout=5)
                scanners[scanner] = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        logger.info(f"Available security scanners: {[k for k, v in scanners.items() if v]}")
        return scanners

    def _load_default_policies(self) -> Dict[str, Any]:
        """Load default security policies"""
        return {
            "aws_security_groups": {
                "no_unrestricted_ingress": {
                    "severity": "HIGH",
                    "pattern": r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                    "message": "Security group allows unrestricted ingress (0.0.0.0/0)",
                },
                "no_ssh_from_internet": {
                    "severity": "CRITICAL",
                    "pattern": r'from_port\s*=\s*22.*cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                    "message": "SSH (port 22) accessible from internet",
                },
            },
            "aws_s3_buckets": {
                "block_public_access": {
                    "severity": "HIGH",
                    "pattern": r"block_public_acls\s*=\s*false",
                    "message": "S3 bucket allows public ACLs",
                },
                "versioning_enabled": {
                    "severity": "MEDIUM",
                    "pattern": r"versioning\s*\{\s*enabled\s*=\s*false",
                    "message": "S3 bucket versioning disabled",
                },
            },
            "aws_rds": {
                "encryption_at_rest": {
                    "severity": "HIGH",
                    "pattern": r"storage_encrypted\s*=\s*false",
                    "message": "RDS instance encryption at rest disabled",
                },
                "backup_retention": {
                    "severity": "MEDIUM",
                    "pattern": r"backup_retention_period\s*=\s*0",
                    "message": "RDS backup retention disabled",
                },
            },
            "general": {
                "no_hardcoded_secrets": {
                    "severity": "CRITICAL",
                    "pattern": r'(password|secret|key)\s*=\s*"[^$][^{][^v][^a][^r].*"',
                    "message": "Hardcoded credentials detected",
                }
            },
        }

    def validate_with_policies(self, terraform_code: str) -> Dict[str, Any]:
        """Validate Terraform code against all available policies"""
        if not terraform_code or not terraform_code.strip():
            return {
                "valid": False,
                "violations": [],
                "error": "Empty Terraform code provided",
            }

        violations = []

        # Run internal policy checks
        internal_violations = self._run_internal_policies(terraform_code)
        violations.extend(internal_violations)

        # Run external security scanners
        external_violations = self._run_external_scanners(terraform_code)
        violations.extend(external_violations)

        # Determine overall validation result
        critical_violations = [v for v in violations if v.get("severity") == "CRITICAL"]
        high_violations = [v for v in violations if v.get("severity") == "HIGH"]

        is_valid = len(critical_violations) == 0 and len(high_violations) == 0

        return {
            "valid": is_valid,
            "violations": violations,
            "summary": {
                "total_violations": len(violations),
                "critical": len(critical_violations),
                "high": len([v for v in violations if v.get("severity") == "HIGH"]),
                "medium": len([v for v in violations if v.get("severity") == "MEDIUM"]),
                "low": len([v for v in violations if v.get("severity") == "LOW"]),
            },
            "scanners_used": [k for k, v in self.enabled_scanners.items() if v],
            "policy_version": "1.0",
        }

    def _run_internal_policies(self, terraform_code: str) -> List[Dict[str, Any]]:
        """Run internal policy rules"""
        violations = []

        import re

        for category, rules in self.policy_rules.items():
            for rule_name, rule_config in rules.items():
                pattern = rule_config.get("pattern", "")
                severity = rule_config.get("severity", "MEDIUM")
                message = rule_config.get("message", f"Policy violation: {rule_name}")

                try:
                    if re.search(pattern, terraform_code, re.IGNORECASE | re.MULTILINE):
                        violations.append(
                            {
                                "rule_id": f"{category}.{rule_name}",
                                "severity": severity,
                                "message": message,
                                "scanner": "internal",
                                "category": category,
                                "fix_suggestion": self._get_fix_suggestion(rule_name),
                            }
                        )
                except re.error as e:
                    logger.error(f"Regex error in rule {category}.{rule_name}: {e}")
                    continue

        return violations

    def _run_external_scanners(self, terraform_code: str) -> List[Dict[str, Any]]:
        """Run external security scanners"""
        violations = []

        with tempfile.TemporaryDirectory() as temp_dir:
            tf_file = Path(temp_dir) / "main.tf"
            tf_file.write_text(terraform_code, encoding="utf-8", errors="ignore")

            # Run tfsec if available
            if self.enabled_scanners.get("tfsec", False):
                tfsec_violations = self._run_tfsec(temp_dir)
                violations.extend(tfsec_violations)

            # Run checkov if available
            if self.enabled_scanners.get("checkov", False):
                checkov_violations = self._run_checkov(temp_dir)
                violations.extend(checkov_violations)

            # Run terrascan if available
            if self.enabled_scanners.get("terrascan", False):
                terrascan_violations = self._run_terrascan(temp_dir)
                violations.extend(terrascan_violations)

            # Run tflint if available
            if self.enabled_scanners.get("tflint", False):
                tflint_violations = self._run_tflint(temp_dir)
                violations.extend(tflint_violations)

        return violations

    def _run_tfsec(self, directory: str) -> List[Dict[str, Any]]:
        """Run tfsec security scanner"""
        violations = []

        try:
            result = subprocess.run(
                ["tfsec", directory, "--format", "json", "--no-color"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                try:
                    tfsec_output = json.loads(result.stdout)
                    results = tfsec_output.get("results", [])

                    for issue in results:
                        violations.append(
                            {
                                "rule_id": issue.get("rule_id", "tfsec-unknown"),
                                "severity": self._normalize_severity(
                                    issue.get("severity", "MEDIUM")
                                ),
                                "message": issue.get("description", "tfsec security issue"),
                                "scanner": "tfsec",
                                "category": "security",
                                "line": issue.get("location", {}).get("start_line"),
                                "fix_suggestion": issue.get(
                                    "resolution",
                                    "Review and fix security issue",
                                ),
                            }
                        )

                except json.JSONDecodeError:
                    logger.warning("Failed to parse tfsec JSON output")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"tfsec scan failed: {str(e)}")

        return violations

    def _run_checkov(self, directory: str) -> List[Dict[str, Any]]:
        """Run checkov security scanner"""
        violations = []

        try:
            result = subprocess.run(
                ["checkov", "-d", directory, "--output", "json", "--quiet"],
                capture_output=True,
                text=True,
                timeout=90,
            )

            if result.stdout:
                try:
                    checkov_output = json.loads(result.stdout)
                    failed_checks = checkov_output.get("results", {}).get("failed_checks", [])

                    for check in failed_checks:
                        violations.append(
                            {
                                "rule_id": check.get("check_id", "checkov-unknown"),
                                "severity": self._normalize_severity(
                                    "HIGH"
                                ),  # Checkov doesn't provide severity
                                "message": check.get(
                                    "check_name",
                                    "Checkov security check failed",
                                ),
                                "scanner": "checkov",
                                "category": "security",
                                "file": check.get("file_path"),
                                "line": check.get("file_line_range", [0])[0],
                                "fix_suggestion": check.get(
                                    "guideline", "Review Checkov documentation"
                                ),
                            }
                        )

                except json.JSONDecodeError:
                    logger.warning("Failed to parse checkov JSON output")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"checkov scan failed: {str(e)}")

        return violations

    def _run_terrascan(self, directory: str) -> List[Dict[str, Any]]:
        """Run terrascan security scanner"""
        violations = []

        try:
            result = subprocess.run(
                ["terrascan", "scan", "-d", directory, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                try:
                    terrascan_output = json.loads(result.stdout)
                    violations_data = terrascan_output.get("results", {}).get("violations", [])

                    for violation in violations_data:
                        violations.append(
                            {
                                "rule_id": violation.get("rule_id", "terrascan-unknown"),
                                "severity": self._normalize_severity(
                                    violation.get("severity", "MEDIUM")
                                ),
                                "message": violation.get("description", "Terrascan security issue"),
                                "scanner": "terrascan",
                                "category": violation.get("category", "security"),
                                "line": violation.get("line"),
                                "fix_suggestion": "Review Terrascan documentation for remediation",
                            }
                        )

                except json.JSONDecodeError:
                    logger.warning("Failed to parse terrascan JSON output")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"terrascan scan failed: {str(e)}")

        return violations

    def _run_tflint(self, directory: str) -> List[Dict[str, Any]]:
        """Run tflint linter"""
        violations = []

        try:
            # Initialize tflint in the directory
            subprocess.run(
                ["tflint", "--init"],
                cwd=directory,
                capture_output=True,
                timeout=30,
            )

            result = subprocess.run(
                ["tflint", "--format", "json"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                try:
                    tflint_output = json.loads(result.stdout)
                    issues = tflint_output.get("issues", [])

                    for issue in issues:
                        violations.append(
                            {
                                "rule_id": issue.get("rule", {}).get("name", "tflint-unknown"),
                                "severity": self._normalize_severity(
                                    issue.get("rule", {}).get("severity", "MEDIUM")
                                ),
                                "message": issue.get("message", "TFLint issue"),
                                "scanner": "tflint",
                                "category": "linting",
                                "line": issue.get("range", {}).get("start", {}).get("line"),
                                "fix_suggestion": "Review TFLint documentation for remediation",
                            }
                        )

                except json.JSONDecodeError:
                    logger.warning("Failed to parse tflint JSON output")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"tflint scan failed: {str(e)}")

        return violations

    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across different scanners"""
        severity_map = {
            "error": "CRITICAL",
            "warning": "MEDIUM",
            "info": "LOW",
            "critical": "CRITICAL",
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW",
        }

        return severity_map.get(severity.lower(), "MEDIUM")

    def _get_fix_suggestion(self, rule_name: str) -> str:
        """Get fix suggestions for common policy violations"""
        suggestions = {
            "no_unrestricted_ingress": "Restrict CIDR blocks to specific IP ranges instead of 0.0.0.0/0",
            "no_ssh_from_internet": "Restrict SSH access to specific IP ranges or use a bastion host",
            "block_public_access": "Enable S3 bucket public access blocks",
            "versioning_enabled": "Enable S3 bucket versioning for data protection",
            "encryption_at_rest": "Enable storage encryption for RDS instances",
            "backup_retention": "Set backup_retention_period > 0 for RDS instances",
            "no_hardcoded_secrets": "Use variables, AWS Secrets Manager, or Parameter Store",
        }

        return suggestions.get(rule_name, "Review security best practices documentation")

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of available policies and scanners"""
        return {
            "policy_engine_version": "1.0",
            "internal_policies": {
                category: len(rules) for category, rules in self.policy_rules.items()
            },
            "external_scanners": self.enabled_scanners,
            "total_internal_rules": sum(len(rules) for rules in self.policy_rules.values()),
            "scanning_capabilities": [
                scanner for scanner, available in self.enabled_scanners.items() if available
            ],
        }

    def create_policy_report(
        self, validation_result: Dict[str, Any], terraform_code: str
    ) -> Dict[str, Any]:
        """Create comprehensive policy validation report"""
        violations = validation_result.get("violations", [])

        # Group violations by scanner
        violations_by_scanner = {}
        for violation in violations:
            scanner = violation.get("scanner", "unknown")
            if scanner not in violations_by_scanner:
                violations_by_scanner[scanner] = []
            violations_by_scanner[scanner].append(violation)

        # Group violations by severity
        violations_by_severity = {}
        for violation in violations:
            severity = violation.get("severity", "MEDIUM")
            if severity not in violations_by_severity:
                violations_by_severity[severity] = []
            violations_by_severity[severity].append(violation)

        # Calculate security score (0-100)
        total_violations = len(violations)
        critical_count = len(violations_by_severity.get("CRITICAL", []))
        high_count = len(violations_by_severity.get("HIGH", []))
        medium_count = len(violations_by_severity.get("MEDIUM", []))

        # Security score calculation
        penalty = critical_count * 25 + high_count * 10 + medium_count * 3
        security_score = max(0, 100 - penalty)

        return {
            # Would use datetime.now() in real implementation
            "report_timestamp": "2024-01-01T00:00:00Z",
            "terraform_code_lines": len(terraform_code.splitlines()),
            "validation_result": validation_result.get("valid", False),
            "security_score": security_score,
            "total_violations": total_violations,
            "violations_by_severity": violations_by_severity,
            "violations_by_scanner": violations_by_scanner,
            "scanners_used": validation_result.get("scanners_used", []),
            "recommendations": self._generate_recommendations(violations),
            "compliance_status": self._assess_compliance(violations),
        }

    def _generate_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        critical_violations = [v for v in violations if v.get("severity") == "CRITICAL"]
        if critical_violations:
            recommendations.append(
                "[CRITICAL] URGENT: Address critical security violations before deployment"
            )

        high_violations = [v for v in violations if v.get("severity") == "HIGH"]
        if high_violations:
            recommendations.append(
                "[WARN] HIGH PRIORITY: Review and fix high-severity security issues"
            )

        # Specific recommendations based on violation patterns
        violation_types = [v.get("category", "unknown") for v in violations]

        if "aws_security_groups" in str(violation_types):
            recommendations.append(
                "[SECURE] Review security group rules and apply principle of least privilege"
            )

        if "aws_s3_buckets" in str(violation_types):
            recommendations.append(
                "[S3] Enable S3 security features: versioning, encryption, and access controls"
            )

        if any("hardcoded" in v.get("message", "").lower() for v in violations):
            recommendations.append(
                "[SECRET] Remove hardcoded credentials and use proper secret management"
            )

        return recommendations

    def _assess_compliance(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess compliance with common frameworks"""
        compliance = {
            "SOC2": "COMPLIANT",
            "PCI_DSS": "COMPLIANT",
            "GDPR": "COMPLIANT",
            "HIPAA": "COMPLIANT",
        }

        critical_violations = [v for v in violations if v.get("severity") == "CRITICAL"]
        high_violations = [v for v in violations if v.get("severity") == "HIGH"]

        if critical_violations or len(high_violations) > 3:
            # Mark all as non-compliant if critical issues exist
            for framework in compliance.keys():
                compliance[framework] = "NON_COMPLIANT"

        return compliance

    # === APPEND: AST-based validation utilities (non-destructive) ===


_SOC2_CONTROLS = {"network_segmentation": "Security groups should restrict ingress/egress broadly."}
_HIPAA_CONTROLS = {"https_only": "ALB/CloudFront listeners should enforce HTTPS."}


def _pe_parse_hcl(tf_code: str) -> _Dict[str, _Any]:
    try:
        return _pe_hcl2.load(_PE_StringIO(tf_code))
    except Exception as e:
        return {"__parse_error__": str(e)}


def _pe_open_sg(ast: _Dict[str, _Any]) -> _List[str]:
    f: _List[str] = []
    for block in ast.get("resource", []):
        if "aws_security_group" in block:
            for name, sg in block["aws_security_group"].items():
                ingress = sg.get("ingress", [])
                if not isinstance(ingress, list):
                    ingress = [ingress]
                for rule in ingress:
                    cidrs = (rule.get("cidr_blocks") or []) + (rule.get("ipv6_cidr_blocks") or [])
                    if "0.0.0.0/0" in cidrs or "::/0" in cidrs:
                        f.append(f"Open ingress in security group '{name}'")
    return f


async def validate_terraform_code_ast(tf_code: str) -> _Dict[str, _Any]:
    """
    Additive validator using AST + tfsec/checkov. Does not alter your original policy engine.
    """
    ast = _pe_parse_hcl(tf_code)
    findings = []
    controls = {"soc2": [], "hipaa": []}
    if "__parse_error__" in ast:
        return {
            "success": False,
            "data": {},
            "error": f"HCL parse failed: {ast['__parse_error__']}",
        }

    findings += _pe_open_sg(ast)
    if any("Open ingress" in f for f in findings):
        controls["soc2"].append(_SOC2_CONTROLS["network_segmentation"])
        controls["hipaa"].append(_HIPAA_CONTROLS["https_only"])

    with _pe_secure_tempdir("policy_v2_") as d:
        import os as _os

        with open(_os.path.join(d, "main.tf"), "w", encoding="utf-8") as f:
            f.write(tf_code)
        rc1, out1, err1 = await _pe_run_cmd_async("tfsec", "--format", "json", d)
        rc2, out2, err2 = await _pe_run_cmd_async("checkov", "-d", d, "-o", "json")
        tfsec = _pe_json.loads(out1 or "{}") if rc1 == 0 else {"error": err1}
        checkov = _pe_json.loads(out2 or "{}") if rc2 == 0 else {"error": err2}

    return {
        "success": True,
        "data": {
            "findings": findings,
            "controls": controls,
            "tfsec": tfsec,
            "checkov": checkov,
        },
        "error": "",
    }


# Global policy engine instance
policy_engine = PolicyEngine()
