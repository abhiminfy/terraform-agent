# guardrails.py
import logging
import os
import re
from io import StringIO as _GR_StringIO
from typing import Any, Dict, List, Tuple

import hcl2 as _gr_hcl2

logger = logging.getLogger(__name__)

# Env-driven large-input handling (soft; do not block)
_MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "200000"))


class TerraformGuardrails:
    """Guardrails to prevent hallucination, catch obvious security issues, and avoid noisy failures."""

    def __init__(self):
        # Keywords that indicate infrastructure-related queries
        self.infra_keywords = [
            # Generic
            "terraform",
            "cloud",
            "infrastructure",
            "deploy",
            "provision",
            "create infrastructure",
            "set up",
            "configure",
            "build",
            "server",
            # AWS
            "aws",
            "ec2",
            "s3",
            "rds",
            "vpc",
            "lambda",
            "iam",
            "eks",
            "ecs",
            "cloudfront",
            # Azure
            "azure",
            "azurerm",
            "nsg",
            "storage account",
            "storage container",
            "sql server",
            # GCP
            "gcp",
            "google",
            "gke",
            "gce",
            "compute",
            "gcs",
            "cloud sql",
            "firewall",
        ]

        # Fast regex checks (string-level) across AWS/Azure/GCP
        # Fixed: Removed redundant (?i) flags since we use re.IGNORECASE in the search
        self.dangerous_patterns = [
            # Hardcoded secrets (all clouds)
            (
                r'\b(password|secret[_\- ]?key|access[_\- ]?key|api[_\- ]?key|token)\s*=\s*"[^"]+"',
                "Hardcoded credential detected",
            ),
            # Any/All networks (common)
            (r'\bcidr_blocks\s*=\s*\[\s*"0\.0\.0\.0/0"\s*\]', "Open security group (0.0.0.0/0)"),
            (r'\bipv6_cidr_blocks\s*=\s*\[\s*".*::/0"\s*\]', "Open security group (IPv6 ::/0)"),
            (r"\bpublicly_accessible\s*=\s*true", "Database publicly accessible"),
            (r"\bforce_destroy\s*=\s*true", "Force destroy enabled - data loss risk"),
            # --- AWS S3 quick checks
            (
                r'\bresource\s+"aws_s3_bucket"\s+".*?"[\s\S]*?\bacl\s*=\s*"(public-read|public-read-write)"',
                "S3 bucket ACL is public",
            ),
            (
                r'\bresource\s+"aws_s3_bucket_policy"\s+".*?"[\s\S]*?"Principal"\s*:\s*"\*"',
                "S3 bucket policy may allow public access",
            ),
            # --- Azure quick checks
            # Public containers
            (
                r'\bresource\s+"azurerm_storage_container"\s+".*?"[\s\S]*?\bcontainer_access_type\s*=\s*"(blob|container)"',
                "Azure Storage container is public",
            ),
            # Storage account public blob access
            (
                r'\bresource\s+"azurerm_storage_account"\s+".*?"[\s\S]*?\ballow_blob_public_access\s*=\s*true',
                "Azure Storage account allows blob public access",
            ),
            # NSG rule everywhere (string-level)
            (
                r'\bresource\s+"azurerm_network_security_rule"\s+".*?"[\s\S]*?\b(source_address_prefix|source_address_prefixes)\s*=\s*("(\*|0\.0\.0\.0/0)"|\[.*"0\.0\.0\.0/0".*\])',
                "Azure NSG rule allows from anywhere",
            ),
            # Public network access for DB/servers
            (r"\bpublic_network_access(_enabled)?\s*=\s*true", "Public network access enabled"),
            # --- GCP quick checks
            # GCS public IAM bindings
            (
                r'\bresource\s+"google_storage_bucket_iam_(binding|member|policy)"\s+".*?"[\s\S]*?\bmembers?\s*=\s*\[?[\s\S]*"(allUsers|allAuthenticatedUsers)"',
                "GCS bucket IAM grants public access",
            ),
            # Uniform bucket-level access disabled
            (
                r'\bresource\s+"google_storage_bucket"\s+".*?"[\s\S]*?\buniform_bucket_level_access\s*=\s*false',
                "GCS bucket uniform bucket-level access is disabled",
            ),
            # GCP firewall everywhere (string-level)
            (
                r'\bresource\s+"google_compute_firewall"\s+".*?"[\s\S]*?\bsource_ranges\s*=\s*\[.*"0\.0\.0\.0/0".*\]',
                "GCP firewall allows from anywhere",
            ),
        ]

    # ---------------------------
    # High level intent checks
    # ---------------------------
    def is_infrastructure_request(self, message: str) -> bool:
        if not message:
            return False
        m = message.lower()
        return any(k in m for k in self.infra_keywords)

    # ---------------------------
    # Terraform-ness validation
    # ---------------------------
    def validate_terraform_response(self, response: str) -> Dict[str, Any]:
        if not response:
            return {"valid": False, "reason": "Empty response", "confidence": 0.0}

        rl = response.lower()
        confidence = 0.0

        tf_patterns = [
            r'\bresource\s+"[^"]+"\s+"[^"]+"\s*\{',
            r'\bprovider\s+"[^"]+"\s*\{',
            r'\bvariable\s+"[^"]+"\s*\{',
            r'\boutput\s+"[^"]+"\s*\{',
            r'\bdata\s+"[^"]+"\s+"[^"]+"\s*\{',
            r"\bterraform\s*\{",
            r'\bmodule\s+"[^"]+"\s*\{',
        ]
        pattern_matches = sum(1 for p in tf_patterns if re.search(p, response, re.I | re.M))
        confidence += 0.15 * pattern_matches

        aws = [
            "aws_instance",
            "aws_s3_bucket",
            "aws_vpc",
            "aws_subnet",
            "aws_security_group",
            "aws_rds_instance",
            "aws_lambda_function",
            "aws_lb",
            "aws_cloudfront_distribution",
            "aws_route53_zone",
        ]
        az = [
            "azurerm_storage_account",
            "azurerm_storage_container",
            "azurerm_network_security_group",
            "azurerm_network_security_rule",
            "azurerm_linux_virtual_machine",
            "azurerm_mssql_server",
            "azurerm_mysql_flexible_server",
            "azurerm_postgresql_flexible_server",
        ]
        gcp = [
            "google_compute_firewall",
            "google_compute_instance",
            "google_container_cluster",
            "google_sql_database_instance",
            "google_storage_bucket",
            "google_storage_bucket_iam_binding",
        ]
        cloud_hits = sum(1 for r in aws + az + gcp if r in rl)
        confidence += 0.1 * cloud_hits

        tf_keywords = [
            "terraform",
            "resource",
            "provider",
            "variable",
            "output",
            "data",
            "module",
            "locals",
            "count",
            "for_each",
        ]
        kw_hits = sum(1 for k in tf_keywords if k in rl)
        confidence += 0.05 * kw_hits

        confidence = min(1.0, confidence)
        is_valid = (pattern_matches > 0) or (cloud_hits > 0 and kw_hits > 0) or (confidence > 0.5)
        return {
            "valid": is_valid,
            "confidence": round(confidence, 2),
            "pattern_matches": pattern_matches,
            "cloud_resource_matches": cloud_hits,
            "keyword_matches": kw_hits,
            "reason": (
                "Valid Terraform content detected"
                if is_valid
                else "No valid Terraform patterns found"
            ),
        }

    # ---------------------------
    # Security checks (regex + light AST)
    # ---------------------------
    def check_security_issues(self, terraform_code: str) -> List[Dict[str, Any]]:
        if not terraform_code:
            return []

        issues: List[Dict[str, Any]] = []

        # 1) Fast regex sweep
        for pattern, desc in self.dangerous_patterns:
            try:
                for m in re.finditer(pattern, terraform_code, re.I | re.M | re.S):
                    issues.append(
                        {
                            "type": "security_warning",
                            "description": desc,
                            "line_content": m.group(0)[:300],
                            "severity": self._get_severity(desc),
                        }
                    )
            except re.error as e:
                logger.error(f"Regex error in pattern '{pattern}': {e}")

        # 2) AST sweep for structured checks (AWS + Azure + GCP)
        try:
            ast = _gr_hcl2.load(_GR_StringIO(terraform_code))

            for block in ast.get("resource", []):
                for rtype, items in block.items():
                    if not isinstance(items, list):
                        continue
                    for res in items:
                        for name, body in res.items():
                            if not isinstance(body, dict):
                                continue

                            # ---------- AWS ----------
                            if rtype in ("aws_security_group", "aws_security_group_rule"):
                                for c in self._collect_aws_cidrs(body):
                                    if c == "0.0.0.0/0" or c.endswith("::/0"):
                                        issues.append(
                                            self._issue(
                                                "Security group allows traffic from anywhere",
                                                f"{rtype}.{name} -> {c}",
                                                "HIGH",
                                            )
                                        )
                            if rtype == "aws_s3_bucket":
                                acl = str(body.get("acl", "")).strip().strip('"')
                                if acl in ("public-read", "public-read-write"):
                                    issues.append(
                                        self._issue(
                                            "S3 bucket ACL is public",
                                            f"{rtype}.{name} acl={acl}",
                                            "HIGH",
                                        )
                                    )
                                if (
                                    "versioning" in body
                                    or "server_side_encryption_configuration" in body
                                ):
                                    issues.append(
                                        self._issue(
                                            "S3 inline 'versioning' or 'server_side_encryption_configuration' found. For AWS provider v5, split into separate resources.",
                                            f"{rtype}.{name}",
                                            "MEDIUM",
                                        )
                                    )

                            # ---------- Azure ----------
                            if rtype in ("azurerm_network_security_group",):
                                # embedded 'security_rule' blocks
                                rules = body.get("security_rule", [])
                                for rule in rules if isinstance(rules, list) else [rules]:
                                    if isinstance(rule, dict):
                                        if self._azure_rule_is_open(rule):
                                            issues.append(
                                                self._issue(
                                                    "Azure NSG security rule allows from anywhere",
                                                    f"{rtype}.{name}",
                                                    "HIGH",
                                                )
                                            )
                            if rtype in ("azurerm_network_security_rule",):
                                if self._azure_rule_is_open(body):
                                    issues.append(
                                        self._issue(
                                            "Azure NSG security rule allows from anywhere",
                                            f"{rtype}.{name}",
                                            "HIGH",
                                        )
                                    )
                            if rtype == "azurerm_storage_container":
                                cat = str(body.get("container_access_type", "")).strip('"').lower()
                                if cat in ("blob", "container"):
                                    issues.append(
                                        self._issue(
                                            "Azure Storage container is public",
                                            f"{rtype}.{name} container_access_type={cat}",
                                            "HIGH",
                                        )
                                    )
                            if rtype == "azurerm_storage_account":
                                if str(body.get("allow_blob_public_access", "")).lower() == "true":
                                    issues.append(
                                        self._issue(
                                            "Azure Storage account allows blob public access",
                                            f"{rtype}.{name}",
                                            "HIGH",
                                        )
                                    )
                            # public network toggles are widely used in Azure DB resources
                            if rtype.startswith("azurerm_") and "public_network_access" in {
                                k.lower() for k in body.keys()
                            }:
                                val = str(
                                    body.get("public_network_access")
                                    or body.get("public_network_access_enabled")
                                )
                                if str(val).lower() == "true":
                                    issues.append(
                                        self._issue(
                                            "Public network access enabled",
                                            f"{rtype}.{name}",
                                            "HIGH",
                                        )
                                    )

                            # ---------- GCP ----------
                            if rtype == "google_compute_firewall":
                                sr = body.get("source_ranges") or []
                                sr = sr if isinstance(sr, list) else [sr]
                                sr = [str(x).strip('"') for x in sr]
                                if any(x == "0.0.0.0/0" or x.endswith("::/0") for x in sr):
                                    issues.append(
                                        self._issue(
                                            "GCP firewall allows traffic from anywhere",
                                            f"{rtype}.{name} source_ranges={sr}",
                                            "HIGH",
                                        )
                                    )
                            if rtype == "google_storage_bucket":
                                ublea = body.get("uniform_bucket_level_access")
                                if str(ublea).lower() == "false":
                                    issues.append(
                                        self._issue(
                                            "GCS bucket uniform bucket-level access is disabled",
                                            f"{rtype}.{name}",
                                            "MEDIUM",
                                        )
                                    )
                            if rtype in (
                                "google_storage_bucket_iam_binding",
                                "google_storage_bucket_iam_member",
                                "google_storage_bucket_iam_policy",
                            ):
                                members = body.get("members") or body.get("member")
                                mems = (
                                    members
                                    if isinstance(members, list)
                                    else [members] if members else []
                                )
                                mems = [str(x).strip('"') for x in mems]
                                if any(m in ("allUsers", "allAuthenticatedUsers") for m in mems):
                                    issues.append(
                                        self._issue(
                                            "GCS bucket IAM grants public access",
                                            f"{rtype}.{name} members={mems}",
                                            "HIGH",
                                        )
                                    )
                            if rtype == "google_sql_database_instance":
                                settings = body.get("settings", {})
                                ipcfg = (settings or {}).get("ip_configuration") or (
                                    settings or {}
                                ).get("ipConfiguration")
                                ipcfg = (
                                    ipcfg
                                    if isinstance(ipcfg, dict)
                                    else (ipcfg[0] if isinstance(ipcfg, list) and ipcfg else {})
                                )
                                if isinstance(ipcfg, dict):
                                    auth = ipcfg.get("authorized_networks") or ipcfg.get(
                                        "authorizedNetworks"
                                    )
                                    nets = (
                                        auth if isinstance(auth, list) else [auth] if auth else []
                                    )
                                    nets = [n if isinstance(n, dict) else {} for n in nets]
                                    for n in nets:
                                        v = n.get("value") or n.get("cidr")
                                        if v and (v == "0.0.0.0/0" or str(v).endswith("::/0")):
                                            issues.append(
                                                self._issue(
                                                    "Cloud SQL allows access from anywhere via authorized networks",
                                                    f"{rtype}.{name}",
                                                    "HIGH",
                                                )
                                            )
                                    ipv4 = ipcfg.get("ipv4_enabled") or ipcfg.get("ipv4Enabled")
                                    if str(ipv4).lower() == "true":
                                        issues.append(
                                            self._issue(
                                                "Cloud SQL IPv4 public address enabled",
                                                f"{rtype}.{name}",
                                                "MEDIUM",
                                            )
                                        )
        except Exception as e:
            logger.debug(f"HCL AST parse skipped/failed: {e}")

        return issues

    # ---- helpers for AST checks ----
    def _collect_aws_cidrs(self, body: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for key in ("ingress", "egress"):
            rules = body.get(key, [])
            rules = rules if isinstance(rules, list) else [rules]
            for r in rules:
                if isinstance(r, dict):
                    v4 = r.get("cidr_blocks") or []
                    v6 = r.get("ipv6_cidr_blocks") or []
                    if isinstance(v4, list):
                        out += [str(x).strip('"') for x in v4]
                    if isinstance(v6, list):
                        out += [str(x).strip('"') for x in v6]
        return out

    def _azure_rule_is_open(self, rule: Dict[str, Any]) -> bool:
        # Accept both singular and plural fields; "*" or 0.0.0.0/0 means everywhere
        s_pref = rule.get("source_address_prefix")
        s_prefs = rule.get("source_address_prefixes") or []
        vals = []
        if isinstance(s_pref, str):
            vals.append(s_pref.strip('"'))
        if isinstance(s_prefs, list):
            vals += [str(x).strip('"') for x in s_prefs]
        return any(v in ("*", "0.0.0.0/0") or v.endswith("::/0") for v in vals)

    def _issue(self, desc: str, line: str, sev: str) -> Dict[str, Any]:
        return {
            "type": "security_warning",
            "description": desc,
            "line_content": line,
            "severity": sev,
        }

    def _get_severity(self, description: str) -> str:
        d = (description or "").lower()
        if any(w in d for w in ["secret", "key", "password", "token", "api key"]):
            return "CRITICAL"
        if any(w in d for w in ["public", "0.0.0.0/0", "::/0", "allusers"]):
            return "HIGH"
        if any(w in d for w in ["destroy", "force"]):
            return "MEDIUM"
        return "LOW"

    # ---------------------------
    # Hallucination filter (conservative)
    # ---------------------------
    def filter_hallucinations(self, response: str, user_message: str) -> Tuple[str, List[str]]:
        if not response:
            return response, ["Empty response received"]

        warnings: List[str] = []
        filtered_lines: List[str] = []
        lines = response.split("\n")

        patterns = [
            r"\b(i think|i believe|probably|maybe|might be|could be)\b",
            r"\b(as far as i know|to my knowledge|if i remember)\b",
            r"\b(let me guess|i assume|presumably)\b",
            r"\b(version \d+\.\d+\.\d+ \(which was released)",
            r"\b(according to the latest|recent updates show)\b",
        ]

        for line in lines:
            if any(re.search(p, line, re.IGNORECASE) for p in patterns):
                warnings.append(f"Filtered uncertain statement: {line.strip()[:100]}")
            else:
                filtered_lines.append(line)

        filtered = "\n".join(filtered_lines)
        if len(filtered.strip()) < len(response.strip()) * 0.3:
            warnings.append("Response may contain uncertain information - please verify")
            return response, warnings
        return filtered, warnings

    # ---------------------------
    # Input validation (soft)
    # ---------------------------
    def validate_user_input(self, message: str) -> Dict[str, Any]:
        """Soft validation. Never blocks long inputs; routes spill them to a file."""
        if not message or not message.strip():
            return {
                "valid": False,
                "reason": "Empty message",
                "suggestions": ["Please provide a clear question or request"],
            }

        message = message.strip()
        if len(message) < 3:
            return {
                "valid": False,
                "reason": "Message too short",
                "suggestions": ["Please provide more details about what you need"],
            }

        if len(message) > _MAX_INPUT_CHARS:
            return {
                "valid": True,
                "reason": f"Large input ({len(message)} chars) — backend will process it from file.",
                "suggestions": [],
            }

        harmful_patterns = [
            r"\b(hack|exploit|attack|vulnerability)\b",
            r"\b(delete all|destroy everything|rm\s+-rf)\b",
            r"\b(bypass security|disable protection)\b",
        ]
        for p in harmful_patterns:
            if re.search(p, message, re.IGNORECASE):
                return {
                    "valid": False,
                    "reason": "Potentially harmful request detected",
                    "suggestions": [
                        "Please rephrase your request focusing on legitimate infrastructure needs"
                    ],
                }

        vague = [r"^(hi|hello|hey)$", r"^(help|what|how)$", r"^(do something|make it work)$"]
        for p in vague:
            if re.search(p, message, re.IGNORECASE):
                return {
                    "valid": True,
                    "reason": "Request is too vague",
                    "suggestions": [
                        "Try being more specific, for example:",
                        "- 'Create an EC2 instance with a web server'",
                        "- 'Set up a VPC with public and private subnets'",
                        "- 'Deploy a load balancer for my application'",
                    ],
                }
        return {"valid": True, "reason": "Input appears valid", "suggestions": []}

    # ---------------------------
    # Clarification logic
    # ---------------------------
    def should_ask_for_clarification(self, message: str) -> Tuple[bool, List[str]]:
        if not message:
            return True, ["Please provide a message"]

        m = message.lower()
        questions: List[str] = []

        if self.is_infrastructure_request(message):
            missing: List[str] = []

            # AWS/GCP region or zone hint; Azure region synonym: location
            if not re.search(r"\b(region|location|us-|eu-|ap-|asia-|australia-)", m):
                missing.append("Which region/location would you like to use?")

            if not re.search(r"\b(dev|development|staging|prod|production|test)\b", m):
                missing.append("Is this for development, staging, or production?")

            if re.search(r"\b(ec2|instance|vm|virtual machine)\b", m) and not re.search(
                r"\b(t[2-4]\.|m[4-7]|c[4-7]|r[4-7]|e2-|n2-|n1-|f1-|b1ms|d2s|b2s|standard_)\b", m
            ):
                missing.append(
                    "What instance/VM size do you need? (e.g., t3.micro, e2-medium, Standard_B2s)"
                )

            if len(missing) >= 2:
                return True, missing

        general = [
            r"^(create|make|build|setup|deploy)\s+(something|infrastructure|app|website)$",
            r"^(help me with|show me|explain)\s+(terraform|aws|azure|gcp|cloud)$",
        ]
        for p in general:
            if re.search(p, message, re.IGNORECASE):
                return True, [
                    "Could you be more specific about what you want to create?",
                    "For example: web server, database, load balancer, VPC/VNet, etc.",
                ]
        return False, []

    # ---------------------------
    # Confidence scoring
    # ---------------------------
    def get_confidence_score(self, response: str, user_message: str) -> float:
        if not response or not user_message:
            return 0.0

        score = 0.0
        if self.is_infrastructure_request(user_message):
            tfv = self.validate_terraform_response(response)
            score += tfv["confidence"] * 0.4

        rl = len(response.strip())
        score += 0.2 if 100 <= rl <= 3000 else (0.05 if rl < 100 else 0.0)

        if "```" in response:
            score += 0.15
        if any(m in response for m in ["resource", "provider", "variable"]):
            score += 0.15

        uncertainty = [
            r"\b(i think|maybe|might|could be|not sure)\b",
            r"\b(probably|possibly|perhaps|seems like)\b",
        ]
        score -= sum(1 for p in uncertainty if re.search(p, response, re.IGNORECASE)) * 0.1

        return max(0.0, min(1.0, score))


# Lightweight "does it parse" check, reused by AST callers elsewhere
def security_ast_checks_v2(tf_code: str):
    try:
        _gr_hcl2.load(_GR_StringIO(tf_code))
        return {"success": True, "data": {"ast_checks": "ok"}, "error": ""}
    except Exception as e:
        return {"success": False, "data": {}, "error": f"HCL parse error: {e}"}
