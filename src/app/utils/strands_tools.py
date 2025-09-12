import abc as _abc
import json
import logging
import re
import subprocess
import tempfile
from datetime import datetime
from io import StringIO as _ST_StringIO
from pathlib import Path
from typing import Any, Dict, List

import hcl2 as _st_hcl2

from src.app.core.config import Settings as _ST_Settings
from src.app.utils.utils import run_cmd_async as _st_run_cmd_async
from src.app.utils.utils import secure_tempdir as _st_secure_tempdir

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalculatorTool:
    """Basic calculator operations for cost calculations and mathematical operations"""

    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @staticmethod
    def subtract(a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b

    @staticmethod
    def divide(a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b

    @staticmethod
    def percentage(value: float, percentage: float) -> float:
        """Calculate percentage of a value"""
        return (value * percentage) / 100

    @staticmethod
    def compound_interest(principal: float, rate: float, time: float, n: int = 12) -> float:
        """Calculate compound interest for cost projections"""
        return principal * (1 + rate / n) ** (n * time)


class TerraformValidatorTool:
    """Enhanced Terraform code validation and formatting tools with better security and resilience"""

    @staticmethod
    def validate_terraform_syntax(terraform_code: str, timeout: int = 30) -> Dict[str, Any]:
        """Validate Terraform code syntax with timeout and graceful error handling"""
        if not terraform_code or not terraform_code.strip():
            return {
                "valid": False,
                "output": "",
                "errors": "Empty Terraform code provided",
            }

        try:
            # Create a temporary directory and file
            with tempfile.TemporaryDirectory() as temp_dir:
                tf_file = Path(temp_dir) / "main.tf"
                tf_file.write_text(terraform_code, encoding="utf-8")

                # Run terraform init with timeout
                try:
                    init_result = subprocess.run(
                        ["terraform", "init", "-input=false"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return {
                        "valid": False,
                        "output": "",
                        "errors": f"Terraform init timed out after {timeout} seconds",
                    }

                if init_result.returncode != 0:
                    return {
                        "valid": False,
                        "output": init_result.stdout,
                        "errors": f"Terraform init failed: {init_result.stderr}",
                    }

                # Run terraform validate with timeout
                try:
                    result = subprocess.run(
                        [
                            "terraform",
                            "validate",
                            "-json",
                        ],  # Use JSON output for better parsing
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return {
                        "valid": False,
                        "output": "",
                        "errors": f"Terraform validate timed out after {timeout} seconds",
                    }

                # Parse JSON output if available
                if result.returncode == 0:
                    try:
                        if result.stdout:
                            json_output = json.loads(result.stdout)
                            return {
                                "valid": json_output.get("valid", True),
                                "output": "Terraform configuration is valid",
                                "errors": None,
                                "json_output": json_output,
                            }
                        else:
                            return {
                                "valid": True,
                                "output": "Terraform configuration is valid",
                                "errors": None,
                            }
                    except json.JSONDecodeError:
                        # Fallback to text output
                        return {
                            "valid": True,
                            "output": result.stdout or "Terraform configuration is valid",
                            "errors": None,
                        }
                else:
                    # Parse error output
                    error_msg = result.stderr
                    try:
                        if result.stdout:
                            json_output = json.loads(result.stdout)
                            if "diagnostics" in json_output:
                                error_details = []
                                for diag in json_output["diagnostics"]:
                                    summary = diag.get("summary", "Unknown error")
                                    detail = diag.get("detail", "")
                                    error_details.append(
                                        f"{summary}: {detail}" if detail else summary
                                    )
                                error_msg = "\n".join(error_details)
                    except json.JSONDecodeError:
                        pass

                    return {
                        "valid": False,
                        "output": result.stdout,
                        "errors": error_msg,
                    }

        except FileNotFoundError:
            return {
                "valid": False,
                "output": "",
                "errors": "Terraform CLI not found. Please install Terraform.",
                "tool_missing": True,
            }
        except Exception as e:
            return {
                "valid": False,
                "output": "",
                "errors": f"Validation failed: {str(e)}",
            }

    @staticmethod
    def format_terraform_code(terraform_code: str, timeout: int = 15) -> str:
        """Format Terraform code using terraform fmt with timeout"""
        if not terraform_code or not terraform_code.strip():
            return terraform_code

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                tf_file = Path(temp_dir) / "main.tf"
                tf_file.write_text(terraform_code, encoding="utf-8")

                # Run terraform fmt with timeout
                try:
                    subprocess.run(
                        ["terraform", "fmt"],
                        cwd=temp_dir,
                        check=True,
                        capture_output=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Terraform fmt timed out after {timeout} seconds, returning original"
                    )
                    return terraform_code

                # Return formatted code
                return tf_file.read_text()

        except FileNotFoundError:
            logger.warning("Terraform CLI not found, returning unformatted code")
            return terraform_code
        except Exception as e:
            logger.error(f"Failed to format Terraform code: {e}")
            return terraform_code  # Return original if formatting fails

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize user/model text to plain ASCII-safe characters."""
        if not text:
            return text
        import unicodedata

        t = unicodedata.normalize("NFKC", text)
        table = {
            0x2018: "'",
            0x2019: "'",
            0x201C: '"',
            0x201D: '"',
            0x2013: "-",
            0x2014: "-",
            0x00A0: " ",
        }
        return t.translate(table)

    @staticmethod
    def autofix_common_hcl(code: str, region: str | None = None):
        """
        Fix frequent generation mistakes:
        - gp2 -> gp3 in AL2 AMI filter
        - aws_default_vpc -> data aws_vpc.default { default = true }
        - remove CDK tag-based subnet filters
        - make SG name references consistent
        - set provider region if present/requested
        Returns (fixed_code, [notes]).
        """
        import re

        notes = []
        if not code:
            return code, notes

        # gp2 -> gp3 in AL2 filter
        c2 = re.sub(r'("amzn2-ami-hvm-[^"]*?)-gp2(")', r"\1-gp3\2", code)
        if c2 != code:
            notes.append("AMI filter: gp2 → gp3")
            code = c2

        # use aws_vpc.default instead of aws_default_vpc
        c2 = re.sub(
            r'data\s+"aws_default_vpc"\s+"default"\s*{[^}]*}',
            'data "aws_vpc" "default" { default = true }',
            code,
            flags=re.S,
        )
        if c2 != code:
            notes.append("Replaced aws_default_vpc with aws_vpc.default")
            code = c2
        code = re.sub(r"data\.aws_default_vpc\.default", "data.aws_vpc.default", code)

        # remove CDK tag-based subnet filter; keep vpc-id filter only
        code = re.sub(
            r'filter\s*{\s*name\s*=\s*"tag:aws-cdk:subnet-type"[^}]*}',
            "",
            code,
        )

        # ensure SG reference matches the declared SG name
        m = re.search(r'resource\s+"aws_security_group"\s+"(\w+)"', code)
        if m:
            sg = m.group(1)
            code = re.sub(
                r"aws_security_group\.\w+\.id",
                f"aws_security_group.{sg}.id",
                code,
            )

        # apply explicit provider region if provided
        if region:
            code = re.sub(
                r'provider\s+"aws"\s*{[^}]*}',
                f'provider "aws" {{ region = "{region}" }}',
                code,
                flags=re.S,
            )

        return code, notes

    @staticmethod
    def run_fmt_validate_plan_from_code(terraform_code: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Write TF code to a temp dir and run:
          - terraform init
          - terraform fmt -check
          - terraform validate
          - terraform plan -no-color -detailed-exitcode
        Returns plan text and exit codes. Does NOT apply.
        """
        import re
        import subprocess
        import tempfile
        from pathlib import Path

        out: Dict[str, Any] = {
            "fmt_ok": False,
            "validate_ok": False,
            "plan_exit_code": None,  # 0=no changes, 1=error, 2=changes
            "plan_output": "",
            "errors": [],
        }

        if not terraform_code or not terraform_code.strip():
            out["errors"].append("empty terraform code")
            return out

        def run(cmd, cwd):
            try:
                p = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
            except FileNotFoundError:
                return {
                    "rc": 127,
                    "out": "",
                    "err": f"tool not found: {cmd[0]}",
                }
            except subprocess.TimeoutExpired:
                return {
                    "rc": 124,
                    "out": "",
                    "err": f"timeout after {timeout}s: {' '.join(cmd)}",
                }

        with tempfile.TemporaryDirectory() as work:
            Path(work, "main.tf").write_text(terraform_code, encoding="utf-8", errors="ignore")

            init = run(["terraform", "init", "-input=false", "-no-color"], work)
            fmt = run(["terraform", "fmt", "-check", "-no-color"], work)
            val = run(["terraform", "validate", "-no-color"], work)
            plan = run(
                [
                    "terraform",
                    "plan",
                    "-detailed-exitcode",
                    "-no-color",
                    "-input=false",
                ],
                work,
            )

            out["fmt_ok"] = fmt["rc"] == 0
            out["validate_ok"] = val["rc"] == 0
            out["plan_exit_code"] = plan["rc"]
            out["plan_output"] = (plan["out"] or "") + (plan["err"] or "")

            if init["rc"] not in (0,):
                out["errors"].append(f"init: {init['err']}")
            if fmt["rc"] not in (0,):
                out["errors"].append(f"fmt: {fmt['err']}")
            if val["rc"] not in (0,):
                out["errors"].append(f"validate: {val['err']}")
            if plan["rc"] not in (0, 2):
                out["errors"].append(f"plan: {plan['err']}")

            # Try to extract the one-line summary
            m = re.search(
                r"Plan:\s+\d+\s+to\s+add,\s+\d+\s+to\s+change,\s+\d+\s+to\s+destroy",
                out["plan_output"],
            )
            if m:
                out["plan_summary"] = m.group(0)
            else:
                out["plan_summary"] = ""

            return out


class AWSPricingTool:
    """AWS pricing calculator and cost estimation tools"""

    # Updated AWS pricing data (simplified - in production, use AWS Pricing
    # API)
    PRICING_DATA = {
        "ec2": {
            "t2.nano": {"hourly": 0.0058, "monthly": 4.23},
            "t2.micro": {"hourly": 0.0116, "monthly": 8.47},
            "t2.small": {"hourly": 0.023, "monthly": 16.79},
            "t2.medium": {"hourly": 0.0464, "monthly": 33.87},
            "t3.micro": {"hourly": 0.0104, "monthly": 7.59},
            "t3.small": {"hourly": 0.0208, "monthly": 15.18},
            "t3.medium": {"hourly": 0.0416, "monthly": 30.37},
            "t3.large": {"hourly": 0.0832, "monthly": 60.74},
            "m5.large": {"hourly": 0.096, "monthly": 70.08},
            "m5.xlarge": {"hourly": 0.192, "monthly": 140.16},
            "c5.large": {"hourly": 0.085, "monthly": 62.05},
            "r5.large": {"hourly": 0.126, "monthly": 91.98},
        },
        "rds": {
            "db.t3.micro": {"hourly": 0.017, "monthly": 12.41},
            "db.t3.small": {"hourly": 0.034, "monthly": 24.82},
            "db.t3.medium": {"hourly": 0.068, "monthly": 49.64},
            "db.t3.large": {"hourly": 0.136, "monthly": 99.28},
            "db.m5.large": {"hourly": 0.192, "monthly": 140.16},
        },
        "s3": {
            "standard": {"per_gb_month": 0.023},
            "standard_ia": {"per_gb_month": 0.0125},
            "glacier": {"per_gb_month": 0.004},
            "deep_archive": {"per_gb_month": 0.00099},
        },
        "vpc": {
            "nat_gateway": {"hourly": 0.045, "monthly": 32.85},
            "data_processing_per_gb": 0.045,
            "elastic_ip": {"monthly": 3.65},  # when not associated with running instance
        },
        "load_balancer": {
            "application": {"hourly": 0.0225, "monthly": 16.43},
            "network": {"hourly": 0.0225, "monthly": 16.43},
            "classic": {"hourly": 0.025, "monthly": 18.25},
        },
    }

    @staticmethod
    def estimate_ec2_cost(
        instance_type: str, count: int = 1, hours_per_month: int = 730
    ) -> Dict[str, Any]:
        """Estimate EC2 instance costs"""
        if instance_type not in AWSPricingTool.PRICING_DATA["ec2"]:
            return {
                "error": f"Unknown instance type: {instance_type}. Available types: {
                    list(
                        AWSPricingTool.PRICING_DATA['ec2'].keys())}"
            }

        hourly_rate = AWSPricingTool.PRICING_DATA["ec2"][instance_type]["hourly"]
        monthly_cost = hourly_rate * hours_per_month * count

        return {
            "instance_type": instance_type,
            "count": count,
            "hourly_rate": hourly_rate,
            "hours_per_month": hours_per_month,
            "monthly_cost": round(monthly_cost, 2),
            "yearly_cost": round(monthly_cost * 12, 2),
        }

    @staticmethod
    def estimate_rds_cost(
        db_instance_class: str, storage_gb: int = 20, storage_type: str = "gp2"
    ) -> Dict[str, Any]:
        """Estimate RDS costs"""
        if db_instance_class not in AWSPricingTool.PRICING_DATA["rds"]:
            return {
                "error": f"Unknown DB instance class: {db_instance_class}. Available types: {
                    list(
                        AWSPricingTool.PRICING_DATA['rds'].keys())}"
            }

        hourly_rate = AWSPricingTool.PRICING_DATA["rds"][db_instance_class]["hourly"]
        monthly_instance_cost = hourly_rate * 730

        # Storage pricing
        storage_rates = {"gp2": 0.115, "gp3": 0.092, "io1": 0.125}
        storage_rate = storage_rates.get(storage_type, 0.115)
        monthly_storage_cost = storage_gb * storage_rate

        total_monthly = monthly_instance_cost + monthly_storage_cost

        return {
            "db_instance_class": db_instance_class,
            "storage_gb": storage_gb,
            "storage_type": storage_type,
            "monthly_instance_cost": round(monthly_instance_cost, 2),
            "monthly_storage_cost": round(monthly_storage_cost, 2),
            "total_monthly_cost": round(total_monthly, 2),
            "yearly_cost": round(total_monthly * 12, 2),
        }

    @staticmethod
    def estimate_s3_cost(
        storage_gb: int,
        storage_class: str = "standard",
        requests_per_month: int = 0,
    ) -> Dict[str, Any]:
        """Estimate S3 storage costs"""
        if storage_class not in AWSPricingTool.PRICING_DATA["s3"]:
            return {
                "error": f"Unknown storage class: {storage_class}. Available types: {
                    list(
                        AWSPricingTool.PRICING_DATA['s3'].keys())}"
            }

        rate_per_gb = AWSPricingTool.PRICING_DATA["s3"][storage_class]["per_gb_month"]
        monthly_storage_cost = storage_gb * rate_per_gb

        # Request pricing (simplified)
        request_cost = (requests_per_month / 1000) * 0.0004  # $0.0004 per 1,000 requests

        total_monthly = monthly_storage_cost + request_cost

        return {
            "storage_gb": storage_gb,
            "storage_class": storage_class,
            "requests_per_month": requests_per_month,
            "monthly_storage_cost": round(monthly_storage_cost, 2),
            "monthly_request_cost": round(request_cost, 2),
            "total_monthly_cost": round(total_monthly, 2),
            "yearly_cost": round(total_monthly * 12, 2),
        }

    @staticmethod
    def estimate_vpc_cost(
        nat_gateways: int = 0, elastic_ips: int = 0, data_transfer_gb: int = 0
    ) -> Dict[str, Any]:
        """Estimate VPC-related costs"""
        nat_cost = nat_gateways * AWSPricingTool.PRICING_DATA["vpc"]["nat_gateway"]["monthly"]
        eip_cost = elastic_ips * AWSPricingTool.PRICING_DATA["vpc"]["elastic_ip"]["monthly"]
        data_cost = data_transfer_gb * AWSPricingTool.PRICING_DATA["vpc"]["data_processing_per_gb"]

        total_monthly = nat_cost + eip_cost + data_cost

        return {
            "nat_gateways": nat_gateways,
            "elastic_ips": elastic_ips,
            "data_transfer_gb": data_transfer_gb,
            "nat_gateway_cost": round(nat_cost, 2),
            "elastic_ip_cost": round(eip_cost, 2),
            "data_transfer_cost": round(data_cost, 2),
            "total_monthly_cost": round(total_monthly, 2),
            "yearly_cost": round(total_monthly * 12, 2),
        }


class GitIntegrationTool:
    """Enhanced Git and GitHub integration tools with better error handling"""

    @staticmethod
    def initialize_git_repo(timeout: int = 15) -> Dict[str, Any]:
        """Initialize a git repository with timeout"""
        try:
            if not Path(".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "init"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    logger.info("Git repository initialized")
                    return {
                        "success": True,
                        "message": "Git repository initialized",
                        "output": result.stdout,
                    }
                except subprocess.TimeoutExpired:
                    return {
                        "success": False,
                        "error": f"Git init timed out after {timeout} seconds",
                    }
            else:
                return {
                    "success": True,
                    "message": "Git repository already exists",
                }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Failed to initialize git: {e.stderr or str(e)}",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git not found. Please install Git.",
                "tool_missing": True,
            }

    @staticmethod
    def create_gitignore(project_type: str = "terraform") -> Dict[str, Any]:
        """Create a comprehensive .gitignore file for different project types"""

        gitignore_templates = {
            "terraform": """# Local .terraform directories
**/.terraform/*

# .tfstate files
*.tfstate
*.tfstate.*

# Crash log files
crash.log
crash.*.log

# Exclude all .tfvars files, which are likely to contain sensitive data
*.tfvars
*.tfvars.json

# Ignore override files as they are usually used to override resources locally
override.tf
override.tf.json
*_override.tf
*_override.tf.json

# Include override files you do wish to add to version control using negated pattern
# !example_override.tf

# Include tfplan files to ignore the plan output of command: terraform plan -out=tfplan
*tfplan*

# Ignore CLI configuration files
.terraformrc
terraform.rc""",
            "general": """# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Python files
__pycache__/
*.pyc
*.pyo
*.pyd
env/
venv/
.env
.env.local

# Node.js files
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Log files
*.log

# Temporary files
*.tmp
*.temp""",
        }

        terraform_ignore = gitignore_templates.get("terraform", "")
        general_ignore = gitignore_templates.get("general", "")

        gitignore_content = f"{terraform_ignore}\n\n{general_ignore}"

        try:
            Path(".gitignore").write_text(gitignore_content.strip(), encoding="utf-8")
            return {
                "success": True,
                "message": f".gitignore created successfully for {project_type} project",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create .gitignore: {str(e)}",
            }

    @staticmethod
    def commit_changes(message: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Standard commit changes method for backward compatibility"""
        return GitIntegrationTool.commit_changes_with_validation(message, timeout)

    @staticmethod
    def commit_changes_with_validation(message: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Commit changes with pre-commit validation"""
        if not message:
            message = f"Terraform code update - {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        try:
            # Check if git is initialized
            if not Path(".git").exists():
                return {
                    "success": False,
                    "error": "Git repository not initialized. Run git init first.",
                }

            # Validate Terraform files before committing
            tf_files = list(Path(".").glob("*.tf"))
            if tf_files:
                logger.info(f"Validating {len(tf_files)} Terraform files before commit")

                validation_errors = []
                for tf_file in tf_files:
                    try:
                        content = tf_file.read_text()
                        # Quick syntax check
                        if content.strip():
                            # Basic validation - check for unclosed braces
                            open_braces = content.count("{")
                            close_braces = content.count("}")
                            if open_braces != close_braces:
                                validation_errors.append(f"{tf_file.name}: Mismatched braces")
                    except Exception as e:
                        validation_errors.append(f"{tf_file.name}: {str(e)}")

                if validation_errors:
                    return {
                        "success": False,
                        "error": f"Pre-commit validation failed: {'; '.join(validation_errors)}",
                    }

            # Add all files
            try:
                subprocess.run(
                    ["git", "add", "."],
                    check=True,
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Git add timed out after {timeout} seconds",
                }

            # Check if there are changes to commit
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Git status check timed out after {timeout} seconds",
                }

            if not result.stdout.strip():
                return {"success": True, "message": "No changes to commit"}

            # Commit changes
            try:
                commit_result = subprocess.run(
                    ["git", "commit", "-m", message],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return {
                    "success": True,
                    "message": f"Changes committed: {message}",
                    "output": commit_result.stdout,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Git commit timed out after {timeout} seconds",
                }

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            return {
                "success": False,
                "error": f"Failed to commit changes: {error_msg}",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git not found. Please install Git.",
                "tool_missing": True,
            }

    @staticmethod
    def push_to_github(repo_url: str, branch: str = "main", timeout: int = 60) -> Dict[str, Any]:
        """Push changes to GitHub repository with timeout"""
        try:
            # Set remote origin if not exists
            try:
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    check=True,
                    capture_output=True,
                    timeout=timeout // 4,
                )
            except subprocess.CalledProcessError:
                # Remote might already exist, try to set URL
                subprocess.run(
                    ["git", "remote", "set-url", "origin", repo_url],
                    check=True,
                    timeout=timeout // 4,
                )
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Remote setup timed out"}

            # Push to remote
            try:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", branch],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return {
                    "success": True,
                    "message": f"Successfully pushed to GitHub: {repo_url}",
                    "output": push_result.stdout,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Git push timed out after {timeout} seconds",
                }

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            return {
                "success": False,
                "error": f"Failed to push to GitHub: {error_msg}",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git not found. Please install Git.",
                "tool_missing": True,
            }


class InfrastructureAnalyzer:
    """Enhanced analyzer with comprehensive security checks and better cost estimation"""

    @staticmethod
    def analyze_terraform_resources(terraform_code: str) -> Dict[str, Any]:
        """Enhanced analysis with comprehensive security checks"""
        if not terraform_code or not terraform_code.strip():
            return {
                "error": "Empty Terraform code provided",
                "resources": [],
                "providers": [],
                "variables": [],
                "outputs": [],
            }

        analysis = {
            "resources": [],
            "providers": [],
            "variables": [],
            "outputs": [],
            "estimated_monthly_cost": 0,
            "security_concerns": [],
            "optimization_suggestions": [],
            "resource_count": 0,
            "compliance_checks": [],
            "blast_radius_score": 0,
        }

        try:
            # Extract resources with more comprehensive patterns
            resource_patterns = [
                r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{',
                r'module\s+"([^"]+)"\s*\{',
                r'data\s+"([^"]+)"\s+"([^"]+)"\s*\{',
            ]

            all_resources = []
            for pattern in resource_patterns:
                matches = re.findall(pattern, terraform_code, re.MULTILINE)
                all_resources.extend(matches)

            for match in all_resources:
                if len(match) == 2:
                    resource_type, resource_name = match
                    analysis["resources"].append({"type": resource_type, "name": resource_name})
                elif len(match) == 1:  # Module case
                    analysis["resources"].append({"type": "module", "name": match[0]})

            analysis["resource_count"] = len(analysis["resources"])

            # Extract other components
            provider_pattern = r'provider\s+"([^"]+)"\s*\{'
            providers = re.findall(provider_pattern, terraform_code, re.MULTILINE)
            analysis["providers"] = list(set(providers))

            variable_pattern = r'variable\s+"([^"]+)"\s*\{'
            variables = re.findall(variable_pattern, terraform_code, re.MULTILINE)
            analysis["variables"] = variables

            output_pattern = r'output\s+"([^"]+)"\s*\{'
            outputs = re.findall(output_pattern, terraform_code, re.MULTILINE)
            analysis["outputs"] = outputs

            # Enhanced security analysis
            security_checks = [
                # Hardcoded credentials
                (
                    r'password\s*=\s*"[^"]*"',
                    "🔐 Hardcoded password detected - use variables or AWS Secrets Manager",
                    "HIGH",
                ),
                (
                    r'secret\s*=\s*"[^"]*"',
                    "🔐 Hardcoded secret detected - use AWS Secrets Manager",
                    "HIGH",
                ),
                (
                    r'access_key\s*=\s*"[^"]*"',
                    "🔐 Hardcoded access key - use IAM roles instead",
                    "CRITICAL",
                ),
                (
                    r'secret_key\s*=\s*"[^"]*"',
                    "🔐 Hardcoded secret key - use IAM roles instead",
                    "CRITICAL",
                ),
                # Network security
                (
                    r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                    "🌐 Open security group rule (0.0.0.0/0) - restrict access",
                    "HIGH",
                ),
                (
                    r"from_port\s*=\s*0.*to_port\s*=\s*65535",
                    "🌐 Security group allows all ports - restrict port range",
                    "MEDIUM",
                ),
                (
                    r'protocol\s*=\s*"-1"',
                    "🌐 Security group allows all protocols - be more specific",
                    "MEDIUM",
                ),
                # Database security
                (
                    r"publicly_accessible\s*=\s*true",
                    "🗄️ Database publicly accessible - consider private subnets",
                    "HIGH",
                ),
                (
                    r"skip_final_snapshot\s*=\s*true",
                    "🗄️ Database skip final snapshot - data loss risk",
                    "MEDIUM",
                ),
                (
                    r"deletion_protection\s*=\s*false",
                    "🗄️ Deletion protection disabled - enable for production",
                    "HIGH",
                ),
                # Storage security
                (
                    r"versioning\s*\{\s*enabled\s*=\s*false",
                    "🪣 S3 versioning disabled - enable for data protection",
                    "MEDIUM",
                ),
                (
                    r"force_destroy\s*=\s*true",
                    "🪣 S3 force destroy enabled - data loss risk",
                    "HIGH",
                ),
                (
                    r"block_public_acls\s*=\s*false",
                    "🪣 S3 public ACLs allowed - security risk",
                    "HIGH",
                ),
                # Encryption
                (
                    r"encrypted\s*=\s*false",
                    "🔒 Encryption disabled - enable encryption at rest",
                    "HIGH",
                ),
                (
                    r"server_side_encryption_configuration\s*\{\s*\}",
                    "🔒 S3 encryption not configured",
                    "MEDIUM",
                ),
                # Instance security
                (
                    r"associate_public_ip_address\s*=\s*true",
                    "🖥️ Instance gets public IP - consider private subnets",
                    "MEDIUM",
                ),
                (
                    r"source_dest_check\s*=\s*false",
                    "🖥️ Source/dest check disabled - security risk",
                    "MEDIUM",
                ),
                # IAM security
                (
                    r'Effect"\s*:\s*"Allow".*"Resource"\s*:\s*"\*"',
                    "👤 IAM policy allows all resources - use least privilege",
                    "HIGH",
                ),
                (
                    r'"Action"\s*:\s*"\*"',
                    "👤 IAM policy allows all actions - use least privilege",
                    "CRITICAL",
                ),
            ]

            for pattern, message, severity in security_checks:
                if re.search(pattern, terraform_code, re.IGNORECASE | re.DOTALL):
                    analysis["security_concerns"].append(
                        {
                            "message": message,
                            "severity": severity,
                            "category": message.split()[0],  # Extract emoji/category
                        }
                    )

            # Calculate blast radius score (0-100, higher = more dangerous)
            blast_radius = 0
            critical_count = len(
                [c for c in analysis["security_concerns"] if c.get("severity") == "CRITICAL"]
            )
            high_count = len(
                [c for c in analysis["security_concerns"] if c.get("severity") == "HIGH"]
            )
            medium_count = len(
                [c for c in analysis["security_concerns"] if c.get("severity") == "MEDIUM"]
            )

            blast_radius = min(100, critical_count * 30 + high_count * 15 + medium_count * 5)
            analysis["blast_radius_score"] = blast_radius

            # Compliance checks for common frameworks
            compliance_checks = []

            # Check for SOC 2 Type II compliance indicators
            if not any(
                "encryption" in concern["message"].lower()
                for concern in analysis["security_concerns"]
            ):
                compliance_checks.append("✅ Encryption practices look compliant")
            else:
                compliance_checks.append("❌ SOC 2: Encryption issues detected")

            # Check for PCI DSS indicators (if payment-related resources)
            if any(
                res["type"] in ["aws_rds_instance", "aws_db_instance"]
                for res in analysis["resources"]
            ):
                if any(
                    "publicly_accessible" in concern["message"]
                    for concern in analysis["security_concerns"]
                ):
                    compliance_checks.append("❌ PCI DSS: Database publicly accessible")
                else:
                    compliance_checks.append("✅ PCI DSS: Database access controls look good")

            analysis["compliance_checks"] = compliance_checks

            # Enhanced cost estimation
            analysis["estimated_monthly_cost"] = InfrastructureAnalyzer._estimate_enhanced_cost(
                analysis["resources"], terraform_code
            )

            # Enhanced optimization suggestions
            analysis["optimization_suggestions"] = (
                InfrastructureAnalyzer._generate_optimization_suggestions(
                    terraform_code, analysis["resources"]
                )
            )

        except Exception as e:
            analysis["error"] = f"Analysis failed: {str(e)}"

        return analysis

    @staticmethod
    def _estimate_enhanced_cost(resources: List[dict], terraform_code: str) -> float:
        """Enhanced cost estimation with region and usage pattern awareness"""
        total_cost = 0

        # Enhanced cost mapping with different tiers
        cost_mapping = {
            "aws_instance": {
                "base": 30,
                "patterns": [
                    (r"t2\.micro", 8.5),
                    (r"t2\.small", 17),
                    (r"t3\.micro", 7.5),
                    (r"t3\.small", 15),
                    (r"t3\.medium", 30),
                    (r"t3\.large", 60),
                    (r"m5\.large", 70),
                    (r"m5\.xlarge", 140),
                    (r"c5\.large", 65),
                    (r"r5\.large", 90),
                ],
            },
            "aws_rds_instance": {
                "base": 50,
                "patterns": [
                    (r"db\.t3\.micro", 12),
                    (r"db\.t3\.small", 25),
                    (r"db\.t3\.medium", 50),
                    (r"db\.t3\.large", 100),
                    (r"db\.m5\.large", 140),
                    (r"multi_az\s*=\s*true", 2.0),  # Multiplier for Multi-AZ
                ],
            },
            "aws_s3_bucket": 5,
            "aws_nat_gateway": 33,
            "aws_lb": 16,
            "aws_application_load_balancer": 18,
            "aws_network_load_balancer": 16,
            "aws_elasticache_cluster": 25,
            "aws_elasticsearch_domain": 45,
            "aws_eks_cluster": 73,  # EKS cluster cost
            "aws_lambda_function": 2,  # Estimated based on moderate usage
            "aws_cloudfront_distribution": 8,
            "aws_route53_zone": 0.5,
            "aws_vpc_endpoint": 7.5,
        }

        for resource in resources:
            resource_type = resource.get("type", "")
            base_cost = 0

            if resource_type in cost_mapping:
                cost_info = cost_mapping[resource_type]
                if isinstance(cost_info, dict):
                    base_cost = cost_info["base"]
                    # Check for specific patterns in the terraform code
                    for pattern, cost_adjustment in cost_info.get("patterns", []):
                        if re.search(pattern, terraform_code, re.IGNORECASE):
                            if cost_adjustment > 10:  # Absolute value
                                base_cost = cost_adjustment
                            else:  # Multiplier
                                base_cost *= cost_adjustment
                            break
                else:
                    base_cost = cost_info

            total_cost += base_cost

        # Regional cost adjustments
        region_multipliers = {
            r"us-east-1": 1.0,
            r"us-west-2": 1.05,
            r"eu-west-1": 1.1,
            r"ap-southeast-1": 1.15,
            r"ap-northeast-1": 1.2,
        }

        for region_pattern, multiplier in region_multipliers.items():
            if re.search(f"region.*{region_pattern}", terraform_code, re.IGNORECASE):
                total_cost *= multiplier
                break

        return round(total_cost, 2)

    @staticmethod
    def _estimate_basic_cost(resources: List[tuple]) -> float:
        """Basic cost estimation for common resources (backward compatibility)"""
        total_cost = 0

        cost_mapping = {
            "aws_instance": 30,  # Average t3.small
            "aws_rds_instance": 50,  # Average db.t3.small
            "aws_s3_bucket": 5,  # Basic usage
            "aws_nat_gateway": 33,
            "aws_lb": 16,
            "aws_elasticache_cluster": 25,
        }

        for resource_type, _ in resources:
            cost = cost_mapping.get(resource_type, 0)
            total_cost += cost

        return round(total_cost, 2)

    @staticmethod
    def _generate_optimization_suggestions(terraform_code: str, resources: List[dict]) -> List[str]:
        """Generate context-aware optimization suggestions"""
        suggestions = []

        # Version pinning suggestions
        if "required_version" not in terraform_code:
            suggestions.append("📌 Pin Terraform version: add required_version constraint")

        if "required_providers" not in terraform_code:
            suggestions.append("📌 Pin provider versions: add required_providers block")

        # Tagging suggestions
        if "tags" not in terraform_code and any(
            res["type"].startswith("aws_") for res in resources
        ):
            suggestions.append("🏷️ Add consistent resource tags for cost tracking and management")

        # State management suggestions
        if "backend" not in terraform_code:
            suggestions.append(
                "💾 Configure remote state backend (S3 + DynamoDB) for team collaboration"
            )

        # Security suggestions based on resource types
        db_resources = [r for r in resources if "db" in r["type"] or "rds" in r["type"]]
        if db_resources and "subnet_group" not in terraform_code:
            suggestions.append("🔒 Use DB subnet groups to isolate database instances")

        # Cost optimization suggestions
        instance_count = len([r for r in resources if r["type"] == "aws_instance"])
        if instance_count > 3:
            suggestions.append("💰 Consider using Auto Scaling Groups for better cost management")

        # High availability suggestions
        if (
            "aws_instance" in [r["type"] for r in resources]
            and "availability_zone" not in terraform_code
        ):
            suggestions.append("🌍 Distribute resources across multiple AZs for high availability")

        # Monitoring suggestions
        if len(resources) > 5 and "cloudwatch" not in terraform_code.lower():
            suggestions.append("📊 Add CloudWatch monitoring and alerting for your infrastructure")

        # Module suggestions for complex deployments
        if len(resources) > 10 and "module" not in terraform_code:
            suggestions.append("🧩 Consider organizing code into reusable modules")

        return suggestions

    @staticmethod
    def suggest_best_practices(terraform_code: str) -> List[str]:
        """Suggest Terraform best practices"""
        if not terraform_code:
            return ["Provide Terraform code for analysis"]

        suggestions = []

        # Check for version constraints
        if "required_version" not in terraform_code:
            suggestions.append("Add Terraform version constraints using required_version")

        # Check for provider version constraints
        if "required_providers" not in terraform_code:
            suggestions.append("Pin provider versions using required_providers block")

        # Check for tags
        if "tags" not in terraform_code and "aws_" in terraform_code:
            suggestions.append(
                "Add consistent tags to AWS resources for better management and cost tracking"
            )

        # Check for remote state
        if "backend" not in terraform_code:
            suggestions.append(
                "Configure remote state backend (S3 + DynamoDB) for team collaboration"
            )

        # Check for data sources
        if (
            "data." not in terraform_code
            and len(re.findall(r'resource\s+"aws_', terraform_code)) > 3
        ):
            suggestions.append("Consider using data sources to reference existing infrastructure")

        # Check for modules
        if len(re.findall(r'resource\s+"', terraform_code)) > 10 and "module" not in terraform_code:
            suggestions.append("Consider organizing code into reusable modules")

        # Check for variables
        if '"${var.' not in terraform_code and "var." not in terraform_code:
            suggestions.append("Use variables to make your code more flexible and reusable")

        # Check for outputs
        if "output" not in terraform_code:
            suggestions.append("Add output values for important resource attributes")

        return suggestions


class FileOperations:
    """File system operations for managing Terraform files"""

    @staticmethod
    def save_terraform_file(content: str, filename: str = "main.tf") -> Dict[str, Any]:
        """Save Terraform code to file"""
        if not content or not content.strip():
            return {"success": False, "error": "Cannot save empty content"}

        try:
            file_path = Path(filename)
            file_path.write_text(content, encoding="utf-8")
            file_size = file_path.stat().st_size

            return {
                "success": True,
                "message": f"Terraform code saved to {filename}",
                "file_path": str(file_path.absolute()),
                "file_size": file_size,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save file: {str(e)}",
            }

    @staticmethod
    def create_terraform_structure(
        project_name: str = "terraform-project",
    ) -> Dict[str, Any]:
        """Create a standard Terraform project structure"""
        try:
            base_dir = Path(project_name)
            base_dir.mkdir(exist_ok=True)

            # Create directory structure
            directories = [
                "modules",
                "environments/dev",
                "environments/staging",
                "environments/prod",
            ]

            for directory in directories:
                (base_dir / directory).mkdir(parents=True, exist_ok=True)

            # Create standard files
            files = {
                "main.tf": """# Main infrastructure resources
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
""",
                "variables.tf": """# Input variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "my-project"
}
""",
                "outputs.tf": """# Output values
output "region" {
  description = "AWS region"
  value       = var.aws_region
}
""",
                "versions.tf": """terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment and configure for remote state
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "state/terraform.tfstate"
  #   region = "us-east-1"
  # }
}
""",
                "terraform.tfvars.example": """# Example variable values
aws_region   = "us-east-1"
environment  = "dev"
project_name = "my-terraform-project"
""",
                "README.md": f"""# {project_name}

Terraform infrastructure project

## Structure

- `main.tf` - Main infrastructure resources
- `variables.tf` - Input variables
- `outputs.tf` - Output values
- `versions.tf` - Terraform and provider version constraints
- `modules/` - Reusable modules
- `environments/` - Environment-specific configurations

## Usage

1. Copy `terraform.tfvars.example` to `terraform.tfvars`
2. Edit `terraform.tfvars` with your values
3. Run `terraform init`
4. Run `terraform plan`
5. Run `terraform apply`

## Requirements

- Terraform >= 1.0
- AWS CLI configured
""",
            }

            # Create files
            for filename, content in files.items():
                (base_dir / filename).write_text(content, encoding="utf-8")

            # Create environment-specific files
            for env in ["dev", "staging", "prod"]:
                env_dir = base_dir / "environments" / env
                (env_dir / "terraform.tfvars").write_text(
                    f"""# {env.title()} environment variables
aws_region   = "us-east-1"
environment  = "{env}"
project_name = "{project_name}-{env}"
""",
                    encoding="utf-8",
                )
                (env_dir / "main.tf").write_text(
                    f"""# {env.title()} environment configuration
terraform {{
  backend "s3" {{
    bucket = "{project_name}-terraform-state-{env}"
    key    = "{env}/terraform.tfstate"
    region = "us-east-1"
  }}
}}

module "main" {{
  source = "../.."

  aws_region   = var.aws_region
  environment  = "{env}"
  project_name = var.project_name
}}
""",
                    encoding="utf-8",
                )

            return {
                "success": True,
                "message": f"Terraform project structure created in {project_name}/",
                "created": {
                    "directories": directories,
                    "files": list(files.keys()),
                    "project_path": str(base_dir.absolute()),
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create structure: {str(e)}",
            }

    @staticmethod
    def read_terraform_file(filename: str = "main.tf") -> Dict[str, Any]:
        """Read Terraform file contents"""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File {filename} does not exist",
                }

            content = file_path.read_text(encoding="utf-8")
            file_size = file_path.stat().st_size

            return {
                "success": True,
                "content": content,
                "file_path": str(file_path.absolute()),
                "file_size": file_size,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
            }


class TerraformPlanAnalyzer:
    """Analyze Terraform plan outputs"""

    @staticmethod
    def analyze_plan_output(plan_output: str) -> Dict[str, Any]:
        """Analyze terraform plan output"""
        if not plan_output:
            return {"error": "Empty plan output provided"}

        analysis = {
            "resources_to_create": 0,
            "resources_to_update": 0,
            "resources_to_destroy": 0,
            "changes": [],
            "warnings": [],
            "errors": [],
        }

        try:
            # Extract resource changes
            create_pattern = r"#\s+(.+?)\s+will be created"
            update_pattern = r"#\s+(.+?)\s+will be updated"
            destroy_pattern = r"#\s+(.+?)\s+will be destroyed"

            creates = re.findall(create_pattern, plan_output)
            updates = re.findall(update_pattern, plan_output)
            destroys = re.findall(destroy_pattern, plan_output)

            analysis["resources_to_create"] = len(creates)
            analysis["resources_to_update"] = len(updates)
            analysis["resources_to_destroy"] = len(destroys)

            # Extract plan summary
            summary_pattern = (
                r"Plan:\s+(\d+)\s+to\s+add,\s+(\d+)\s+to\s+change,\s+(\d+)\s+to\s+destroy"
            )
            summary_match = re.search(summary_pattern, plan_output)
            if summary_match:
                analysis["summary"] = {
                    "add": int(summary_match.group(1)),
                    "change": int(summary_match.group(2)),
                    "destroy": int(summary_match.group(3)),
                }

            # Look for warnings and errors
            warning_lines = [line for line in plan_output.split("\n") if "Warning:" in line]
            error_lines = [line for line in plan_output.split("\n") if "Error:" in line]

            analysis["warnings"] = warning_lines
            analysis["errors"] = error_lines

        except Exception as e:
            analysis["error"] = f"Plan analysis failed: {str(e)}"

        return analysis


class StrandsTools:
    """Enhanced main tools registry for the Terraform AI Agent"""

    def __init__(self):
        self.calculator = CalculatorTool()
        self.terraform_validator = TerraformValidatorTool()
        self.aws_pricing = AWSPricingTool()
        self.git_integration = GitIntegrationTool()
        self.infrastructure_analyzer = InfrastructureAnalyzer()
        self.file_operations = FileOperations()
        self.plan_analyzer = TerraformPlanAnalyzer()

    def get_all_tools(self) -> Dict[str, Any]:
        """Get all available tools"""
        return {
            "calculator": self.calculator,
            "terraform_validator": self.terraform_validator,
            "aws_pricing": self.aws_pricing,
            "git_integration": self.git_integration,
            "infrastructure_analyzer": self.infrastructure_analyzer,
            "file_operations": self.file_operations,
            "plan_analyzer": self.plan_analyzer,
        }

    def get_tool_info(self) -> Dict[str, str]:
        """Get information about available tools"""
        return {
            "calculator": "Basic mathematical operations for cost calculations",
            "terraform_validator": "Validate and format Terraform code with enhanced security",
            "aws_pricing": "Estimate AWS infrastructure costs with regional pricing",
            "git_integration": "Git repository management and GitHub integration with validation",
            "infrastructure_analyzer": "Comprehensive Terraform analysis with security and compliance checks",
            "file_operations": "File system operations for Terraform projects",
            "plan_analyzer": "Analyze Terraform plan outputs for changes and impacts",
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all tools"""
        health = {"overall_status": "healthy", "tools": {}, "issues": []}

        # Check each tool's availability
        tool_checks = [
            ("terraform", ["terraform", "--version"]),
            ("git", ["git", "--version"]),
            ("infracost", ["infracost", "--version"]),
        ]

        for tool_name, command in tool_checks:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                health["tools"][tool_name] = {
                    "available": result.returncode == 0,
                    "version": (result.stdout.split("\n")[0] if result.returncode == 0 else None),
                }

                if result.returncode != 0:
                    health["issues"].append(f"{tool_name} not working properly")

            except (FileNotFoundError, subprocess.TimeoutExpired):
                health["tools"][tool_name] = {
                    "available": False,
                    "version": None,
                }
                health["issues"].append(f"{tool_name} not installed or not in PATH")

        # Determine overall status
        terraform_available = health["tools"].get("terraform", {}).get("available", False)
        if not terraform_available:
            health["overall_status"] = "degraded"

        if len(health["issues"]) > 2:
            health["overall_status"] = "unhealthy"

        return health


# Export the main tools for easy importing
tools = StrandsTools()
calculator = tools.calculator
terraform_validator = tools.terraform_validator
aws_pricing = tools.aws_pricing
git_integration = tools.git_integration
infrastructure_analyzer = tools.infrastructure_analyzer
file_operations = tools.file_operations
plan_analyzer = tools.plan_analyzer


# Backward compatibility
def calculate(operation: str, a: float, b: float = None) -> float:
    """Backward compatible calculator function"""
    try:
        if operation == "add" and b is not None:
            return calculator.add(a, b)
        elif operation == "subtract" and b is not None:
            return calculator.subtract(a, b)
        elif operation == "multiply" and b is not None:
            return calculator.multiply(a, b)
        elif operation == "divide" and b is not None:
            return calculator.divide(a, b)
        else:
            raise ValueError(f"Unsupported operation: {operation} or missing parameter")
    except Exception as e:
        logger.error(f"Calculation error: {str(e)}")
        raise


# Export main functions for direct import


# ------------- ADDED: strict static sanity checks for ECS Fargate behind
def static_sanity_checks(hcl_text: str) -> List[str]:
    """
    Strict invariants for our ECS Fargate behind ALB, NAT-less pattern.
    Returns a list of human-readable errors; empty list means "looks OK".
    This is a static heuristic pass; runtime validation still happens via terraform validate.
    """
    issues: List[str] = []
    txt = hcl_text or ""

    # 1) Disallow legacy aws_alb*
    if re.search(r"\\baws_alb(?:_listener|_target_group)?\\b", txt):
        issues.append("Use aws_lb/aws_lb_listener/aws_lb_target_group; aws_alb* is not supported.")

    # 2) Require ALB + TG + Listener
    if not re.search(r'\\bresource\\s+"aws_lb"\\b', txt):
        issues.append('Missing Application Load Balancer: add resource "aws_lb".')
    if not re.search(r'\\bresource\\s+"aws_lb_target_group"\\b', txt):
        issues.append(
            'Missing target group: add resource "aws_lb_target_group" with target_type="ip" for Fargate.'
        )
    if not re.search(r'\\bresource\\s+"aws_lb_listener"\\b', txt):
        issues.append('Missing listener: add resource "aws_lb_listener" on port 80.')

    # Listener must NOT have tags
    if re.search(
        r'\\bresource\\s+"aws_lb_listener"\\s+"[^"]+"\\s*{[^}]*\\btags\\s*=',
        txt,
        re.S,
    ):
        issues.append("aws_lb_listener does not support tags.")

    # 3) Fargate invariants
    if re.search(r'\\bresource\\s+"aws_ecs_task_definition"\\b', txt):
        if 'requires_compatibilities = ["FARGATE"]' not in txt:
            issues.append("Task definition must require FARGATE.")
        if 'network_mode = "awsvpc"' not in txt:
            issues.append('Task definition must set network_mode="awsvpc".')
    if re.search(r'\\bresource\\s+"aws_ecs_service"\\b', txt):
        if 'launch_type = "FARGATE"' not in txt:
            issues.append('ECS service must set launch_type="FARGATE".')
        m = re.search(
            r'resource\\s+"aws_ecs_service"\\s+"[^"]+"\\s*{[^}]*network_configuration\\s*{(?P<body>[^}]*)}',
            txt,
            re.S,
        )
        if m:
            net = m.group("body")
            if "assign_public_ip = false" not in net:
                issues.append("ECS service must set assign_public_ip = false (NAT-less design).")
            if re.search(r"aws_subnet\\.public", net):
                issues.append(
                    "ECS service should use private subnets (not public) in network_configuration.subnets."
                )
        else:
            issues.append("ECS service missing network_configuration block.")

    # 4) Target group must be 'ip' for Fargate
    if (
        re.search(r'\\bresource\\s+"aws_lb_target_group"\\b', txt)
        and 'target_type = "ip"' not in txt
    ):
        issues.append('LB target group must set target_type="ip" for Fargate.')

    # 5) NAT-less endpoints required
    must_ifaces = ["ecr.api", "ecr.dkr", "logs", "ssm", "ec2", "sts"]
    for svc in must_ifaces:
        pat1 = rf"com\\.amazonaws\\.\\$\\{{[^}}]+ }}\\.{re.escape(svc)}"
        pat2 = rf"com\\.amazonaws\\.[a-z0-9-]+\\.{re.escape(svc)}"
        if re.search(pat1, txt) is None and re.search(pat2, txt) is None:
            issues.append(f"Missing Interface VPC endpoint for {svc}.")

    # S3 must be Gateway and attached to PRIVATE route tables (heuristic)
    if re.search(r"com\\.amazonaws\\.(?:\\$\\{[^}]+}|[a-z0-9-]+)\\.s3", txt):
        if 'vpc_endpoint_type = "Gateway"' not in txt:
            issues.append("S3 endpoint must be a Gateway endpoint.")
        if re.search(
            r"route_table_ids\\s*=\\s*\\[\\s*aws_route_table\\.public(?:\\[.*?\\])?\\.id\\s*\\]",
            txt,
        ):
            issues.append(
                "Attach the S3 Gateway endpoint to PRIVATE route tables, not the public route table."
            )

    # 6) ECS SG heuristic (allow from ALB SG)
    if (
        re.search(r'resource\\s+"aws_security_group"\\s+"ecs', txt)
        and "source_security_group_id" not in txt
    ):
        issues.append(
            "ECS SG should allow inbound from ALB SG via source_security_group_id (heuristic)."
        )

    # 7) Duplicate variables
    names = re.findall(r'\\bvariable\\s+"([^"]+)"', txt)
    if len(names) != len(set(names)):
        issues.append("Duplicate variable names detected.")

    return issues


__all__ = [
    "tools",
    "calculator",
    "terraform_validator",
    "aws_pricing",
    "git_integration",
    "infrastructure_analyzer",
    "file_operations",
    "plan_analyzer",
    "calculate",
    "StrandsTools",
    "CalculatorTool",
    "TerraformValidatorTool",
    "AWSPricingTool",
    "GitIntegrationTool",
    "InfrastructureAnalyzer",
    "FileOperations",
    "TerraformPlanAnalyzer",
    "static_sanity_checks",
    "extract_best_terraform_block",
    "ensure_minimal_scaffold",
]


# ---------------- Best-block extractor (added) ----------------
def extract_best_terraform_block(response_text: str) -> str:
    # Heuristically choose the best Terraform/HCL block from an LLM reply.
    # Prefers the largest fenced block labelled ```terraform|hcl|tf```,
    # falls back to the largest fenced block of any language, then to
    # HCL-looking text.
    import re
    from typing import List

    if not response_text:
        return ""
    # 1) prefer fenced blocks labelled terraform/hcl/tf
    labelled = re.findall(
        r"```(?:terraform|hcl|tf)\s*\n(.*?)```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    candidates: List[str] = labelled[:]
    # 2) any fenced block
    any_fenced = re.findall(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    candidates.extend(any_fenced)
    # 3) HCL-ish spans without fences (very conservative)
    hcl_spans = re.findall(
        r"((?:terraform|provider|variable|output|data|module|resource)\s+[^{]*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(hcl_spans)
    if not candidates:
        return ""

    # score by size + terraform-y tokens
    def score(txt: str) -> int:
        tokens = sum(
            1
            for t in [
                "resource ",
                "provider ",
                "variable ",
                "output ",
                "module ",
                "terraform ",
            ]
            if t in txt
        )
        return len(txt) + 200 * tokens

    best = max(candidates, key=score)
    return best.strip()


# ---------------- Minimal scaffold injector (added) ----------------
def ensure_minimal_scaffold(terraform_code: str, default_region: str = "us-east-1") -> str:
    # If terraform/provider scaffold is missing, add a minimal, pinned setup.
    # Does NOT overwrite existing blocks.
    code = terraform_code or ""
    needs_tf = "terraform {" not in code
    needs_provider = 'provider "aws"' not in code
    if not (needs_tf or needs_provider):
        return code
    scaffold_parts = []
    if needs_tf:
        scaffold_parts.append(
            "terraform {\n"
            '  required_version = ">= 1.3.0"\n'
            "  required_providers {\n"
            "    aws = {\n"
            '      source  = "hashicorp/aws"\n'
            '      version = "~> 5.0"\n'
            "    }\n"
            "  }\n"
            "}\n"
        )
    if needs_provider:
        scaffold_parts.append('provider "aws" {\n' f'  region = "{default_region}"\n' "}\n")
    return ("\n".join(scaffold_parts) + "\n" + code).strip()


# === APPEND: pluggable tools + HCL parsing (non-destructive) ===


_ST = _ST_Settings()


class Tool(_abc.ABC):
    name: str

    @_abc.abstractmethod
    async def run(self, **kwargs): ...


class TerraformValidateTool(Tool):
    name = "terraform_validate"

    async def run(self, tf_code: str):
        try:
            _st_hcl2.load(_ST_StringIO(tf_code))
        except Exception as e:
            return {
                "success": False,
                "data": {},
                "error": f"HCL parse error: {e}",
            }
        with _st_secure_tempdir("tf_val_v2_") as d:
            import os as _os

            with open(_os.path.join(d, "main.tf"), "w", encoding="utf-8") as f:
                f.write(tf_code)
            rc, out, err = await _st_run_cmd_async(_ST.TF_BIN, "init", "-input=false", cwd=d)
            if rc != 0:
                return {
                    "success": False,
                    "data": {"stderr": err},
                    "error": "terraform init failed",
                }
            rc, out, err = await _st_run_cmd_async(_ST.TF_BIN, "validate", cwd=d)
            ok = rc == 0
            return {
                "success": ok,
                "data": {"stdout": out, "stderr": err},
                "error": "" if ok else err,
            }


class AzureTool(Tool):
    name = "azure_inspect"

    async def run(self, subscription_id: str = ""):
        rc, out, err = await _st_run_cmd_async("az", "account", "show")
        return {
            "success": rc == 0,
            "data": {"subscription": out},
            "error": "" if rc == 0 else err,
        }


class GcpTool(Tool):
    name = "gcp_inspect"

    async def run(self, project_id: str = ""):
        rc, out, err = await _st_run_cmd_async("gcloud", "config", "list")
        return {
            "success": rc == 0,
            "data": {"config": out},
            "error": "" if rc == 0 else err,
        }


TOOLS_REGISTRY_V2 = {
    "terraform_validate": TerraformValidateTool(),
    "azure_inspect": AzureTool(),
    "gcp_inspect": GcpTool(),
}
