import json
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cachetools import TTLCache as _TTLCache

from src.app.core.config import Settings as _IC_Settings
from src.app.utils.utils import run_cmd_async as _ic_run_cmd_async
from src.app.utils.utils import secure_tempdir as _ic_secure_tempdir

logger = logging.getLogger(__name__)


class InfracostIntegration:
    """Real Infracost CLI integration with budget management"""

    def __init__(self):
        self.infracost_available = self._check_infracost_availability()
        self.budgets = self._load_budgets()

    def _check_infracost_availability(self) -> bool:
        """Check if Infracost CLI is available and configured"""
        try:
            result = subprocess.run(
                ["infracost", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Check if API key is configured
                auth_result = subprocess.run(
                    ["infracost", "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return auth_result.returncode == 0
            return False

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _load_budgets(self) -> Dict[str, Any]:
        """Load budget configurations from file"""
        budget_file = Path("budgets.json")

        default_budgets = {
            "workspaces": {
                "default": {
                    "monthly_limit": 100.0,
                    "alert_thresholds": [50.0, 80.0, 100.0],
                    "currency": "USD",
                },
                "development": {
                    "monthly_limit": 200.0,
                    "alert_thresholds": [100.0, 150.0, 200.0],
                    "currency": "USD",
                },
                "staging": {
                    "monthly_limit": 500.0,
                    "alert_thresholds": [250.0, 400.0, 500.0],
                    "currency": "USD",
                },
                "production": {
                    "monthly_limit": 2000.0,
                    "alert_thresholds": [1000.0, 1500.0, 2000.0],
                    "currency": "USD",
                },
            },
            "global_settings": {
                "default_currency": "USD",
                "enable_alerts": True,
                "cost_increase_threshold_percent": 20.0,
            },
        }

        if budget_file.exists():
            try:
                with open(budget_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load budgets.json: {e}, using defaults")
                return default_budgets
        else:
            # Create default budget file
            try:
                with open(budget_file, "w", encoding="utf-8") as f:
                    json.dump(default_budgets, f, indent=2)
                logger.info("Created default budgets.json file")
            except IOError as e:
                logger.warning(f"Failed to create budgets.json: {e}")

            return default_budgets

    def generate_cost_estimate(
        self, terraform_code: str, workspace: str = "default"
    ) -> Dict[str, Any]:
        """Generate comprehensive cost estimate using Infracost CLI"""
        if not self.infracost_available:
            return {
                "success": False,
                "error": "Infracost CLI not available or not configured",
                "fallback_available": True,
            }

        if not terraform_code or not terraform_code.strip():
            return {"success": False, "error": "Empty Terraform code provided"}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write Terraform code
                main_tf = temp_path / "main.tf"
                main_tf.write_text(terraform_code, encoding="utf-8", errors="ignore")

                # Create a basic terraform block if not present
                if "terraform {" not in terraform_code:
                    versions_tf = temp_path / "versions.tf"
                    versions_tf.write_text(
                        """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"  # Default region for cost estimation
}
""",
                        encoding="utf-8",
                    )

                # Initialize Terraform
                init_result = subprocess.run(
                    ["terraform", "init", "-input=false"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if init_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Terraform init failed: {init_result.stderr}",
                        "fallback_available": True,
                    }

                # Generate plan
                plan_result = subprocess.run(
                    ["terraform", "plan", "-out=tfplan", "-input=false"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if plan_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Terraform init failed: {init_result.stderr}",
                        "fallback_available": True,
                    }

                # Generate Infracost breakdown
                breakdown_result = subprocess.run(
                    [
                        "infracost",
                        "breakdown",
                        "--path",
                        temp_dir,
                        "--format",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )

                if breakdown_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Infracost breakdown failed: {
                            breakdown_result.stderr}",
                        "fallback_available": True,
                    }

                # Parse Infracost output
                try:
                    infracost_data = json.loads(breakdown_result.stdout)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse Infracost output: {e}",
                        "fallback_available": True,
                    }

                # Process and format results
                return self._process_infracost_output(infracost_data, workspace)

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "error": f"Infracost operation timed out: {e}",
                "fallback_available": True,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during cost estimation: {e}",
                "fallback_available": True,
            }

    def _process_infracost_output(
        self, infracost_data: Dict[str, Any], workspace: str
    ) -> Dict[str, Any]:
        """Process Infracost output and apply budget checks"""
        projects = infracost_data.get("projects", [])

        if not projects:
            return {
                "success": False,
                "error": "No cost data found in Infracost output",
            }

        # Extract cost information
        total_monthly_cost = 0.0
        total_hourly_cost = 0.0
        resource_breakdown = []

        for project in projects:
            breakdown = project.get("breakdown", {})
            resources = breakdown.get("resources", [])

            project_monthly = breakdown.get("totalMonthlyCost")
            project_hourly = breakdown.get("totalHourlyCost")

            if project_monthly:
                total_monthly_cost += float(project_monthly)
            if project_hourly:
                total_hourly_cost += float(project_hourly)

            # Process individual resources
            for resource in resources:
                resource_cost = resource.get("monthlyCost")
                if resource_cost:
                    resource_breakdown.append(
                        {
                            "name": resource.get("name", "unknown"),
                            "resource_type": resource.get("resourceType", "unknown"),
                            "monthly_cost": float(resource_cost),
                            "hourly_cost": float(resource.get("hourlyCost", 0)),
                            "cost_components": self._extract_cost_components(resource),
                        }
                    )

        # Sort resources by cost
        resource_breakdown.sort(key=lambda x: x["monthly_cost"], reverse=True)

        # Apply budget checks
        budget_analysis = self._check_budget_compliance(total_monthly_cost, workspace)

        # Calculate additional metrics
        yearly_cost = total_monthly_cost * 12
        daily_cost = total_monthly_cost / 30

        return {
            "success": True,
            "cost_estimate": {
                "monthly_cost": round(total_monthly_cost, 2),
                "hourly_cost": round(total_hourly_cost, 4),
                "daily_cost": round(daily_cost, 2),
                "yearly_cost": round(yearly_cost, 2),
                "currency": "USD",
            },
            "resource_breakdown": resource_breakdown[:10],  # Top 10 most expensive resources
            "budget_analysis": budget_analysis,
            "total_resources": len(resource_breakdown),
            "workspace": workspace,
            "generated_at": datetime.now().isoformat(),
            "infracost_version": self._get_infracost_version(),
        }

    def _extract_cost_components(self, resource: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract cost components for a resource"""
        components = []
        cost_components = resource.get("costComponents", [])

        for component in cost_components:
            components.append(
                {
                    "name": component.get("name", "unknown"),
                    "unit": component.get("unit", ""),
                    "hourly_quantity": component.get("hourlyQuantity"),
                    "monthly_quantity": component.get("monthlyQuantity"),
                    "hourly_cost": float(component.get("hourlyCost", 0)),
                    "monthly_cost": float(component.get("monthlyCost", 0)),
                }
            )

        return components

    def _check_budget_compliance(self, estimated_cost: float, workspace: str) -> Dict[str, Any]:
        """Check budget compliance and generate alerts"""
        workspace_budget = self.budgets.get("workspaces", {}).get(workspace)

        if not workspace_budget:
            workspace_budget = self.budgets.get("workspaces", {}).get(
                "default",
                {
                    "monthly_limit": 100.0,
                    "alert_thresholds": [50.0, 80.0, 100.0],
                    "currency": "USD",
                },
            )

        monthly_limit = workspace_budget.get("monthly_limit", 100.0)
        alert_thresholds = workspace_budget.get("alert_thresholds", [50.0, 80.0, 100.0])

        budget_utilization = (estimated_cost / monthly_limit) * 100 if monthly_limit > 0 else 0

        # Determine alert level
        alert_level = "GREEN"
        triggered_threshold = None

        for threshold in sorted(alert_thresholds):
            if estimated_cost >= threshold:
                triggered_threshold = threshold
                if threshold == alert_thresholds[-1]:  # Highest threshold
                    alert_level = "RED"
                elif threshold == alert_thresholds[-2]:  # Second highest
                    alert_level = "YELLOW"
                else:
                    alert_level = "ORANGE"

        over_budget = estimated_cost > monthly_limit
        remaining_budget = max(0, monthly_limit - estimated_cost)

        return {
            "monthly_limit": monthly_limit,
            "estimated_cost": estimated_cost,
            "remaining_budget": round(remaining_budget, 2),
            "budget_utilization_percent": round(budget_utilization, 1),
            "over_budget": over_budget,
            "alert_level": alert_level,
            "triggered_threshold": triggered_threshold,
            "recommendations": self._generate_budget_recommendations(
                estimated_cost, monthly_limit, over_budget
            ),
        }

    def _generate_budget_recommendations(
        self, cost: float, limit: float, over_budget: bool
    ) -> List[str]:
        """Generate budget-related recommendations"""
        recommendations = []

        if over_budget:
            excess = cost - limit
            recommendations.append(
                f"[COST] BUDGET EXCEEDED: Estimated cost is ${
                    excess:.2f} over budget"
            )
            recommendations.append("[CHECK] Review resource sizing and consider optimization")
            recommendations.append(" Consider implementing auto-scaling and scheduled shutdowns")

        utilization = (cost / limit) * 100 if limit > 0 else 0

        if utilization > 80:
            recommendations.append("[WARN]  HIGH BUDGET UTILIZATION: Consider cost optimization")
            recommendations.append(
                "[METRICS] Review most expensive resources for right-sizing opportunities"
            )

        if cost > 500:
            recommendations.append(
                " Consider Reserved Instances or Savings Plans for long-term workloads"
            )

        return recommendations

    def _get_infracost_version(self) -> Optional[str]:
        """Get Infracost version"""
        try:
            result = subprocess.run(
                ["infracost", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def generate_cost_diff(
        self,
        old_terraform: str,
        new_terraform: str,
        workspace: str = "default",
    ) -> Dict[str, Any]:
        """Generate cost difference between two Terraform configurations"""
        if not self.infracost_available:
            return {
                "success": False,
                "error": "Infracost CLI not available for diff generation",
                "fallback_available": False,
            }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create directories for old and new configurations
                old_dir = temp_path / "old"
                new_dir = temp_path / "new"
                old_dir.mkdir()
                new_dir.mkdir()

                # Write configurations
                (old_dir / "main.tf").write_text(old_terraform, encoding="utf-8", errors="ignore")
                (new_dir / "main.tf").write_text(new_terraform, encoding="utf-8", errors="ignore")

                # Generate diff
                diff_result = subprocess.run(
                    [
                        "infracost",
                        "diff",
                        "--path",
                        str(old_dir),
                        "--compare-to",
                        str(new_dir),
                        "--format",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if diff_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Infracost diff failed: {
                            diff_result.stderr}",
                    }

                diff_data = json.loads(diff_result.stdout)
                return self._process_cost_diff(diff_data, workspace)

        except Exception as e:
            return {
                "success": False,
                "error": f"Cost diff generation failed: {e}",
            }

    def _process_cost_diff(self, diff_data: Dict[str, Any], workspace: str) -> Dict[str, Any]:
        """Process cost diff data"""
        projects = diff_data.get("projects", [])

        if not projects:
            return {"success": False, "error": "No cost diff data available"}

        total_monthly_diff = 0.0
        resource_diffs = []

        for project in projects:
            diff = project.get("diff", {})
            total_monthly_diff += float(diff.get("totalMonthlyCost", 0))

            # Process resource differences
            resources = diff.get("resources", [])
            for resource in resources:
                monthly_diff = float(resource.get("monthlyCost", 0))
                if abs(monthly_diff) > 0.01:  # Only include significant changes
                    resource_diffs.append(
                        {
                            "name": resource.get("name", "unknown"),
                            "resource_type": resource.get("resourceType", "unknown"),
                            "monthly_cost_diff": monthly_diff,
                            "change_type": (
                                "added"
                                if monthly_diff > 0
                                else ("removed" if monthly_diff < 0 else "modified")
                            ),
                        }
                    )

        # Sort by absolute cost impact
        resource_diffs.sort(key=lambda x: abs(x["monthly_cost_diff"]), reverse=True)

        return {
            "success": True,
            "cost_diff": {
                "monthly_diff": round(total_monthly_diff, 2),
                "yearly_diff": round(total_monthly_diff * 12, 2),
                "percentage_change": self._calculate_percentage_change(diff_data),
                "currency": "USD",
            },
            "resource_changes": resource_diffs[:10],
            "summary": self._generate_diff_summary(total_monthly_diff, resource_diffs),
            "workspace": workspace,
            "generated_at": datetime.now().isoformat(),
        }

    def _calculate_percentage_change(self, diff_data: Dict[str, Any]) -> Optional[float]:
        """Calculate percentage change in cost"""
        try:
            projects = diff_data.get("projects", [])
            if projects:
                diff = projects[0].get("diff", {})
                total_current = float(diff.get("totalMonthlyCost", 0))
                total_previous = total_current - float(diff.get("totalMonthlyCost", 0))

                if total_previous > 0:
                    return round(
                        ((total_current - total_previous) / total_previous) * 100,
                        1,
                    )
        except (KeyError, ValueError, ZeroDivisionError):
            pass
        return None

    def _generate_diff_summary(
        self, total_diff: float, resource_diffs: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable diff summary"""
        if abs(total_diff) < 0.01:
            return "No significant cost changes detected"

        change_type = "increase" if total_diff > 0 else "decrease"
        abs_diff = abs(total_diff)

        summary = f"Monthly cost {change_type} of ${abs_diff:.2f}"

        if resource_diffs:
            added_count = len([r for r in resource_diffs if r["change_type"] == "added"])
            removed_count = len([r for r in resource_diffs if r["change_type"] == "removed"])

            if added_count > 0:
                summary += f" ({added_count} resources added"
            if removed_count > 0:
                summary += (
                    f", {removed_count} resources removed"
                    if added_count > 0
                    else f" ({removed_count} resources removed"
                )
            if added_count > 0 or removed_count > 0:
                summary += ")"

        return summary

    def update_budget(
        self,
        workspace: str,
        monthly_limit: float,
        alert_thresholds: List[float],
    ) -> Dict[str, Any]:
        """Update budget configuration for a workspace"""
        try:
            if workspace not in self.budgets.get("workspaces", {}):
                self.budgets["workspaces"][workspace] = {}

            self.budgets["workspaces"][workspace].update(
                {
                    "monthly_limit": monthly_limit,
                    "alert_thresholds": sorted(alert_thresholds),
                    "currency": "USD",
                    "updated_at": datetime.now().isoformat(),
                }
            )

            # Save updated budgets
            with open("budgets.json", "w", encoding="utf-8") as f:
                json.dump(self.budgets, f, indent=2)

            return {
                "success": True,
                "message": f"Budget updated for workspace '{workspace}'",
                "new_config": self.budgets["workspaces"][workspace],
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to update budget: {e}"}

    def get_budget_status(self, workspace: str = "default") -> Dict[str, Any]:
        """Get current budget status for a workspace"""
        workspace_budget = self.budgets.get("workspaces", {}).get(workspace)

        if not workspace_budget:
            return {
                "workspace": workspace,
                "budget_configured": False,
                "message": "No budget configured for this workspace",
            }

        return {
            "workspace": workspace,
            "budget_configured": True,
            "monthly_limit": workspace_budget.get("monthly_limit"),
            "alert_thresholds": workspace_budget.get("alert_thresholds"),
            "currency": workspace_budget.get("currency", "USD"),
            "last_updated": workspace_budget.get("updated_at"),
        }

    # === APPEND: async + cache + multi-currency (non-destructive) ===


_ic_settings = _IC_Settings()
_ic_cache = _TTLCache(maxsize=256, ttl=3600)
_IC_RATES = {
    ("USD", "USD"): 1.0,
    ("USD", "INR"): 83.0,
    ("USD", "EUR"): 0.9,
    ("EUR", "USD"): 1.11,
    ("INR", "USD"): 1 / 83.0,
}


def _ic_convert(amount_usd: float, target: str) -> float:
    return round(amount_usd * _IC_RATES.get(("USD", target.upper()), 1.0), 2)


async def estimate_cost_async_v2(tf_code: str, currency: str = "USD"):
    key = f"v2:{hash(tf_code)}:{currency.upper()}"
    cached = _ic_cache.get(key)
    if cached:
        return {
            "success": True,
            "data": {**cached, "cached": True},
            "error": "",
        }

    try:
        with _ic_secure_tempdir("ic_v2_") as d:
            import json as _json
            import os as _os

            with open(_os.path.join(d, "main.tf"), "w", encoding="utf-8") as f:
                f.write(tf_code)
            rc, out, err = await _ic_run_cmd_async(
                _ic_settings.TF_BIN, "init", "-input=false", cwd=d
            )
            if rc != 0:
                return {
                    "success": False,
                    "data": {"stderr": err},
                    "error": "terraform init failed",
                }
            rc, out, err = await _ic_run_cmd_async(
                _ic_settings.TF_BIN, "plan", "-out=plan.out", cwd=d
            )
            if rc != 0:
                return {
                    "success": False,
                    "data": {"stderr": err},
                    "error": "terraform plan failed",
                }
            rc, out, err = await _ic_run_cmd_async(
                _ic_settings.INFRACOST_BIN,
                "breakdown",
                f"--path={d}",
                "--format=json",
                cwd=d,
            )
            if rc != 0:
                return {
                    "success": False,
                    "data": {"stderr": err},
                    "error": "infracost breakdown failed",
                }
            report = _json.loads(out or "{}")
            total_usd = float(report.get("totalMonthlyCost", 0.0) or 0.0)
            total_conv = _ic_convert(total_usd, currency)
            data = {
                "currency": currency.upper(),
                "total_monthly_cost_converted": total_conv,
                "total_monthly_cost_usd": total_usd,
                "breakdown": report,
            }
            _ic_cache[key] = data
            return {"success": True, "data": data, "error": ""}
    except Exception as e:
        return {"success": False, "data": {}, "error": str(e)}


# Global Infracost integration instance
infracost_integration = InfracostIntegration()
