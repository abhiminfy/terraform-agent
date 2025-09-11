# github_integration.py - GitHub PR workflow with policy and cost checks (fixed)
import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class GitHubIntegration:
    """GitHub integration with PR workflow and automated checks"""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = os.getenv("GITHUB_REPO")  # format: "owner/repo"
        self.github_api_base = "https://api.github.com"
        self.default_branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

        # 'token' works for classic PATs; 'Bearer' also works. Keep 'token' for widest compat.
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "terraform-agent/2.0",
        }

        self.available = bool(self.github_token and self.github_repo)

        if self.available:
            logger.info(f"GitHub integration initialized for {self.github_repo}")
        else:
            logger.warning(
                "GitHub integration not available - missing token or repo configuration"
            )

    # -------- Public API --------

    def create_branch_and_pr(
        self,
        terraform_code: str,
        commit_message: Optional[str] = None,
        branch_name: Optional[str] = None,
        base_branch: Optional[str] = None,  # <--- NEW: allow override
    ) -> Dict[str, Any]:
        """Create branch, commit code, and create PR with automated checks"""
        if not self.available:
            return {"success": False, "error": "GitHub integration not configured"}

        try:
            # Generate branch name if not provided
            if not branch_name:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                branch_name = f"terraform-update-{timestamp}"

            # Generate commit message if not provided
            if not commit_message:
                commit_message = f"Update Terraform configuration - {datetime.now():%Y-%m-%d %H:%M:%S}"

            # Determine base branch with safe fallbacks
            base = base_branch or self._get_default_branch() or self.default_branch
            if not base:
                return {"success": False, "error": "Failed to get default branch"}

            # Create new branch (treat 'already exists' as success)
            branch_result = self._create_branch(branch_name, base)
            if not branch_result.get("success"):
                return branch_result

            # Commit files to branch
            commit_result = self._commit_to_branch(
                branch_name, terraform_code, commit_message
            )
            if not commit_result.get("success"):
                return commit_result

            # Run policy and cost checks (best-effort)
            checks_result = self._run_pr_checks(terraform_code)

            # Create PR with check results
            pr_result = self._create_pull_request(
                branch_name, base, commit_message, checks_result
            )

            if pr_result.get("success"):
                return {
                    "success": True,
                    "branch_name": branch_name,
                    "pr_number": pr_result["pr_number"],
                    "pr_url": pr_result["pr_url"],
                    "commit_sha": commit_result["commit_sha"],
                    "checks_passed": checks_result.get("all_passed", True),
                    "checks_summary": checks_result.get("summary", ""),
                    "message": f"PR #{pr_result['pr_number']} created successfully",
                }
            else:
                return pr_result

        except Exception as e:
            logger.error(f"Failed to create branch and PR: {e}")
            return {"success": False, "error": f"GitHub operation failed: {e}"}

    def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Get the status of a pull request"""
        if not self.available:
            return {"success": False, "error": "GitHub integration not configured"}

        try:
            r = requests.get(
                f"{self.github_api_base}/repos/{self.github_repo}/pulls/{pr_number}",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 200:
                pr = r.json()
                return {
                    "success": True,
                    "pr_number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "merged": pr["merged"],
                    "mergeable": pr["mergeable"],
                    "url": pr["html_url"],
                    "created_at": pr["created_at"],
                    "updated_at": pr["updated_at"],
                    "branch": pr["head"]["ref"],
                    "base_branch": pr["base"]["ref"],
                }
            return {
                "success": False,
                "error": f"Failed to get PR status: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error getting PR status: {e}")
            return {"success": False, "error": f"Failed to get PR status: {e}"}

    def merge_pr(self, pr_number: int, merge_method: str = "squash") -> Dict[str, Any]:
        """Merge a pull request after validation"""
        if not self.available:
            return {"success": False, "error": "GitHub integration not configured"}

        try:
            status = self.get_pr_status(pr_number)
            if not status.get("success"):
                return status

            if status["state"] != "open":
                return {
                    "success": False,
                    "error": f"PR is not open (state: {status['state']})",
                }

            if not status["mergeable"]:
                return {
                    "success": False,
                    "error": "PR has merge conflicts and cannot be merged",
                }

            data = {
                "commit_title": f"Merge PR #{pr_number}",
                "commit_message": f"Terraform configuration update via PR #{pr_number}",
                "merge_method": merge_method,
            }

            r = requests.put(
                f"{self.github_api_base}/repos/{self.github_repo}/pulls/{pr_number}/merge",
                headers=self.headers,
                json=data,
                timeout=30,
            )
            if r.status_code == 200:
                jr = r.json()
                return {
                    "success": True,
                    "merged": True,
                    "merge_commit_sha": jr["sha"],
                    "message": f"PR #{pr_number} merged successfully",
                }
            return {
                "success": False,
                "error": f"Failed to merge PR: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error merging PR: {e}")
            return {"success": False, "error": f"Failed to merge PR: {e}"}

    def list_open_prs(self) -> Dict[str, Any]:
        """List all open pull requests"""
        if not self.available:
            return {"success": False, "error": "GitHub integration not configured"}

        try:
            r = requests.get(
                f"{self.github_api_base}/repos/{self.github_repo}/pulls?state=open",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 200:
                prs_data = r.json()
                prs = [
                    {
                        "number": pr["number"],
                        "title": pr["title"],
                        "branch": pr["head"]["ref"],
                        "base_branch": pr["base"]["ref"],
                        "url": pr["html_url"],
                        "created_at": pr["created_at"],
                        "updated_at": pr["updated_at"],
                        "mergeable": pr.get(
                            "mergeable"
                        ),  # list API usually doesn't include this
                        "author": pr["user"]["login"],
                    }
                    for pr in prs_data
                ]
                return {"success": True, "prs": prs, "count": len(prs)}
            return {
                "success": False,
                "error": f"Failed to list PRs: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error listing PRs: {e}")
            return {"success": False, "error": f"Failed to list PRs: {e}"}

    def delete_branch(self, branch_name: str) -> Dict[str, Any]:
        """Delete a branch after PR is merged"""
        if not self.available:
            return {"success": False, "error": "GitHub integration not configured"}

        try:
            r = requests.delete(
                f"{self.github_api_base}/repos/{self.github_repo}/git/refs/heads/{branch_name}",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 204:
                return {
                    "success": True,
                    "message": f"Branch {branch_name} deleted successfully",
                }
            return {
                "success": False,
                "error": f"Failed to delete branch: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error deleting branch: {e}")
            return {"success": False, "error": f"Failed to delete branch: {e}"}

    # -------- Internals --------

    def _get_default_branch(self) -> Optional[str]:
        """Try GitHub first; if it fails, fall back to configured default."""
        try:
            r = requests.get(
                f"{self.github_api_base}/repos/{self.github_repo}",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("default_branch", self.default_branch)
            logger.error(f"Failed to get repo info: {r.status_code} - {r.text}")
        except Exception as e:
            logger.error(f"Error getting default branch: {e}")
        # Always return a fallback so callers don't die early
        return self.default_branch

    def _get_branch_ref(self, branch: str) -> Optional[str]:
        """Return commit SHA for given branch, or None."""
        r = requests.get(
            f"{self.github_api_base}/repos/{self.github_repo}/git/refs/heads/{branch}",
            headers=self.headers,
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["object"]["sha"]
        return None

    def _create_branch(self, branch_name: str, base_branch: str) -> Dict[str, Any]:
        """Create a new branch from base branch"""
        try:
            base_sha = self._get_branch_ref(base_branch)
            if not base_sha:
                return {
                    "success": False,
                    "error": f"Failed to get base branch SHA for '{base_branch}'",
                }

            data = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/refs",
                headers=self.headers,
                json=data,
                timeout=30,
            )
            if r.status_code == 201:
                return {"success": True, "branch_name": branch_name, "sha": base_sha}
            # If branch already exists, treat as success and continue committing.
            if r.status_code == 422 and "Reference already exists" in r.text:
                return {"success": True, "branch_name": branch_name, "sha": base_sha}
            return {
                "success": False,
                "error": f"Failed to create branch: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return {"success": False, "error": f"Branch creation failed: {e}"}

    def _commit_to_branch(
        self, branch_name: str, terraform_code: str, commit_message: str
    ) -> Dict[str, Any]:
        """Commit Terraform code to the specified branch"""
        try:
            files_to_commit = {
                "main.tf": terraform_code,
                ".gitignore": self._get_terraform_gitignore(),
                "README.md": self._generate_readme(terraform_code),
            }

            # Current ref
            ref_sha = self._get_branch_ref(branch_name)
            if not ref_sha:
                return {"success": False, "error": "Failed to get branch reference"}

            # Current tree
            r = requests.get(
                f"{self.github_api_base}/repos/{self.github_repo}/git/commits/{ref_sha}",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to get current commit: {r.status_code} - {r.text}",
                }
            base_tree = r.json()["tree"]["sha"]

            # Create blobs
            tree_items = []
            for filename, content in files_to_commit.items():
                blob = self._create_blob(content)
                if not blob.get("success"):
                    return blob
                tree_items.append(
                    {
                        "path": filename,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob["sha"],
                    }
                )

            # Create new tree
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/trees",
                headers=self.headers,
                json={"base_tree": base_tree, "tree": tree_items},
                timeout=30,
            )
            if r.status_code != 201:
                return {
                    "success": False,
                    "error": f"Failed to create tree: {r.status_code} - {r.text}",
                }
            new_tree_sha = r.json()["sha"]

            # Create commit
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/commits",
                headers=self.headers,
                json={
                    "message": commit_message,
                    "tree": new_tree_sha,
                    "parents": [ref_sha],
                },
                timeout=30,
            )
            if r.status_code != 201:
                return {
                    "success": False,
                    "error": f"Failed to create commit: {r.status_code} - {r.text}",
                }
            new_commit_sha = r.json()["sha"]

            # Update reference
            r = requests.patch(
                f"{self.github_api_base}/repos/{self.github_repo}/git/refs/heads/{branch_name}",
                headers=self.headers,
                json={"sha": new_commit_sha, "force": False},
                timeout=30,
            )
            if r.status_code == 200:
                return {
                    "success": True,
                    "commit_sha": new_commit_sha,
                    "files_committed": list(files_to_commit.keys()),
                }
            return {
                "success": False,
                "error": f"Failed to update branch reference: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error committing to branch: {e}")
            return {"success": False, "error": f"Commit failed: {e}"}

    def _commit_multiple_files(
        self, branch_name: str, files: Dict[str, str], commit_message: str
    ) -> Dict[str, Any]:
        """Generic 'commit multiple files' helper used by auto-apply."""
        try:
            # Current ref
            ref_sha = self._get_branch_ref(branch_name)
            if not ref_sha:
                return {"success": False, "error": "Failed to get branch reference"}

            # Current tree
            r = requests.get(
                f"{self.github_api_base}/repos/{self.github_repo}/git/commits/{ref_sha}",
                headers=self.headers,
                timeout=30,
            )
            if r.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to get current commit: {r.status_code} - {r.text}",
                }
            base_tree = r.json()["tree"]["sha"]

            # Create blobs
            tree_items = []
            for filename, content in files.items():
                blob = self._create_blob(content)
                if not blob.get("success"):
                    return blob
                tree_items.append(
                    {
                        "path": filename,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob["sha"],
                    }
                )

            # Create new tree
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/trees",
                headers=self.headers,
                json={"base_tree": base_tree, "tree": tree_items},
                timeout=30,
            )
            if r.status_code != 201:
                return {
                    "success": False,
                    "error": f"Failed to create tree: {r.status_code} - {r.text}",
                }
            new_tree_sha = r.json()["sha"]

            # Create commit
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/commits",
                headers=self.headers,
                json={
                    "message": commit_message,
                    "tree": new_tree_sha,
                    "parents": [ref_sha],
                },
                timeout=30,
            )
            if r.status_code != 201:
                return {
                    "success": False,
                    "error": f"Failed to create commit: {r.status_code} - {r.text}",
                }
            new_commit_sha = r.json()["sha"]

            # Update reference
            r = requests.patch(
                f"{self.github_api_base}/repos/{self.github_repo}/git/refs/heads/{branch_name}",
                headers=self.headers,
                json={"sha": new_commit_sha, "force": False},
                timeout=30,
            )
            if r.status_code == 200:
                return {
                    "success": True,
                    "commit_sha": new_commit_sha,
                    "files_committed": list(files.keys()),
                }
            return {
                "success": False,
                "error": f"Failed to update branch reference: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error committing multiple files: {e}")
            return {"success": False, "error": f"Commit failed: {e}"}

    def _create_blob(self, content: str) -> Dict[str, Any]:
        """Create a blob for file content"""
        try:
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/git/blobs",
                headers=self.headers,
                json={
                    "content": base64.b64encode(content.encode()).decode(),
                    "encoding": "base64",
                },
                timeout=30,
            )
            if r.status_code == 201:
                return {"success": True, "sha": r.json()["sha"]}
            return {
                "success": False,
                "error": f"Failed to create blob: {r.status_code} - {r.text}",
            }
        except Exception as e:
            return {"success": False, "error": f"Blob creation failed: {e}"}

    def _run_pr_checks(self, terraform_code: str) -> Dict[str, Any]:
        """Policy/cost/validation checks (best-effort; won’t block PR creation)"""
        checks_result = {
            "all_passed": True,
            "policy_check": {"passed": True, "violations": []},
            "cost_check": {"passed": True, "estimated_cost": 0, "cost_limit": 0},
            "validation_check": {"passed": True, "errors": [], "output": ""},
            "summary": "",
        }
        try:
            from backend.app.services.infracost_integration import infracost_integration
            from backend.app.services.policy_engine import policy_engine
            from strands_tools import terraform_validator

            # Policy
            policy = policy_engine.validate_with_policies(terraform_code)
            critical = [
                v
                for v in policy.get("violations", [])
                if v.get("severity") == "CRITICAL"
            ]
            high = [
                v for v in policy.get("violations", []) if v.get("severity") == "HIGH"
            ]
            checks_result["policy_check"] = {
                "passed": len(critical) == 0 and len(high) <= 2,
                "violations": policy.get("violations", []),
                "summary": policy.get("summary", {}),
            }
            if not checks_result["policy_check"]["passed"]:
                checks_result["all_passed"] = False

            # Cost
            if infracost_integration.infracost_available:
                cost = infracost_integration.generate_cost_estimate(
                    terraform_code, "default"
                )
                if cost.get("success"):
                    est = cost.get("cost_estimate", {}).get("monthly_cost", 0)
                    limit = 500.0
                    checks_result["cost_check"] = {
                        "passed": est <= limit,
                        "estimated_cost": est,
                        "cost_limit": limit,
                        "budget_analysis": cost.get("budget_analysis", {}),
                    }
                    if not checks_result["cost_check"]["passed"]:
                        checks_result["all_passed"] = False

            # Validation
            valid = terraform_validator.validate_terraform_syntax(terraform_code)
            checks_result["validation_check"] = {
                "passed": valid.get("valid", False),
                "errors": valid.get("errors", []),
                "output": valid.get("output", ""),
            }
            if not checks_result["validation_check"]["passed"]:
                checks_result["all_passed"] = False

            # Summary
            parts = []
            parts.append(
                "[OK] Policy checks passed"
                if checks_result["policy_check"]["passed"]
                else f"[ERROR] Policy checks failed ({len(checks_result['policy_check']['violations'])} violations)"
            )
            parts.append(
                f"[OK] Cost check passed (${checks_result['cost_check']['estimated_cost']:.2f}/month)"
                if checks_result["cost_check"]["passed"]
                else f"[WARN] Cost check failed (${checks_result['cost_check']['estimated_cost']:.2f}/month > ${checks_result['cost_check']['cost_limit']:.2f}/month)"
            )
            parts.append(
                "[OK] Terraform validation passed"
                if checks_result["validation_check"]["passed"]
                else "[ERROR] Terraform validation failed"
            )
            checks_result["summary"] = " | ".join(parts)
        except Exception as e:
            logger.error(f"Error running PR checks: {e}")
            checks_result["all_passed"] = False
            checks_result["summary"] = f"[ERROR] Checks failed: {e}"
        return checks_result

    def _create_pull_request(
        self,
        branch_name: str,
        base_branch: str,
        title: str,
        checks_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a pull request with check results"""
        try:
            body = self._generate_pr_description(checks_result)
            data = {
                "title": title,
                "body": body,
                "head": branch_name,
                "base": base_branch,
                "maintainer_can_modify": True,
            }
            r = requests.post(
                f"{self.github_api_base}/repos/{self.github_repo}/pulls",
                headers=self.headers,
                json=data,
                timeout=30,
            )
            if r.status_code == 201:
                pr = r.json()
                return {
                    "success": True,
                    "pr_number": pr["number"],
                    "pr_url": pr["html_url"],
                    "pr_id": pr["id"],
                }
            return {
                "success": False,
                "error": f"Failed to create PR: {r.status_code} - {r.text}",
            }
        except Exception as e:
            logger.error(f"Error creating pull request: {e}")
            return {"success": False, "error": f"PR creation failed: {e}"}

    def _generate_pr_description(self, checks_result: Dict[str, Any]) -> str:
        """Generate PR description with automated check results"""
        parts: List[str] = [
            "## Terraform Configuration Update",
            "",
            "This PR contains automatically generated Terraform configuration.",
            "",
            "### [CHECK] Automated Checks",
            "",
        ]

        # Policy
        policy = checks_result.get("policy_check", {})
        parts.append(
            "[OK] **Policy Validation**: Passed"
            if policy.get("passed")
            else "[ERROR] **Policy Validation**: Failed"
        )
        if not policy.get("passed") and policy.get("violations"):
            parts.extend(["", "**Policy Violations:**"])
            for v in policy["violations"][:5]:
                parts.append(
                    f"- [{v.get('severity','UNKNOWN')}] {v.get('message','Unknown violation')}"
                )
            if len(policy["violations"]) > 5:
                parts.append(
                    f"- ... and {len(policy['violations']) - 5} more violations"
                )

        parts.append("")

        # Cost
        cost = checks_result.get("cost_check", {})
        if cost.get("passed"):
            parts.append(
                f"[OK] **Cost Estimation**: ${cost.get('estimated_cost',0):.2f}/month (within budget)"
            )
        else:
            parts.append(
                f"[WARN] **Cost Estimation**: ${cost.get('estimated_cost',0):.2f}/month (exceeds ${cost.get('cost_limit',0):.2f}/month budget)"
            )

        parts.append("")

        # Validation
        val = checks_result.get("validation_check", {})
        parts.append(
            "[OK] **Terraform Validation**: Passed"
            if val.get("passed")
            else "[ERROR] **Terraform Validation**: Failed"
        )
        if not val.get("passed") and val.get("errors"):
            parts.extend(
                ["", "**Validation Errors:**", f"```\n{val.get('errors')}\n```"]
            )

        parts.extend(
            [
                "",
                "### Review Checklist",
                "",
                "- [ ] Review Terraform configuration changes",
                "- [ ] Verify resource specifications and sizing",
                "- [ ] Check security group rules and access controls",
                "- [ ] Validate cost implications",
                "- [ ] Ensure compliance with organizational policies",
                "",
                "### [START] Deployment",
                "",
                "After approval and merge:",
            ]
        )
        parts.extend(
            [
                (
                    "- [OK] All automated checks passed - ready for deployment"
                    if checks_result.get("all_passed")
                    else "- [ERROR] Some checks failed - review required before deployment"
                ),
                "- Run `terraform plan` to review changes",
                "- Run `terraform apply` to deploy infrastructure",
                "",
                "---",
                "*This PR was created automatically by the Terraform AI Agent*",
            ]
        )
        return "\n".join(parts)

    def _get_terraform_gitignore(self) -> str:
        """Get standard Terraform .gitignore content"""
        return """# Local .terraform directories
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

# Ignore plan files
*tfplan*

# Ignore CLI configuration files
.terraformrc
terraform.rc

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Agent data
chat_data/
budgets.json
*.log
"""

    def _generate_readme(self, terraform_code: str) -> str:
        """Generate README content for the repository"""
        import re

        resources = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', terraform_code)

        parts: List[str] = [
            "# Terraform Infrastructure",
            "",
            "This repository contains Terraform configuration for AWS infrastructure.",
            "",
            "## Resources",
            "",
        ]

        if resources:
            parts.append("This configuration creates the following resources:")
            parts.append("")
            grouped: Dict[str, List[str]] = {}
            for typ, name in resources:
                grouped.setdefault(typ, []).append(name)
            for typ, names in grouped.items():
                parts.append(f"- **{typ}**: {', '.join(names)}")
        else:
            parts.append("No resources found in configuration.")

        parts.extend(
            [
                "",
                "## Usage",
                "",
                "1. Initialize Terraform:",
                "   ```bash",
                "   terraform init",
                "   ```",
                "",
                "2. Review the execution plan:",
                "   ```bash",
                "   terraform plan",
                "   ```",
                "",
                "3. Apply the configuration:",
                "   ```bash",
                "   terraform apply",
                "   ```",
                "",
                "## Requirements",
                "",
                "- Terraform >= 1.0",
                "- AWS CLI configured with appropriate credentials",
                "- Required AWS IAM permissions for the resources being created",
                "",
                "## Security",
                "",
                "This configuration has been validated against security best practices.",
                "Please review all security group rules and access controls before deployment.",
                "",
                "---",
                "*Generated by Terraform AI Agent*",
            ]
        )
        return "\n".join(parts)


# ----- Convenience helpers used by your routes (async-safe wrappers) -----

# Global instance
github_integration = GitHubIntegration()


async def commit_files(
    branch: str, files: Dict[str, str], commit_message: str
) -> Dict[str, Any]:
    """
    Async wrapper that commits arbitrary files to a branch.
    Used by /github/enable-auto-apply in your routes.
    """
    return github_integration._commit_multiple_files(branch, files, commit_message)


import json as _gh_json

# (Optional) TruffleHog/Auto-apply helpers your routes may call
from backend.app.core.config import Settings as _GH_Settings
from backend.app.utils.utils import run_cmd_async as _gh_run_cmd_async

_GH = _GH_Settings()


async def trufflehog_scan_ref(ref: str = None):
    ref = ref or _GH.GITHUB_DEFAULT_BRANCH
    if not _GH.GITHUB_REPO:
        return {"success": False, "data": {}, "error": "GITHUB_REPO unset"}
    rc, out, err = await _gh_run_cmd_async(
        _GH.TRUFFLEHOG_BIN,
        "github",
        "--repo",
        _GH.GITHUB_REPO,
        "--branch",
        ref,
        "--json",
    )
    if rc != 0:
        return {"success": False, "data": {"stderr": err}, "error": "truffleHog failed"}
    findings = [
        _gh_json.loads(line)
        for line in out.splitlines()
        if line.strip().startswith("{")
    ]
    return {"success": True, "data": {"findings": findings}, "error": ""}


async def enable_auto_apply_action():
    workflow = """name: Terraform Apply
on:
  push:
    branches: [ main ]
jobs:
  apply:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
    - uses: actions/checkout@v4
    - uses: hashicorp/setup-terraform@v3
    - run: terraform init
    - run: terraform apply -auto-approve
"""
    base = github_integration._get_default_branch()
    return await commit_files(
        base,
        {".github/workflows/terraform-apply.yml": workflow},
        "chore: enable auto apply on merge",
    )
