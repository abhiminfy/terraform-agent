# -*- coding: utf-8 -*-
class ToolExecutionError(Exception):
    """Raised when executing an external tool/CLI fails."""


class IntegrationError(Exception):
    """Raised for third-party integration errors (e.g., GitHub)."""


class AuthError(Exception):
    """Raised for auth-related problems (e.g., JWT)."""


class ValidationError(Exception):
    """Raised for validation issues (e.g., HCL parse errors)."""
