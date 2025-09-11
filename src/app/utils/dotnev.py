"""
Minimal stub for python-dotenv's load_dotenv function.

In production, install python-dotenv to parse .env files. In this
sandboxed environment the stub simply returns without doing anything so
imports of `load_dotenv` succeed.
"""

from typing import Any, Dict, Optional


def load_dotenv(path: Optional[str] = None, *args: Any, **kwargs: Any) -> bool:
    """Stub load_dotenv that does nothing and returns False.

    Args:
        path: Optional path to .env file. Ignored in this stub.

    Returns:
        False, indicating that no .env file was loaded.
    """
    return False
