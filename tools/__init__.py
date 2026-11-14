"""Tool package: importing it registers every tool. Public API: get_schemas, execute, set_confirm_handler."""
from tools.registry import execute, get_schemas, register, set_confirm_handler  # noqa: F401

# Importing the modules runs their @register decorators.
from tools import gui_ops, system_ops, web_ops  # noqa: F401,E402

__all__ = ["execute", "get_schemas", "register", "set_confirm_handler"]
