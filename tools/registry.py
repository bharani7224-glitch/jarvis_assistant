"""Tool registry: decorator, JSON schemas for the LLM, and a safe dispatcher."""
import logging
from dataclasses import dataclass
from typing import Callable

import config

log = logging.getLogger("jarvis.tools")


@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    required: list
    func: Callable
    dangerous: bool = False


REGISTRY: dict[str, Tool] = {}
_confirm_handler: Callable[[str], bool] | None = None


def register(description: str, properties: dict | None = None, required: list | None = None,
             dangerous: bool = False, name: str | None = None):
    def deco(func: Callable) -> Callable:
        tool_name = name or func.__name__
        REGISTRY[tool_name] = Tool(tool_name, description, properties or {}, required or [], func, dangerous)
        return func
    return deco


def set_confirm_handler(handler: Callable[[str], bool]) -> None:
    """handler(question) -> True if the user confirmed (e.g. by voice)."""
    global _confirm_handler
    _confirm_handler = handler


def get_schemas() -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": {"type": "object", "properties": t.properties, "required": t.required},
        }
        for t in REGISTRY.values()
    ]


def execute(name: str, args: dict) -> str:
    tool = REGISTRY.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'."
    log.info("TOOL %s(%s)", name, args)

    if tool.dangerous and config.REQUIRE_CONFIRMATION_FOR_DANGEROUS:
        if _confirm_handler is None or not _confirm_handler(f"Are you sure you want me to run {name.replace('_', ' ')}?"):
            return "Cancelled: the user did not confirm."

    try:
        result = tool.func(**args)
        return str(result)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"
    except Exception as exc:
        log.exception("Tool %s failed", name)
        return f"Error: {type(exc).__name__}: {exc}"
