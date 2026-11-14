"""LLM client with native tool calling (Gemini or Claude) and short-term conversation memory."""
import logging
from typing import Callable

import config
import tools

log = logging.getLogger("jarvis.brain")


def _trim(history: list, is_user_turn: Callable[[object], bool], limit: int) -> None:
    """Drop the oldest messages, cutting only at a plain user turn so tool-call pairs stay intact."""
    if len(history) <= limit:
        return
    for i in range(len(history) - limit, len(history)):
        if is_user_turn(history[i]):
            del history[:i]
            return


# ====================================================================== Gemini
class _GeminiBackend:
    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        decls = []
        for s in tools.get_schemas():
            d = {"name": s["name"], "description": s["description"]}
            if s["parameters"]["properties"]:          # Gemini rejects empty OBJECT schemas
                d["parameters"] = s["parameters"]
            decls.append(d)
        self._config = types.GenerateContentConfig(
            system_instruction=config.SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=decls)],
            max_output_tokens=config.LLM_MAX_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._history: list = []

    def _is_user_text(self, content) -> bool:
        return content.role == "user" and any(getattr(p, "text", None) for p in content.parts or [])

    def reset(self) -> None:
        self._history.clear()

    def respond(self, user_text: str) -> str:
        t = self._types
        self._history.append(t.Content(role="user", parts=[t.Part.from_text(text=user_text)]))
        _trim(self._history, self._is_user_text, config.MAX_HISTORY_MESSAGES)

        for _ in range(config.MAX_TOOL_ROUNDS):
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL, contents=self._history, config=self._config)
            if not resp.candidates or not resp.candidates[0].content:
                return "Sorry, I didn't get a response from the model."
            content = resp.candidates[0].content
            self._history.append(content)

            calls = [p.function_call for p in (content.parts or []) if p.function_call]
            if not calls:
                text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None))
                return text.strip() or "Done."

            results = []
            for call in calls:
                out = tools.execute(call.name, dict(call.args or {}))
                results.append(t.Part.from_function_response(name=call.name, response={"result": out}))
            self._history.append(t.Content(role="user", parts=results))
        return "That took too many steps, so I stopped."


# ====================================================================== Claude
class _ClaudeBackend:
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._tools = [
            {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
            for s in tools.get_schemas()
        ]
        self._history: list = []

    @staticmethod
    def _is_user_text(msg) -> bool:
        return msg["role"] == "user" and isinstance(msg["content"], str)

    def reset(self) -> None:
        self._history.clear()

    def respond(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        _trim(self._history, self._is_user_text, config.MAX_HISTORY_MESSAGES)

        for _ in range(config.MAX_TOOL_ROUNDS):
            resp = self._client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                system=config.SYSTEM_PROMPT,
                tools=self._tools,
                messages=self._history,
            )
            self._history.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return text.strip() or "Done."

            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = tools.execute(block.name, dict(block.input or {}))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            self._history.append({"role": "user", "content": results})
        return "That took too many steps, so I stopped."


# ====================================================================== Public
class Brain:
    def __init__(self) -> None:
        self._backend = _ClaudeBackend() if config.LLM_PROVIDER == "claude" else _GeminiBackend()
        log.info("Brain ready: provider=%s, %d tools", config.LLM_PROVIDER, len(tools.get_schemas()))

    def reset(self) -> None:
        self._backend.reset()

    def respond(self, user_text: str) -> str:
        try:
            return self._backend.respond(user_text)
        except Exception as exc:  # network, quota, auth...
            log.exception("LLM call failed")
            name = type(exc).__name__
            return f"Sorry, I couldn't reach the AI service. {name}."
