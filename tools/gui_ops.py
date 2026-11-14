"""GUI tools: typing, hotkeys, window switching, volume. Act on whatever window is focused.

Note: Windows blocks synthetic input into elevated (admin) windows unless Jarvis is also run as admin.
pyautogui's fail-safe is left ON: slam the mouse into a screen corner to abort any runaway automation.
"""
import time

import pyautogui
import pygetwindow as gw
import pyperclip

from tools.registry import register

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

_KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl",
    "windows": "win", "super": "win", "meta": "win", "cmd": "win",
    "return": "enter", "escape": "esc",
    "del": "delete", "back": "backspace", "backspace": "backspace",
    "spacebar": "space", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "pgup": "pageup", "pgdn": "pagedown",
}

_WINDOW_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome",
    "edge": "edge",
    "vscode": "visual studio code", "vs code": "visual studio code", "code": "visual studio code",
    "explorer": "file explorer", "files": "file explorer",
    "terminal": "terminal", "cmd": "command prompt",
}


def _norm_key(key: str) -> str:
    k = str(key).strip().lower()
    return _KEY_ALIASES.get(k, k)


# ---------------------------------------------------------------- typing
@register(
    "Type text at the current cursor position in the focused window (uses the clipboard for speed and accuracy). "
    "Set press_enter=true to submit afterwards, e.g. for a search box or chat message.",
    {
        "text": {"type": "string", "description": "The text to type."},
        "press_enter": {"type": "boolean", "description": "Press Enter after typing. Default false."},
    },
    ["text"],
)
def type_text(text: str, press_enter: bool = False) -> str:
    if not text:
        return "Error: no text provided."
    time.sleep(0.15)  # let focus settle
    previous = ""
    try:
        previous = pyperclip.paste()
    except Exception:
        pass

    try:
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.15)
    except Exception as exc:
        if not text.isascii():
            return f"Error: clipboard paste failed ({exc}) and the text has non-ASCII characters."
        pyautogui.write(text, interval=0.01)
    finally:
        if previous:  # restore the old clipboard text (non-text contents can't be restored)
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

    if press_enter:
        time.sleep(0.05)
        pyautogui.press("enter")
    return f"Success: typed {len(text)} characters{' and pressed Enter' if press_enter else ''}."


# ---------------------------------------------------------------- hotkeys
@register(
    "Press a keyboard key or shortcut in the focused window. Pass keys in order, e.g. ['ctrl','c'], "
    "['alt','tab'], ['win','d'], ['ctrl','t'], ['enter'], ['backspace'], ['f5'].",
    {"keys": {"type": "array", "items": {"type": "string"},
              "description": "Key names; one key presses it alone, several press them together."}},
    ["keys"],
)
def press_hotkey(keys: list[str]) -> str:
    if isinstance(keys, str):
        keys = keys.split("+")
    keys = [_norm_key(k) for k in keys if str(k).strip()]
    if not keys:
        return "Error: no keys provided."
    bad = [k for k in keys if k not in pyautogui.KEYBOARD_KEYS]
    if bad:
        return f"Error: unknown key(s): {', '.join(bad)}."
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)
    return f"Success: pressed {'+'.join(keys)}."


# ---------------------------------------------------------------- windows
def _visible_windows():
    return [w for w in gw.getAllWindows() if w.title and w.title.strip()]


def _focus(win) -> bool:
    if getattr(win, "isMinimized", False):
        win.restore()
    try:
        win.activate()
    except Exception:
        # pygetwindow raises spuriously (even on success), and Windows can refuse focus changes.
        # Tapping Alt lifts the foreground lock; then retry once.
        try:
            pyautogui.press("alt")
            win.activate()
        except Exception:
            pass
    time.sleep(0.2)
    active = gw.getActiveWindow()
    return active is not None and getattr(active, "_hWnd", None) == getattr(win, "_hWnd", object())


@register(
    "Bring an open window to the front by part of its title (e.g. 'chrome', 'notepad', 'visual studio code').",
    {"title_keyword": {"type": "string", "description": "Part of the window title, case-insensitive."}},
    ["title_keyword"],
)
def switch_window(title_keyword: str) -> str:
    raw = title_keyword.strip().lower()
    if not raw:
        return "Error: no window name given."
    keyword = _WINDOW_ALIASES.get(raw, raw)
    matches = [w for w in _visible_windows() if keyword in w.title.lower()]
    if not matches:
        return f"Error: no open window matches '{title_keyword}'."
    win = matches[0]
    ok = _focus(win)
    extra = f" ({len(matches)} matches, used the first)" if len(matches) > 1 else ""
    if ok:
        return f"Success: focused '{win.title[:60]}'{extra}."
    return f"Error: found '{win.title[:60]}' but Windows would not give it focus."


@register("List the titles of currently open windows, to help pick one to switch to.")
def list_windows() -> str:
    titles = []
    for w in _visible_windows():
        t = w.title.strip()[:50]
        if t not in titles:
            titles.append(t)
    return "; ".join(titles[:15]) or "No windows found."


# ---------------------------------------------------------------- volume
_VOLUME_KEYS = {"up": "volumeup", "down": "volumedown"}


@register(
    "Control system volume. action is 'up', 'down', 'mute' or 'unmute'. Each step is about 2 percent, "
    "so steps=5 changes volume by roughly 10 percent. Mute and unmute both toggle the mute state.",
    {
        "action": {"type": "string", "enum": ["up", "down", "mute", "unmute"],
                   "description": "What to do with the volume."},
        "steps": {"type": "integer", "description": "Number of volume steps for up/down. Default 5."},
    },
    ["action"],
)
def control_volume(action: str, steps: int = 5) -> str:
    action = str(action).strip().lower()
    if action in ("mute", "unmute", "toggle", "toggle_mute"):
        pyautogui.press("volumemute")
        return "Success: toggled mute (Windows doesn't report the current mute state to me)."
    key = _VOLUME_KEYS.get(action)
    if key is None:
        return "Error: action must be 'up', 'down', 'mute' or 'unmute'."
    try:
        steps = max(1, min(int(steps), 50))
    except (TypeError, ValueError):
        steps = 5
    pyautogui.press(key, presses=steps, interval=0.02)
    return f"Success: volume {action} by about {steps * 2} percent."


# ---------------------------------------------------------------- timing helper
@register(
    "Pause for a few seconds, e.g. to let a newly opened app finish loading before typing into it.",
    {"seconds": {"type": "number", "description": "Seconds to wait, 0.5 to 10."}},
    ["seconds"],
)
def wait_seconds(seconds: float) -> str:
    try:
        seconds = max(0.5, min(float(seconds), 10.0))
    except (TypeError, ValueError):
        seconds = 1.0
    time.sleep(seconds)
    return f"Success: waited {seconds:g} seconds."
