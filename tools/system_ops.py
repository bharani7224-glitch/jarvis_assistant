"""System tools: launch/close apps, time, battery, CPU/RAM stats, shutdown (confirmed)."""
import datetime as dt
import subprocess
import sys

import psutil

from tools.registry import register

IS_WINDOWS = sys.platform.startswith("win")

# spoken name -> what `start` should launch
APP_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome",
    "edge": "msedge", "microsoft edge": "msedge",
    "firefox": "firefox",
    "notepad": "notepad",
    "calculator": "calc", "calc": "calc",
    "explorer": "explorer", "file explorer": "explorer", "files": "explorer",
    "vscode": "code", "vs code": "code", "visual studio code": "code", "code": "code",
    "cmd": "cmd", "command prompt": "cmd",
    "powershell": "powershell",
    "terminal": "wt", "windows terminal": "wt",
    "task manager": "taskmgr",
    "paint": "mspaint",
    "settings": "ms-settings:",
    "spotify": "spotify",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
    "snipping tool": "snippingtool",
}

# spoken name -> process image names to terminate
PROCESS_ALIASES = {
    "chrome": ["chrome.exe"], "google chrome": ["chrome.exe"],
    "edge": ["msedge.exe"], "microsoft edge": ["msedge.exe"],
    "firefox": ["firefox.exe"],
    "notepad": ["notepad.exe"],
    "calculator": ["calculatorapp.exe", "calculator.exe", "calc.exe"],
    "vscode": ["code.exe"], "vs code": ["code.exe"], "visual studio code": ["code.exe"], "code": ["code.exe"],
    "spotify": ["spotify.exe"],
    "paint": ["mspaint.exe"],
    "task manager": ["taskmgr.exe"],
    "word": ["winword.exe"], "excel": ["excel.exe"], "powerpoint": ["powerpnt.exe"],
}

_PROTECTED = {"explorer.exe", "system", "csrss.exe", "winlogon.exe", "svchost.exe", "wininit.exe", "services.exe"}


def _windows_only() -> str | None:
    return None if IS_WINDOWS else "Error: this tool only works on Windows."


@register(
    "Open or launch a desktop application by name (e.g. chrome, notepad, vscode, calculator, spotify, explorer).",
    {"app_name": {"type": "string", "description": "Spoken app name, e.g. 'vs code'."}},
    ["app_name"],
)
def open_app(app_name: str) -> str:
    if (err := _windows_only()):
        return err
    key = app_name.strip().lower()
    target = APP_ALIASES.get(key, key)
    if any(c in target for c in '&|<>^"%'):
        return "Error: invalid app name."
    proc = subprocess.run(["cmd", "/c", "start", "", target], capture_output=True, text=True)
    if proc.returncode != 0:
        return f"Error: could not launch '{app_name}'. {proc.stderr.strip()}"
    return f"Success: launched {app_name}."


@register(
    "Close a running desktop application by name (e.g. chrome, notepad, spotify).",
    {"app_name": {"type": "string", "description": "Spoken app name."}},
    ["app_name"],
)
def close_app(app_name: str) -> str:
    key = app_name.strip().lower()
    targets = {n.lower() for n in PROCESS_ALIASES.get(key, [])} or {key if key.endswith(".exe") else key + ".exe"}
    targets -= _PROTECTED
    killed = 0
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower()
        if name in targets:
            try:
                p.terminate()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return f"Success: closed {app_name} ({killed} processes)." if killed else f"Error: {app_name} is not running."


@register("List the running applications using the most memory.")
def list_running_apps() -> str:
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["name"], p.info["memory_info"].rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    agg: dict[str, int] = {}
    for name, rss in procs:
        agg[name] = agg.get(name, 0) + rss
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return "; ".join(f"{n} {r / 1e6:.0f} MB" for n, r in top)


@register("Get the current local date and time.")
def get_time() -> str:
    return dt.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@register("Get battery percentage and whether the laptop is charging.")
def get_battery() -> str:
    b = psutil.sensors_battery()
    if b is None:
        return "No battery detected (desktop PC)."
    state = "charging" if b.power_plugged else "on battery"
    mins = ""
    if not b.power_plugged and b.secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED):
        mins = f", about {int(b.secsleft // 60)} minutes left"
    return f"Battery {b.percent:.0f}%, {state}{mins}."


@register("Get current CPU, RAM and disk usage.")
def get_system_stats() -> str:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if IS_WINDOWS else "/")
    return (f"CPU {cpu:.0f}%, RAM {ram.percent:.0f}% of {ram.total / 1e9:.0f} GB, "
            f"disk {disk.percent:.0f}% used.")


@register("Lock the Windows screen.")
def lock_pc() -> str:
    if (err := _windows_only()):
        return err
    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
    return "Success: PC locked."


@register("Shut down the computer in 10 seconds. Requires voice confirmation.", dangerous=True)
def shutdown_pc() -> str:
    if (err := _windows_only()):
        return err
    subprocess.run(["shutdown", "/s", "/t", "10"], check=False)
    return "Success: shutdown scheduled in 10 seconds. (Run 'shutdown /a' to abort.)"
