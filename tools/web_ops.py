"""Web tools: Google search, YouTube playback, URL opener."""
import re
import webbrowser
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from tools.registry import register


def _first_youtube_video_id(query: str) -> str | None:
    url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
    try:
        with urlopen(req, timeout=6) as resp:
            html = resp.read().decode("utf-8", "ignore")
    except Exception:
        return None
    m = re.search(r'"videoId":"([\w-]{11})"', html)
    return m.group(1) if m else None


@register(
    "Search Google for a query in the default web browser.",
    {"query": {"type": "string", "description": "What to search for."}},
    ["query"],
)
def google_search(query: str) -> str:
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(query))
    return f"Success: opened Google results for '{query}'."


@register(
    "Play a video or song on YouTube. Opens the top result for the query in the browser.",
    {"query": {"type": "string", "description": "Video/song name or topic."}},
    ["query"],
)
def play_youtube(query: str) -> str:
    vid = _first_youtube_video_id(query)
    if vid:
        webbrowser.open(f"https://www.youtube.com/watch?v={vid}")
        return f"Success: playing top YouTube result for '{query}'."
    webbrowser.open("https://www.youtube.com/results?search_query=" + quote_plus(query))
    return f"Success: opened YouTube search results for '{query}' (couldn't pick the top video)."


@register(
    "Open a website URL in the default browser.",
    {"url": {"type": "string", "description": "Website address, e.g. github.com"}},
    ["url"],
)
def open_url(url: str) -> str:
    if "://" not in url:
        url = "https://" + url
    if urlparse(url).scheme not in ("http", "https"):
        return "Error: only http and https links are allowed."
    webbrowser.open(url)
    return f"Success: opened {url}."
