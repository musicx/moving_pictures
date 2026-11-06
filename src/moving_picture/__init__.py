import os
import threading
import webbrowser
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Read KEY=VALUE lines from the project's .env without overriding the real environment."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    import uvicorn

    threading.Timer(1.5, webbrowser.open, [f"http://localhost:{PORT}"]).start()
    uvicorn.run("moving_picture.app:app", host="127.0.0.1", port=PORT)
