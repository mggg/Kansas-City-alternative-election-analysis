"""
Tiny zero-dependency web server for the config builder UI.

Serves the config-builder page and accepts a POST of the assembled config
JSON, writing it into the project's ``configs/`` directory.

Run from the project root:

    python pipeline-config/server.py

then open http://localhost:8000 in a browser. Uses only the Python
standard library, so no extra packages need to be installed.
"""

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# pipeline-config/server.py -> project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

HOST = "localhost"
PORT = 8000


def _safe_filename(name: str) -> str:
    """Turn an arbitrary run/file name into a safe ``<name>.json`` filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip()) or "config"
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    return f"{stem}.json"


class ConfigBuilderHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
            except FileNotFoundError:
                self.send_error(404, "index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/configs":
            names = sorted(p.name for p in CONFIGS_DIR.glob("*.json"))
            self._send_json(200, {"configs": names})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):  # noqa: N802 (stdlib naming)
        if self.path != "/api/save-config":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return

        config = payload.get("config")
        filename = payload.get("filename") or (config or {}).get("run_name", "config")
        overwrite = bool(payload.get("overwrite", False))
        if not isinstance(config, dict):
            self._send_json(400, {"ok": False, "error": "Missing 'config' object."})
            return

        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        dest = CONFIGS_DIR / _safe_filename(str(filename))
        if dest.exists() and not overwrite:
            self._send_json(
                409,
                {
                    "ok": False,
                    "error": f"{dest.name} already exists.",
                    "filename": dest.name,
                },
            )
            return

        dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        rel = dest.relative_to(PROJECT_ROOT)
        self._send_json(200, {"ok": True, "path": str(rel), "filename": dest.name})

    def log_message(self, fmt, *args):  # keep the console quiet-ish
        return


def main():
    server = HTTPServer((HOST, PORT), ConfigBuilderHandler)
    print(f"Config builder running at http://{HOST}:{PORT}")
    print(f"Writing configs into: {CONFIGS_DIR}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
