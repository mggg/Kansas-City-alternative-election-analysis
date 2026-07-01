"""
Small web server for the config builder UI.

Serves the config-builder page, lists geodata files and their columns, and
accepts a POST of the assembled config JSON, writing it into the project's
``configs/`` directory.

Run from the project root (use the project venv so geopandas' fast schema
readers are importable — needed only for column introspection):

    python pipeline-config/server.py
    # or: .venv/bin/python pipeline-config/server.py

then open http://localhost:8000 in a browser. The page and config-saving work
on the standard library alone; the geodata column dropdowns additionally need
``pyogrio`` / ``pyarrow`` (already project dependencies).
"""

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# pipeline-config/server.py -> project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

# File extensions we treat as selectable geodata sources.
GEODATA_EXTS = {".gpkg", ".parquet", ".shp", ".geojson"}

HOST = "localhost"
PORT = 8000


def _safe_filename(name: str) -> str:
    """Turn an arbitrary run/file name into a safe ``<name>.json`` filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip()) or "config"
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    return f"{stem}.json"


def _list_geodata_files() -> list:
    """Return geodata files under data/ as project-relative './data/...' paths."""
    if not DATA_DIR.exists():
        return []
    files = [
        p for p in sorted(DATA_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in GEODATA_EXTS
    ]
    return [f"./{p.relative_to(PROJECT_ROOT).as_posix()}" for p in files]


def _resolve_within_data(path_str: str) -> Path:
    """Resolve a client-supplied path and confirm it sits inside data/."""
    candidate = (PROJECT_ROOT / path_str).resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in candidate.parents and candidate != data_root:
        raise ValueError("Path must be inside the data/ directory.")
    if not candidate.is_file():
        raise FileNotFoundError(f"No such file: {path_str}")
    return candidate


def _read_columns(path_str: str) -> list:
    """Return the column/field names of a geodata file without loading rows."""
    path = _resolve_within_data(path_str)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        names = list(pq.ParquetFile(path).schema.names)
    else:
        import pyogrio
        names = list(pyogrio.read_info(path)["fields"])
    # Drop geometry columns; they aren't valid population-count fields.
    return [n for n in names if n.lower() not in ("geometry", "geom")]


class ConfigBuilderHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        route = urlparse(self.path)
        path = route.path
        if path in ("/", "/index.html"):
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
        elif path == "/api/configs":
            names = sorted(p.name for p in CONFIGS_DIR.glob("*.json"))
            self._send_json(200, {"configs": names})
        elif path == "/api/config":
            name = parse_qs(route.query).get("name", [""])[0]
            if not name:
                self._send_json(400, {"ok": False, "error": "Missing 'name'."})
                return
            # basename only — never read outside the configs directory.
            target = CONFIGS_DIR / Path(name).name
            if not target.is_file():
                self._send_json(404, {"ok": False, "error": f"No such config: {name}"})
                return
            try:
                cfg = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self._send_json(400, {"ok": False, "error": f"Invalid JSON in {name}: {exc}"})
            else:
                self._send_json(200, {"ok": True, "config": cfg})
        elif path == "/api/geodata-files":
            self._send_json(200, {"files": _list_geodata_files()})
        elif path == "/api/geodata-columns":
            file_path = parse_qs(route.query).get("path", [""])[0]
            if not file_path:
                self._send_json(400, {"ok": False, "error": "Missing 'path'."})
                return
            try:
                cols = _read_columns(file_path)
            except (ValueError, FileNotFoundError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            except ImportError:
                self._send_json(500, {
                    "ok": False,
                    "error": "pyogrio/pyarrow not available — run the server with "
                             "the project venv to read columns.",
                })
            except Exception as exc:  # unreadable/corrupt file
                self._send_json(500, {"ok": False, "error": f"Could not read columns: {exc}"})
            else:
                self._send_json(200, {"ok": True, "columns": cols})
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
