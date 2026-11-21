"""`cozmo trace` / `/trace`: a local page to read traces and eval runs.

Standard library only. It binds to 127.0.0.1, answers only requests addressed to
localhost, and reads the files under `.cozmo/` fresh on every request.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from cozmo.infra.telemetry import trace_reader

DEFAULT_PORT = 7788
_PAGE = Path(__file__).with_name("dashboard.html")
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _handler(store: Path) -> type[BaseHTTPRequestHandler]:
    traces = store / "traces.jsonl"
    evals = store / "evals"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # keep the terminal quiet
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            # Another website could point a DNS name at 127.0.0.1; only answer to localhost.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in _LOCAL_HOSTS:
                self._send(403, b"Forbidden", "text/plain")
                return
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/turns":
                self._json(
                    {"workspace": str(store.parent), "turns": trace_reader.list_turns(traces)}
                )
            elif url.path == "/api/turn":
                detail = trace_reader.turn_detail(traces, q.get("id", ""))
                self._json(detail or {"error": "No turn has that id."}, 200 if detail else 404)
            elif url.path == "/api/evals":
                self._json({"runs": trace_reader.list_eval_runs(evals)})
            elif url.path == "/api/eval":
                run = trace_reader.eval_run(evals, q.get("id", ""))
                self._json(run or {"error": "No eval run has that id."}, 200 if run else 404)
            else:
                self._send(404, b"Not found", "text/plain")

    return Handler


def start_server(workdir: Path, *, port: int = DEFAULT_PORT) -> tuple[ThreadingHTTPServer, str]:
    """Bind the dashboard to *port* (or any free port if that one is busy)."""
    store = workdir.resolve() / ".cozmo"
    handler = _handler(store)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://localhost:{server.server_address[1]}"
    return server, url


_background: dict[str, str] = {}


def open_in_background(workdir: Path, *, open_browser: bool = True) -> str:
    """For `/trace` in the REPL: start once per workspace, then just reopen the page."""
    key = str(workdir.resolve())
    url = _background.get(key)
    if url is None:
        server, url = start_server(workdir)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _background[key] = url
    if open_browser:
        webbrowser.open(url)
    return url
