"""Research web UI for the frozen early-DKD model."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .predictor import HERE, InputError, Predictor


STATIC = HERE / "static"
PREDICTOR = Predictor()
MAX_BODY = 32_768


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Never persist URLs, request bodies, predictions, or patient values.
        pass

    def response(self, code: int, data: bytes, mime: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, code: int, value: dict) -> None:
        self.response(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.json_response(200, {"ok": True, "model": PREDICTOR.bundle["version"]})
        if path == "/api/schema":
            return self.json_response(200, PREDICTOR.schema())
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.css": ("app.css", "text/css; charset=utf-8"),
                 "/validation.css": ("validation.css", "text/css; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path not in files:
            return self.json_response(404, {"error": "Page not found."})
        filename, mime = files[path]
        return self.response(200, (STATIC / filename).read_bytes(), mime)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            return self.json_response(404, {"error": "Endpoint not found."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise InputError("The request body is empty or too large.")
            payload = json.loads(self.rfile.read(length))
            result = PREDICTOR.predict(payload)
        except (InputError, ValueError, json.JSONDecodeError) as error:
            return self.json_response(400, {"error": str(error)})
        return self.json_response(200, result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("DKD_WEB_PORT", "8765")))
    host = os.environ.get("DKD_WEB_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Early-DKD predictor listening on {host}:{port}", flush=True)
    server.serve_forever()
