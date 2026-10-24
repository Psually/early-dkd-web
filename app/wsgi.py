"""WSGI entry point for hosts such as PythonAnywhere."""
from __future__ import annotations

import json

from .predictor import HERE, InputError, Predictor


PREDICTOR = Predictor()
STATIC = HERE / "static"
MAX_BODY = 32_768
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/validation.css": ("validation.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def application(environ, start_response):
    """Serve the UI and run prediction without persisting submitted values."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    status = "200 OK"
    mime = "application/json; charset=utf-8"

    if method == "GET" and path == "/api/health":
        body = json.dumps({"ok": True, "model": PREDICTOR.bundle["version"]}).encode("utf-8")
    elif method == "GET" and path == "/api/schema":
        body = json.dumps(PREDICTOR.schema(), ensure_ascii=False, allow_nan=False).encode("utf-8")
    elif method == "GET" and path in FILES:
        filename, mime = FILES[path]
        body = (STATIC / filename).read_bytes()
    elif method == "POST" and path == "/api/predict":
        try:
            length = int(environ.get("CONTENT_LENGTH") or "0")
            if not 0 < length <= MAX_BODY:
                raise InputError("The request body is empty or too large.")
            payload = json.loads(environ["wsgi.input"].read(length))
            result = PREDICTOR.predict(payload)
            body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (InputError, ValueError, json.JSONDecodeError) as error:
            status = "400 Bad Request"
            body = json.dumps({"error": str(error)}).encode("utf-8")
    else:
        status = "404 Not Found"
        body = b'{"error":"Endpoint not found."}'

    headers = [
        ("Content-Type", mime),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'"),
    ]
    start_response(status, headers)
    return [body]
