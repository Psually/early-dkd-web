"""Patient-free check of the hosted WSGI endpoints."""
from __future__ import annotations

from io import BytesIO
import json
import math

from app.wsgi import application
from scripts.self_test import EXPECTED, SYNTHETIC


def request(method: str, path: str, payload: dict | None = None) -> tuple[str, bytes]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    environ = {"REQUEST_METHOD": method, "PATH_INFO": path,
               "CONTENT_LENGTH": str(len(body)), "wsgi.input": BytesIO(body)}
    result = {}

    def start_response(status, headers):
        result["status"] = status
        result["headers"] = dict(headers)

    data = b"".join(application(environ, start_response))
    assert int(result["headers"]["Content-Length"]) == len(data)
    return result["status"], data


def main() -> None:
    status, health = request("GET", "/api/health")
    assert status == "200 OK" and json.loads(health)["ok"] is True
    status, page = request("GET", "/")
    assert status == "200 OK" and b"<html" in page.lower()
    status, data = request("POST", "/api/predict", SYNTHETIC)
    result = json.loads(data)
    assert status == "200 OK"
    assert math.isclose(result["probability"], EXPECTED, abs_tol=1e-8)
    assert len(result["all_contributions"]) == 31
    assert result["waterfall_png"].startswith("data:image/png;base64,")
    print("PASS: WSGI health, UI, prediction, and waterfall")


if __name__ == "__main__":
    main()
