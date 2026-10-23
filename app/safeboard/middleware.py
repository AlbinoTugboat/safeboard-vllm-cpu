"""ASGI authentication and streaming-safe access audit without reading request bodies."""
import asyncio
import hmac
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger("safeboard.audit")
OPERATIONS = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/models": "models",
    "/tokenize": "tokenize",
    "/metrics": "metrics",
}


async def discard_small_unauthorized_body(scope, receive):
    """Finish short HTTP requests before 401 without buffering or logging bodies.

    An early close with unread bytes can reset the TCP connection/port-forward.
    Large, unknown-length and slow bodies are never drained without a bound.
    """
    length = dict(scope.get("headers", [])).get(b"content-length", b"0")
    try:
        length = int(length)
    except ValueError:
        return
    if not 0 < length <= 32768:
        return
    try:
        async with asyncio.timeout(0.25):
            consumed = 0
            while consumed <= 32768:
                message = await receive()
                if message["type"] != "http.request":
                    break
                consumed += len(message.get("body", b""))
                if not message.get("more_body", False):
                    break
    except TimeoutError:
        pass


class RequestMiddleware:
    def __init__(self, app):
        self.app = app
        self.token = Path(os.environ.get("API_TOKEN_FILE", "/secrets/api-token")).read_bytes().strip()
        if len(self.token) < 32:
            raise RuntimeError("API token must contain at least 32 bytes")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == "/health" and scope["method"] in ("GET", "HEAD"):
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        status = 500
        error_type = None

        async def audit_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            header = dict(scope.get("headers", [])).get(b"authorization", b"")
            parts = header.split(b" ", 1)
            authenticated = (len(parts) == 2 and parts[0].lower() == b"bearer"
                             and hmac.compare_digest(parts[1], self.token))
            if not authenticated:
                await discard_small_unauthorized_body(scope, receive)
                await audit_send({"type": "http.response.start", "status": 401,
                                  "headers": [(b"content-type", b"application/json"),
                                              (b"www-authenticate", b"Bearer")]})
                await audit_send({"type": "http.response.body", "body": b'{"error":"Unauthorized"}'})
                return
            await self.app(scope, receive, audit_send)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            audit = {"request_id": request_id, "operation": OPERATIONS.get(scope["path"], "other"),
                     "status": status, "duration_ms": round((time.perf_counter() - started) * 1000, 3)}
            if error_type:
                audit["error_type"] = error_type
            logger.info("request_complete", extra={"audit": audit})
