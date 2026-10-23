import asyncio
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from model_cache import read_lock, verify_files
from safeboard.middleware import RequestMiddleware, discard_small_unauthorized_body
from safeboard.privacy import MetadataFormatter

SENTINEL = "PRIVATE_PROMPT_RESPONSE_TOKEN_MUST_NOT_BE_LOGGED"


class PrivacyTests(unittest.TestCase):
    def test_destructor_and_atexit_exceptions_are_sanitized(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "app")}
        snippets = [f"class E:\n def __del__(self): raise ValueError({SENTINEL!r})\nx=E(); del x",
                    f"import atexit\ndef cleanup(): raise RuntimeError({SENTINEL!r})\natexit.register(cleanup)"]
        for code in snippets:
            with self.subTest(code=code):
                child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
                self.assertNotIn(SENTINEL, child.stdout + child.stderr)
                self.assertIn("unhandled_error", child.stderr)

    def test_direct_tracebacks_are_sanitized_in_child_process(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "app")}
        code = f"import traceback;\ntry: raise ValueError({SENTINEL!r})\nexcept ValueError: traceback.print_exc()\nraise RuntimeError({SENTINEL!r})"
        child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertNotEqual(child.returncode, 0)
        self.assertNotIn(SENTINEL, child.stdout + child.stderr)
        self.assertIn("unhandled_error", child.stderr)

    def test_exception_arguments_and_traceback_are_not_formatted(self):
        try:
            raise ValueError(SENTINEL)
        except ValueError:
            record = logging.LogRecord("vllm.engine", logging.ERROR, "example.py", 1,
                                       "request=%s", (SENTINEL,), sys.exc_info())
        formatted = MetadataFormatter().format(record)
        self.assertNotIn(SENTINEL, formatted)
        self.assertNotIn("Traceback", formatted)
        self.assertEqual(json.loads(formatted)["level"], "ERROR")

    def test_audit_rejects_unapproved_fields(self):
        record = logging.LogRecord("safeboard.audit", logging.INFO, "", 1, SENTINEL, (), None)
        record.audit = {"request_id": "a" * 32, "status": 200, "prompt": SENTINEL,
                        "response": SENTINEL, "authorization": SENTINEL}
        parsed = json.loads(MetadataFormatter().format(record))
        self.assertNotIn(SENTINEL, json.dumps(parsed))
        self.assertEqual(parsed["status"], 200)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_unauthorized_body_is_discarded_without_logging(self):
        with tempfile.TemporaryDirectory() as folder:
            secret = Path(folder) / "token"
            secret.write_text("s" * 48)
            body = SENTINEL.encode()
            received, output = [], []
            async def app(*args):
                raise AssertionError("Unauthorized app must not run")
            async def receive():
                received.append(True)
                return {"type": "http.request", "body": body, "more_body": False}
            async def send(message):
                output.append(message)
            with patch.dict(os.environ, {"API_TOKEN_FILE": str(secret)}):
                middleware = RequestMiddleware(app)
            await middleware({"type": "http", "path": "/invocations", "method": "POST",
                              "headers": [(b"content-length", str(len(body)).encode())]}, receive, send)
            self.assertEqual(received, [True])
            self.assertEqual(output[0]["status"], 401)
            self.assertNotIn(SENTINEL, repr(output))

    async def test_unauthorized_body_drain_has_size_and_time_bounds(self):
        async def forbidden():
            raise AssertionError("Large body must not be read")
        await discard_small_unauthorized_body({"headers": [(b"content-length", b"32769")]}, forbidden)
        async def slow():
            await asyncio.sleep(10)
        await asyncio.wait_for(discard_small_unauthorized_body(
            {"headers": [(b"content-length", b"1")]}, slow), timeout=1)

    async def run_request(self, *, token=True, fail=False, path="/v1/chat/completions"):
        with tempfile.TemporaryDirectory() as folder:
            secret = Path(folder) / "token"
            secret.write_text("s" * 48)
            called = []
            output = []
            log = io.StringIO()
            handler = logging.StreamHandler(log)
            handler.setFormatter(MetadataFormatter())
            logger = logging.getLogger("safeboard.audit")
            old_level = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            try:
                async def app(scope, receive, send):
                    called.append(True)
                    if fail:
                        raise ValueError(SENTINEL)
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    await send({"type": "http.response.body", "body": SENTINEL.encode(), "more_body": True})
                    await asyncio.sleep(0)
                    await send({"type": "http.response.body", "body": b"done", "more_body": False})

                async def receive():
                    raise AssertionError("Audit/authentication middleware must not read the body")

                async def send(message):
                    output.append(message)

                headers = [(b"authorization", b"Bearer " + b"s" * 48)] if token else []
                scope = {"type": "http", "path": path, "method": "POST", "headers": headers,
                         "query_string": SENTINEL.encode()}
                with patch.dict(os.environ, {"API_TOKEN_FILE": str(secret)}):
                    middleware = RequestMiddleware(app)
                if fail:
                    with self.assertRaises(ValueError):
                        await middleware(scope, receive, send)
                else:
                    await middleware(scope, receive, send)
                return called, output, log.getvalue()
            finally:
                logger.removeHandler(handler)
                logger.setLevel(old_level)

    async def test_streaming_logs_once_without_content(self):
        called, output, log = await self.run_request()
        self.assertTrue(called)
        self.assertEqual(len(log.splitlines()), 1)
        self.assertEqual(json.loads(log)["status"], 200)
        self.assertNotIn(SENTINEL, log)
        self.assertIn((b"x-request-id", json.loads(log)["request_id"].encode()), output[0]["headers"])

    async def test_unauthenticated_alternate_inference_route_is_blocked(self):
        called, output, log = await self.run_request(token=False, path="/invocations")
        self.assertFalse(called)
        self.assertEqual(output[0]["status"], 401)
        self.assertEqual(json.loads(log)["status"], 401)

    async def test_exception_still_records_request_without_private_data(self):
        _, _, log = await self.run_request(fail=True)
        self.assertEqual(json.loads(log)["status"], 500)
        self.assertEqual(json.loads(log)["error_type"], "ValueError")
        self.assertNotIn(SENTINEL, log)


class CacheTests(unittest.TestCase):
    def test_size_and_hash_are_both_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            data = b"valid weights"
            file = Path(folder) / "weights.bin"
            file.write_bytes(data)
            lock = {"files": {"weights.bin": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
            self.assertTrue(verify_files(Path(folder), lock))
            file.write_bytes(b"other weights")
            self.assertFalse(verify_files(Path(folder), lock))

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "lock.json"
            file.write_text(json.dumps({"model_id": "test/model", "revision": "a" * 40,
                                       "files": {"../weights": {"size": 1, "sha256": "b" * 64}}}))
            with self.assertRaises(ValueError):
                read_lock(file)


if __name__ == "__main__":
    unittest.main()
