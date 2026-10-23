import contextlib
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import client
import create_secret
from check_model_quality import grade
from evidence import bind_endpoint, canonical_json_sha256, server_snapshot, text_sha256


class QualityGradingTests(unittest.TestCase):
    def test_json_booleans_are_not_confused_with_numbers(self):
        self.assertFalse(grade('{"ok":1}', "json", {"ok": True}))
        self.assertTrue(grade('{"count":3,"ok":true}', "json", {"ok": True, "count": 3}))

    def test_truncated_json_and_wrong_arithmetic_fail(self):
        self.assertFalse(grade('{"ok":', "json", {"ok": True}))
        self.assertFalse(grade("43", "number", "42"))
        self.assertTrue(grade("42.", "number", "42"))

    def test_word_grading_accepts_case_and_punctuation(self):
        self.assertTrue(grade("ПАРИЖ.", "word", "Париж"))
        self.assertFalse(grade("Париж или Лондон", "word", "Париж"))


class ClientTests(unittest.TestCase):
    def test_loopback_request_bypasses_ambient_proxy(self):
        received = {"proxy": 0, "direct": 0}
        class Listener(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received[self.server.role] += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), Listener) for _ in range(2)]
        threads = []
        for server, role in zip(servers, ("proxy", "direct")):
            server.role = role
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            threads.append(thread)
        try:
            with patch.dict(os.environ, {"http_proxy": f"http://127.0.0.1:{servers[0].server_port}",
                                        "HTTP_PROXY": f"http://127.0.0.1:{servers[0].server_port}", "no_proxy": "", "NO_PROXY": ""}):
                with client.request(f"http://127.0.0.1:{servers[1].server_port}", "/v1/models",
                                    payload={}, token="synthetic-proxy-test") as response:
                    self.assertEqual(response.read(), b"{}")
            self.assertEqual(received, {"proxy": 0, "direct": 1})
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(2)

    def test_unsafe_endpoints_fail_before_any_network_request(self):
        for base in ("http://example.invalid", "file:///tmp/test", "http://localhost.evil.invalid",
                     "https://user:password@example.invalid", "https://example.invalid?token=private"):
            with self.subTest(base=base), patch("client.urllib.request.build_opener") as opener:
                with self.assertRaises(ValueError):
                    client.request(base, "/v1/models", token="synthetic")
                opener.assert_not_called()

    def test_redirects_never_forward_credentials(self):
        received = []

        class Sink(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(int(self.path.strip("/")))
                self.send_header("Location", f"http://localhost:{sink.server_port}/target")
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()

        sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
        source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
                   for s in (sink, source)]
        for thread in threads:
            thread.start()
        try:
            for code in (301, 302, 303, 307, 308):
                with self.subTest(code=code):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        client.request(f"http://127.0.0.1:{source.server_port}", f"/{code}",
                                       payload={}, token="synthetic-review-token")
                    self.assertEqual(caught.exception.code, code)
                    caught.exception.close()
            self.assertEqual(received, [])
        finally:
            for server in (source, sink):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

    def test_explicit_token_context_overrides_environment(self):
        with patch.dict(os.environ, {"KUBE_CONTEXT": "other", "LLM_API_TOKEN": ""}), patch(
                "client.subprocess.check_output", return_value=b'{"data":{"api-token":"dGVzdA=="}}') as run:
            self.assertEqual(client.api_token(context="safeboard"), "test")
        self.assertEqual(run.call_args.args[0][2], "safeboard")


class SecretTests(unittest.TestCase):
    def test_default_context_ignores_unrelated_environment_on_create(self):
        with patch.dict(os.environ, {"KUBE_CONTEXT": "other", "HF_TOKEN": "synthetic-hf-token"}), patch(
                "create_secret.subprocess.run", return_value=SimpleNamespace(stdout="")) as run:
            create_secret.create_secret()
        self.assertEqual(len(run.call_args_list), 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][2], "safeboard")
            self.assertNotIn("synthetic-hf-token", " ".join(call.args[0]))
        self.assertEqual(json.loads(run.call_args_list[1].kwargs["input"])["kind"], "Secret")

    def test_explicit_context_retains_existing_secret(self):
        with patch.dict(os.environ, {"KUBE_CONTEXT": "other"}), patch(
                "create_secret.subprocess.run", return_value=SimpleNamespace(stdout="secret/vllm-tokens")) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                create_secret.create_secret("selected")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][2], "selected")


class EvidenceTests(unittest.TestCase):
    def test_manifest_hash_survives_json_formatting_and_matches_report(self):
        raw = (ROOT / "model.lock.json").read_text(encoding="utf-8")
        value = json.loads(raw)
        reformatted = json.dumps(value, indent=4).replace("\n", "\r\n")
        digest = canonical_json_sha256(value)
        self.assertEqual(digest, canonical_json_sha256(json.loads(reformatted)))
        report = json.loads((ROOT / "results" / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(report["source_manifest_hash_format"], "sha256-canonical-json-v1")
        self.assertEqual(report["source_manifest_sha256"], digest)

    def test_config_text_hash_survives_checkout_line_endings(self):
        self.assertEqual(text_sha256("port: 8000\r\ndtype: bfloat16\r\n"),
                         text_sha256("port: 8000\ndtype: bfloat16\n"))

    def test_snapshot_reads_actual_pod_configuration_without_secrets(self):
        pod = {"metadata": {"name": "vllm-test", "uid": "pod-uid"},
               "spec": {"containers": [{"name": "inference", "image": "test:1", "resources": {},
                                        "envFrom": [{"configMapRef": {"name": "env-actual"}}]}],
                        "volumes": [{"name": "settings", "configMap": {"name": "settings-actual"}}]},
               "status": {"conditions": [{"type": "Ready", "status": "True"}],
                          "containerStatuses": [{"name": "inference", "imageID": "sha256:actual", "restartCount": 0}]}}
        settings = {"serve.yaml": "port: 8000\n", "model.lock.json": '{"revision":"pinned"}'}
        objects = [{"items": [pod]}, {"data": settings}, {"data": {"OMP_NUM_THREADS": "8"}}]
        with patch("evidence.subprocess.check_output", side_effect=[json.dumps(x).encode() for x in objects]) as run:
            snapshot = server_snapshot("selected")
        self.assertEqual(snapshot["image_id"], "sha256:actual")
        self.assertEqual(snapshot["serve_config_sha256"], text_sha256(settings["serve.yaml"]))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all(command[2] == "selected" for command in commands))
        self.assertTrue(any("settings-actual" in command for command in commands))
        self.assertTrue(all("secret" not in command for command in commands))

    def test_endpoint_binding_rejects_audit_from_another_request(self):
        request_id = "a" * 32
        response = Mock()
        response.headers = {"X-Request-Id": request_id}
        response.read.return_value = b'{"data":[]}'
        manager = contextlib.nullcontext(response)
        wrong = json.dumps({"event": "model_request", "request_id": "b" * 32, "status": 200, "operation": "models"})
        with patch("evidence.request", return_value=manager), patch(
                "evidence.subprocess.check_output", return_value=wrong), patch("evidence.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "could not be linked"):
                bind_endpoint("http://example.invalid", "synthetic", {"context": "test", "pod_name": "test"})

    def test_endpoint_binding_accepts_matching_audit(self):
        request_id = "a" * 32
        response = Mock()
        response.headers = {"X-Request-Id": request_id}
        response.read.return_value = b'{"data":[]}'
        audit = json.dumps({"event": "model_request", "request_id": request_id, "status": 200, "operation": "models"})
        with patch("evidence.request", return_value=contextlib.nullcontext(response)), patch(
                "evidence.subprocess.check_output", return_value=audit):
            self.assertEqual(bind_endpoint("http://example.invalid", "synthetic",
                                          {"context": "test", "pod_name": "test"}), request_id)


class BenchmarkTests(unittest.TestCase):
    def test_streaming_decode_metric_excludes_wait_for_first_fragment(self):
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
                            b'data: {"usage":{"prompt_tokens":8,"completion_tokens":2},"choices":[]}\n\n'
                            b'data: [DONE]\n\n')
        with patch("benchmark.request", return_value=stream) as request, patch(
                "benchmark.time.perf_counter", side_effect=[10.0, 11.0, 11.5]):
            row = benchmark.one_request("http://example.invalid", "synthetic", "synthetic", 2,
                                        cache_salt="unique-cache-salt")
        self.assertEqual(row["ttft_s"], 1.0)
        self.assertEqual(row["tpot_s"], 0.5)
        self.assertEqual(row["decode_tokens_per_s"], 2.0)
        self.assertEqual(request.call_args.kwargs["payload"]["cache_salt"], "unique-cache-salt")

    def test_incomplete_stream_is_rejected(self):
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        with patch("benchmark.request", return_value=stream):
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                benchmark.one_request("http://example.invalid", "synthetic", "synthetic", 128)

    def test_mismatched_live_configuration_fails_before_generation(self):
        with patch("benchmark.api_token", return_value="synthetic"), patch(
                "benchmark.server_snapshot", return_value={"serve_config_sha256": "different"}), patch(
                "benchmark.one_request") as generate:
            with self.assertRaisesRegex(RuntimeError, "differ"):
                benchmark.benchmark("http://example.invalid", 4, 2, "test")
        generate.assert_not_called()

    def test_pod_restart_during_benchmark_discards_results(self):
        digest = text_sha256((ROOT / "k8s" / "serve.yaml").read_text(encoding="utf-8"))
        before = {"serve_config_sha256": digest, "pod_uid": "before"}
        after = {**before, "pod_uid": "after"}
        measurement = {"ttft_s": 0.01, "latency_s": 0.02, "prompt_tokens": 8, "output_tokens": 2}
        with patch("benchmark.api_token", return_value="synthetic"), patch(
                "benchmark.server_snapshot", side_effect=[before, after]), patch(
                "benchmark.bind_endpoint", return_value="a" * 32), patch(
                "benchmark.one_request", return_value=measurement), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                benchmark.benchmark("http://example.invalid", 4, 2, "test")

    def test_remote_benchmark_does_not_claim_server_configuration(self):
        measurement = {"ttft_s": 0.01, "latency_s": 0.02, "prompt_tokens": 8, "output_tokens": 2}
        with patch("benchmark.api_token", return_value="synthetic"), patch(
                "benchmark.server_snapshot") as snapshot, patch(
                "benchmark.one_request", return_value=measurement), contextlib.redirect_stdout(io.StringIO()):
            report = benchmark.benchmark("http://example.invalid", 4, 2)
        snapshot.assert_not_called()
        self.assertNotIn("server_config_sha256", report)
        self.assertEqual(report["kubernetes_evidence"]["status"], "not_collected")


if __name__ == "__main__":
    unittest.main()
