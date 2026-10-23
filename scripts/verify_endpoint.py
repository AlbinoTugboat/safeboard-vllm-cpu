"""Check authentication, inference, invalid input and absence of content in server logs."""
import argparse
import json
import subprocess
import time
import urllib.error
import uuid
from pathlib import Path

from client import api_token, request


def verify(base, context="safeboard"):
    token = api_token(context=context)
    sentinel = "PRIVATE_TEST_" + uuid.uuid4().hex
    report = {}
    for path in ("/v1/models", "/invocations", "/tokenize"):
        try:
            request(base, path, payload={} if path != "/v1/models" else None)
            raise AssertionError("Unauthenticated endpoint was accessible")
        except urllib.error.HTTPError as error:
            try:
                assert error.code == 401, (path, error.code)
                error.read(4096)
            finally:
                error.close()
    report["authentication"] = "passed"
    with request(base, "/v1/models", token=token) as response:
        assert json.load(response)["data"][0]["id"] == "safeboard-qwen"
    payload = {"model": "safeboard-qwen", "messages": [{"role": "user", "content":
               "Repeat this exact identifier and nothing else: " + sentinel}],
               "max_tokens": 48, "temperature": 0}
    with request(base, "/v1/chat/completions", payload=payload, token=token) as response:
        request_id = response.headers.get("X-Request-Id")
        result = json.load(response)
    answer = result["choices"][0]["message"]["content"]
    assert answer and request_id
    report["inference"] = "passed"
    # The unknown model name and validation body must also stay out of logs.
    for invalid in ({**payload, "model": sentinel}, {**payload, "max_tokens": -1}):
        try:
            request(base, "/v1/chat/completions", payload=invalid, token=token)
            raise AssertionError("Invalid request was accepted")
        except urllib.error.HTTPError as error:
            try:
                assert error.code in (400, 404, 422)
                error.read(4096)
            finally:
                error.close()
    time.sleep(0.25)
    logs = subprocess.check_output([
        "kubectl", "--context", context, "-n", "safeboard-llm", "logs", "deployment/vllm",
        "-c", "inference",
    ], text=True)
    assert sentinel not in logs and token not in logs
    assert answer not in logs
    audit = []
    for line in logs.splitlines():
        try:
            item = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
        if item.get("event") == "model_request":
            audit.append(item)
    assert any(item.get("request_id") == request_id and item["status"] == 200 for item in audit)
    assert any(item.get("status") in (400, 404, 422) for item in audit)
    report["private_content_absent_from_logs"] = "passed"
    report["success_and_error_requests_audited"] = "passed"
    report["output_tokens"] = result["usage"]["completion_tokens"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--context", default="safeboard")
    parser.add_argument("--output", default="results/verification.json")
    args = parser.parse_args()
    report = verify(args.url, args.context)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
