"""Standard-library client. The token stays in memory and is never printed."""
import argparse
import base64
import json
import ipaddress
import os
import subprocess
import urllib.error
import urllib.request
import urllib.parse


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """API calls must never forward credentials through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_token(*, context=None):
    if token := os.getenv("LLM_API_TOKEN"):
        return token
    raw = subprocess.check_output([
        "kubectl", "--context", context or os.getenv("KUBE_CONTEXT", "safeboard"), "-n", "safeboard-llm",
        "get", "secret", "vllm-tokens", "-o", "json",
    ])
    return base64.b64decode(json.loads(raw)["data"]["api-token"]).decode()


def request(base, path, *, payload=None, token=None, timeout=180):
    parsed = urllib.parse.urlsplit(base)
    host = parsed.hostname
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (parsed.scheme not in ("http", "https") or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or any(c.isspace() for c in base)
            or (parsed.scheme == "http" and not loopback)):
        raise ValueError("Use a direct loopback HTTP endpoint or an HTTPS endpoint")
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(urllib.request.Request(base + path, data=data, headers=headers), timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Назови столицу Франции одним словом.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with request(args.url, "/v1/chat/completions", token=api_token(), payload={
        "model": "safeboard-qwen", "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64, "temperature": 0,
    }) as response:
        print(json.load(response)["choices"][0]["message"]["content"])
