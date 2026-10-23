"""Measure real streaming latency, first content arrival and aggregate output throughput."""
import argparse
import concurrent.futures
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from client import api_token, request
from evidence import bind_endpoint, server_snapshot, text_sha256


def one_request(base, token, prompt, output_tokens, *, cache_salt=None):
    payload = {"model": "safeboard-qwen", "messages": [{"role": "user", "content": prompt}],
               "stream": True, "stream_options": {"include_usage": True},
               "temperature": 0, "max_tokens": output_tokens, "ignore_eos": True}
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    started = time.perf_counter()
    first = None
    usage = None
    with request(base, "/v1/chat/completions", token=token, payload=payload, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            item = json.loads(body)
            if item.get("error"):
                raise RuntimeError("Streaming API returned an error")
            if item.get("usage"):
                usage = item["usage"]
            if first is None and any(choice.get("delta", {}).get("content") for choice in item.get("choices", [])):
                first = time.perf_counter()
    finished = time.perf_counter()
    if first is None or not usage or usage["completion_tokens"] != output_tokens:
        raise RuntimeError("Incomplete response or missing token usage")
    # Client-observed TPOT includes final SSE/transport overhead. It is an estimate
    # when a content fragment contains more than one token.
    tpot = (finished - first) / (output_tokens - 1) if output_tokens > 1 else None
    return {"ttft_s": first - started, "latency_s": finished - started,
            "tpot_s": tpot, "decode_tokens_per_s": 1 / tpot if tpot and tpot > 0 else None,
            "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]}


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def benchmark(base, samples, output_tokens, kube_context=None):
    local_config = (Path(__file__).resolve().parents[1] / "k8s" / "serve.yaml").read_text(encoding="utf-8")
    local_config_hash = text_sha256(local_config)
    token = api_token(context=kube_context)
    evidence = {"status": "not_collected", "reason": "No --kube-context provided"}
    if kube_context is not None:
        before = server_snapshot(kube_context)
        if before["serve_config_sha256"] != local_config_hash:
            raise RuntimeError("Pod settings differ from the local serve.yaml; apply the intended configuration first")
        binding_before = bind_endpoint(base, token, before)
    run_id = uuid.uuid4().hex
    rows = []
    raw_cases = []
    for kind in ("short", "long"):
        text = "Name a color." if kind == "short" else ("The local service processes synthetic benchmark data. " * 210)
        for concurrency in (1, 4):
            # Unique IDs precede the long common text, avoiding shared-prefix cache hits.
            prompts = [f"Request {run_id}-{kind}-{concurrency}-{i}. {text}" for i in range(samples)]
            one_request(base, token, f"Warmup {uuid.uuid4().hex}. {text}", output_tokens)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                measurements = list(pool.map(lambda p: one_request(base, token, p, output_tokens), prompts))
            elapsed = time.perf_counter() - started
            ttfts = [m["ttft_s"] for m in measurements]
            latencies = [m["latency_s"] for m in measurements]
            row = {"prompt": kind, "concurrency": concurrency, "requests": samples,
                   "prompt_tokens_mean": round(statistics.mean(m["prompt_tokens"] for m in measurements), 1),
                   "output_tokens_each": output_tokens,
                   "ttft_mean_s": round(statistics.mean(ttfts), 4), "ttft_p95_s": round(percentile95(ttfts), 4),
                   "latency_mean_s": round(statistics.mean(latencies), 4),
                   "latency_p95_s": round(percentile95(latencies), 4),
                   "output_tokens_per_s": round(sum(m["output_tokens"] for m in measurements) / elapsed, 3),
                   "requests_per_s": round(samples / elapsed, 4), "wall_s": round(elapsed, 3)}
            rows.append(row)
            raw_cases.append({"prompt": kind, "concurrency": concurrency, "measurements": measurements})
            print(json.dumps(row), flush=True)
    if kube_context is not None:
        after = server_snapshot(kube_context)
        if after != before:
            raise RuntimeError("Pod or its configuration changed during the benchmark; results discarded")
        binding_after = bind_endpoint(base, token, after)
        evidence = {"status": "pod_verified", "before": before, "after": after,
                    "binding_request_ids": [binding_before, binding_after]}
    return {"run_id": run_id, "local_config_sha256": local_config_hash,
            "config_hash_format": "sha256-utf8-lf-v1", "kubernetes_evidence": evidence,
            "notes": "Warmups excluded. Unique prefixes. TTFT is first non-empty SSE content arrival. p95 uses nearest rank.",
            "cases": rows, "raw_cases": raw_cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--kube-context", help="Verify the endpoint against the ready safeboard-llm Pod in this context")
    args = parser.parse_args()
    if args.requests < 4 or not 1 <= args.output_tokens <= 256:
        parser.error("At least four requests and 1..256 output tokens are required")
    result = benchmark(args.url, args.requests, args.output_tokens, args.kube_context)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
