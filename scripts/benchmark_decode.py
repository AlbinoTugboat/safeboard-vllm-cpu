"""Compare CPU settings on identical inputs with longer, fixed-length responses."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import statistics
import time
import uuid
from pathlib import Path

from benchmark import one_request, percentile95
from client import api_token
from evidence import bind_endpoint, server_snapshot, text_sha256


def run(base, samples=4, output_tokens=128, kinds=("short",), concurrency=(1,), context="safeboard"):
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    token = api_token(context=context)
    before = server_snapshot(context)
    local_hash = text_sha256((Path(__file__).resolve().parents[1] / "k8s/serve.yaml").read_text(encoding="utf-8"))
    if before["serve_config_sha256"] != local_hash:
        raise RuntimeError("Pod settings differ from the local serve.yaml")
    binding_before = bind_endpoint(base, token, before)
    rows, raw_cases = [], []
    for kind in kinds:
        text = "List simple facts about the ocean." if kind == "short" else (
            "The local service processes synthetic benchmark data. " * 210)
        for parallelism in concurrency:
            # Identical tokenized inputs in every variant; separate salts prevent
            # prefix-cache reuse without adding random text to model inputs.
            prompts = [f"Case {i:04d}. {text}" for i in range(samples)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
                list(pool.map(lambda i: one_request(base, token, f"Warmup {i}. " + text,
                              min(output_tokens, 32), cache_salt=uuid.uuid4().hex), range(parallelism)))
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
                measurements = list(pool.map(lambda prompt: one_request(
                    base, token, prompt, output_tokens, cache_salt=uuid.uuid4().hex), prompts))
            elapsed = time.perf_counter() - started
            row = {"prompt": kind, "concurrency": parallelism, "requests": samples,
                   "prompt_tokens_mean": statistics.mean(m["prompt_tokens"] for m in measurements),
                   "output_tokens_each": output_tokens,
                   "ttft_mean_s": statistics.mean(m["ttft_s"] for m in measurements),
                   "ttft_p95_s": percentile95([m["ttft_s"] for m in measurements]),
                   "latency_mean_s": statistics.mean(m["latency_s"] for m in measurements),
                   "latency_p95_s": percentile95([m["latency_s"] for m in measurements]),
                   "tpot_mean_s": statistics.mean(m["tpot_s"] for m in measurements),
                   "decode_tokens_per_s_mean": statistics.mean(m["decode_tokens_per_s"] for m in measurements),
                   "output_tokens_per_s": samples * output_tokens / elapsed, "wall_s": elapsed}
            rows.append(row)
            raw_cases.append({"prompt": kind, "concurrency": parallelism,
                              "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
                              "measurements": measurements})
            print(json.dumps(row), flush=True)
    after = server_snapshot(context)
    if before != after:
        raise RuntimeError("Pod or configuration changed during measurement")
    binding_after = bind_endpoint(base, token, after)
    return {"started_at_utc": started_at, "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "local_config_sha256": local_hash, "config_hash_format": "sha256-utf8-lf-v1",
            "kubernetes_evidence": {"status": "pod_verified", "before": before, "after": after,
                                    "binding_request_ids": [binding_before, binding_after]},
            "notes": "Fixed prompts; unique per-request cache_salt; warmups excluded. Client TPOT=(latency-TTFT)/(output_tokens-1). Decode rate is per stream, not aggregate throughput.",
            "cases": rows, "raw_cases": raw_cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prompts", nargs="+", choices=("short", "long"), default=["short"])
    parser.add_argument("--concurrency", nargs="+", type=int, choices=(1, 4), default=[1])
    parser.add_argument("--kube-context", default="safeboard")
    parser.add_argument("--output", default="results/performance/decode.json")
    args = parser.parse_args()
    if args.requests < 2 or not 2 <= args.output_tokens <= 512:
        parser.error("At least two requests and 2..512 output tokens are required")
    report = run(args.url, args.requests, args.output_tokens, args.prompts, args.concurrency, args.kube_context)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
