"""Small deterministic smoke test; does not establish general model quality."""
import argparse
import json
import re
from pathlib import Path

from client import api_token, request
from evidence import bind_endpoint, server_snapshot

CASES = [
    ("addition", "What is 17 + 25? Answer only with the number.", "number", "42"),
    ("multiplication", "What is 12 times 13? Answer only with the number.", "number", "156"),
    ("subtraction", "What is 100 minus 37? Answer only with the number.", "number", "63"),
    ("square_root", "What is the square root of 144? Answer only with the number.", "number", "12"),
    ("precedence", "Calculate 3 * 7 + 2. Answer only with the number.", "number", "23"),
    ("word_problem", "I have 4 apples, buy 7 and eat 3. How many remain? Answer only with the number.", "number", "8"),
    ("france", "What is the capital of France? Answer only with one word.", "word", "Paris"),
    ("japan", "What is the capital of Japan? Answer only with one word.", "word", "Tokyo"),
    ("russian", "Как называется столица Франции? Ответь одним словом.", "word", "Париж"),
    ("extraction", "Extract the identifier from: Ticket AX-731 is open. Output only the identifier.", "word", "AX-731"),
    ("json", 'Return only this JSON object with no extra text: {"ok":true,"count":3}', "json", {"ok": True, "count": 3}),
    ("sorting", "Sort these integers ascending: 9, 2, 5. Return only a JSON array.", "json", [2, 5, 9]),
]


def grade(answer, kind, expected):
    if kind == "json":
        try:
            # Compare JSON encodings to distinguish true from 1 and false from 0.
            return json.dumps(json.loads(answer), sort_keys=True) == json.dumps(expected, sort_keys=True)
        except (ValueError, TypeError):
            return False
    normalized = answer.strip().strip(".!\"'").strip()
    if kind == "number":
        return bool(re.fullmatch(r"[+-]?\d+", normalized)) and normalized == expected
    return normalized.casefold() == expected.casefold()


def run(base, label, context="safeboard"):
    token = api_token(context=context)
    before = server_snapshot(context)
    binding_before = bind_endpoint(base, token, before)
    results = []
    for case_id, prompt, kind, expected in CASES:
        payload = {"model": "safeboard-qwen", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 48, "temperature": 0, "seed": 42}
        with request(base, "/v1/chat/completions", token=token, payload=payload) as response:
            answer = json.load(response)["choices"][0]["message"]["content"]
        results.append({"case": case_id, "expected": expected, "answer": answer,
                        "passed": grade(answer, kind, expected)})
    after = server_snapshot(context)
    if before != after:
        raise RuntimeError("Pod or configuration changed during quality check")
    binding_after = bind_endpoint(base, token, after)
    return {"label": label, "passed": sum(row["passed"] for row in results), "total": len(results),
            "scope": "12 synthetic factual, arithmetic and formatting checks; not a general accuracy benchmark",
            "kubernetes_evidence": {"status": "pod_verified", "before": before, "after": after,
                                    "binding_request_ids": [binding_before, binding_after]}, "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kube-context", default="safeboard")
    args = parser.parse_args()
    report = run(args.url, args.label, args.kube_context)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"label": args.label, "passed": report["passed"], "total": report["total"]}))
