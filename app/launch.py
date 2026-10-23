"""Validate local prerequisites and start the CPU-only vLLM server."""
import json
import os
import sys
from pathlib import Path

from model_cache import cache_path, read_lock, stage_verified_model


def main():
    lock = read_lock(os.getenv("MODEL_LOCK_PATH", "/config/model.lock.json"))
    source = cache_path(os.getenv("MODEL_STORAGE_DIR", "/models"), lock)
    if len(Path(os.getenv("API_TOKEN_FILE", "/secrets/api-token")).read_bytes().strip()) < 32:
        raise RuntimeError("Missing API token")
    model = stage_verified_model(source, lock, os.getenv("VERIFIED_MODEL_DIR", "/verified-model"))
    print(json.dumps({"event": "model_snapshot_prepared", "revision": lock["revision"],
                      "integrity": "sha256_verified"}), flush=True)
    # These constraints are independent of the editable inference ConfigMap.
    os.environ.update(VLLM_TARGET_DEVICE="cpu", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                      VLLM_DEBUG_LOG_API_SERVER_RESPONSE="0")
    args = ["vllm", "serve", str(model), "--config", "/config/serve.yaml",
            "--middleware", "safeboard.middleware.RequestMiddleware",
            "--no-enable-log-requests", "--no-enable-log-outputs",
            "--disable-uvicorn-access-log", "--disable-fastapi-docs"]
    os.execvp(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "startup_failed", "error_type": type(exc).__name__}), flush=True)
        sys.exit(1)
