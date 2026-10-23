"""Populate a PVC once, then validate and use it entirely offline."""
import fcntl
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from model_cache import cache_layout_valid, cache_path, plain_directory, read_lock, verify_files


def event(name, **fields):
    print(json.dumps({"event": name, **fields}), flush=True)


def prepare():
    lock = read_lock(os.getenv("MODEL_LOCK_PATH", "/config/model.lock.json"))
    root = Path(os.getenv("MODEL_STORAGE_DIR", "/models"))
    root.mkdir(parents=True, exist_ok=True)
    if not plain_directory(root):
        raise RuntimeError("Model storage directory is invalid")
    target = cache_path(root, lock)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(root / ".download.lock", flags, 0o600), "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if verify_files(target, lock):
            event("model_cache_hit", revision=lock["revision"], integrity="sha256_verified")
            return
        if target.exists() or target.is_symlink():
            # Fail before importing Hub or downloading anything into .partial.
            raise RuntimeError("Existing cache failed integrity verification")
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            raise RuntimeError("Offline cache is incomplete")
        partial = root / (target.name + ".partial")
        if ((partial.exists() or partial.is_symlink())
                and not cache_layout_valid(partial, lock, complete=False)):
            raise RuntimeError("Partial cache layout is invalid")
        # Import only after the complete local cache has been checked.
        from huggingface_hub import snapshot_download

        token_file = Path(os.getenv("HF_TOKEN_FILE", "/secrets/hf-token"))
        token = token_file.read_text().strip() if token_file.exists() else ""
        event("model_download_started", revision=lock["revision"])
        snapshot_download(
            repo_id=lock["model_id"], revision=lock["revision"], local_dir=partial,
            allow_patterns=list(lock["files"]), token=token or False, max_workers=2,
        )
        if not verify_files(partial, lock):
            raise RuntimeError("Downloaded files failed integrity verification")
        partial.rename(target)
        event("model_download_complete", revision=lock["revision"], integrity="sha256_verified")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    try:
        prepare()
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        frame = frames[-1] if frames else None
        code = "download_error"
        for fragment, candidate in (("Existing cache", "existing_cache_invalid"),
                                    ("client has been closed", "http_client_closed"),
                                    ("etag is empty", "metadata_missing"),
                                    ("integrity verification", "integrity_failure"),
                                    ("Offline cache", "offline_cache_incomplete")):
            if fragment in str(exc):
                code = candidate
                break
        event("model_download_failed", error_type=type(exc).__name__, reason=code,
              location=(f"{Path(frame.filename).name}:{frame.lineno}" if frame else "unknown"))
        sys.exit(1)
