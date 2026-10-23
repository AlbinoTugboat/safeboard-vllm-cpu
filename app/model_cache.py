"""Local model identity and integrity checks; never performs network access."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def read_lock(path):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", lock["model_id"]):
        raise ValueError("Invalid model identifier")
    if not re.fullmatch(r"[a-f0-9]{40}", lock["revision"]):
        raise ValueError("A full immutable model revision is required")
    if not isinstance(lock.get("files"), dict) or not lock["files"]:
        raise ValueError("An explicit nonempty file manifest is required")
    for name, spec in lock["files"].items():
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or not isinstance(spec, dict)
                or type(spec.get("size")) is not int or spec["size"] < 0
                or not isinstance(spec.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])):
            raise ValueError("Invalid file manifest")
    return lock


def cache_path(root, lock):
    return Path(root) / (lock["model_id"].replace("/", "--") + "@" + lock["revision"])


def plain_directory(directory):
    """Reject symlink/junction directories, including ancestors."""
    directory = Path(directory).absolute()
    try:
        return all(stat.S_ISDIR(p.lstat().st_mode) and not p.is_symlink()
                   and not (hasattr(p, "is_junction") and p.is_junction())
                   for p in (directory, *directory.parents))
    except OSError:
        return False


def cache_layout_valid(directory, lock, *, complete=True):
    """Only manifest files and inert Hub bookkeeping may exist in a cache."""
    directory = Path(directory)
    if not lock.get("files") or not plain_directory(directory):
        return False
    try:
        entries = {p.name: p for p in directory.iterdir()}
        if set(entries) - set(lock["files"]) - {".cache"}:
            return False
        if complete and not set(lock["files"]) <= set(entries):
            return False
        for name in set(entries) & set(lock["files"]):
            if not stat.S_ISREG(entries[name].lstat().st_mode):
                return False
        metadata = entries.get(".cache")
        if metadata is not None:
            if not plain_directory(metadata) or set(p.name for p in metadata.iterdir()) - {"huggingface"}:
                return False
            if (metadata / "huggingface").exists() and not plain_directory(metadata / "huggingface"):
                return False
            # Hub metadata is never copied to the engine's private model directory.
            for parent, dirs, files in os.walk(metadata, followlinks=False):
                for name in dirs:
                    if not plain_directory(Path(parent) / name):
                        return False
                for name in files:
                    if not stat.S_ISREG((Path(parent) / name).lstat().st_mode):
                        return False
        return True
    except OSError:
        return False


def _open_regular(path):
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Model file must be regular")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def verify_files(directory, lock):
    directory = Path(directory)
    if not cache_layout_valid(directory, lock):
        return False
    for name, spec in lock["files"].items():
        try:
            with _open_regular(directory / name) as stream:
                if os.fstat(stream.fileno()).st_size != spec["size"]:
                    return False
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except (OSError, ValueError):
            return False
        if digest != spec["sha256"]:
            return False
    return True


def stage_verified_model(source, lock, private_root):
    """Copy and hash the exact engine inputs into an inference-only emptyDir.

    The engine never opens the shared PVC. A later PVC change cannot alter this
    snapshot; hashes validate the bytes actually copied, not an earlier pathname.
    """
    source, private_root = Path(source), Path(private_root)
    if not cache_layout_valid(source, lock):
        raise RuntimeError("Model cache layout is invalid")
    private_root.mkdir(parents=True, exist_ok=True)
    if not plain_directory(private_root):
        raise RuntimeError("Private model directory is invalid")
    target = private_root / "model"
    if target.exists() or target.is_symlink():
        if not plain_directory(target):
            raise RuntimeError("Private model directory is invalid")
        # This dedicated emptyDir contains only our previous flat snapshot.
        previous = list(target.iterdir())
        if any(not stat.S_ISREG(p.lstat().st_mode) for p in previous):
            raise RuntimeError("Unexpected private model entry")
        target.chmod(0o700)
        for path in previous:
            path.chmod(0o600)
            path.unlink()
        target.rmdir()
    target.mkdir(mode=0o700)
    for name, spec in lock["files"].items():
        digest, size = hashlib.sha256(), 0
        with _open_regular(source / name) as src, (target / name).open("xb") as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                size += len(chunk)
                if size > spec["size"]:
                    raise RuntimeError("Model snapshot failed integrity verification")
                digest.update(chunk)
                dst.write(chunk)
        if size != spec["size"] or digest.hexdigest() != spec["sha256"]:
            raise RuntimeError("Model snapshot failed integrity verification")
        (target / name).chmod(0o400)
    target.chmod(0o500)
    return target
