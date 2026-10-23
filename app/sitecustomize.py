"""Sanitize direct Python warnings and uncaught/multiprocessing tracebacks too."""
import json
import sys
import threading
import traceback
import warnings
from pathlib import Path


def safe_exception(*args, **kwargs):
    exc = args[0] if args and isinstance(args[0], BaseException) else (args[1] if len(args) > 1 else None)
    tb = kwargs.get("tb", args[2] if len(args) > 2 else getattr(exc, "__traceback__", None))
    frames = traceback.extract_tb(tb) if tb else []
    record = {"event": "unhandled_error", "error_type": type(exc).__name__}
    if frames:
        record["error_location"] = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}"
    print(json.dumps(record), file=kwargs.get("file") or sys.stderr, flush=True)


def safe_print_exc(limit=None, file=None, chain=True):
    safe_exception(*sys.exc_info(), limit=limit, file=file, chain=chain)


traceback.print_exception = safe_exception
traceback.print_exc = safe_print_exc
sys.excepthook = safe_exception
threading.excepthook = lambda args: safe_exception(args.exc_type, args.exc_value, args.exc_traceback)
sys.unraisablehook = lambda args: safe_exception(args.exc_type, args.exc_value, args.exc_traceback)
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    json.dumps({"event": "python_warning", "category": category.__name__,
                "location": f"{Path(filename).name}:{lineno}"}), file=file or sys.stderr, flush=True)
