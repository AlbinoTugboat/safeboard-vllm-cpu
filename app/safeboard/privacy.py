"""Only explicitly approved metadata is serialized; message and traceback are omitted."""
import datetime
import json
import logging
import traceback
from pathlib import Path


class MetadataFormatter(logging.Formatter):
    def format(self, record):
        result = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": "runtime_log",
        }
        if record.exc_info and record.exc_info[1] is not None:
            result["error_type"] = type(record.exc_info[1]).__name__
            frames = traceback.extract_tb(record.exc_info[2])
            if frames:
                result["error_location"] = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}"
        if record.name == "safeboard.audit":
            data = getattr(record, "audit", {})
            result["event"] = "model_request"
            for key in ("request_id", "operation", "status", "duration_ms", "error_type"):
                if key in data:
                    result[key] = data[key]
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))
