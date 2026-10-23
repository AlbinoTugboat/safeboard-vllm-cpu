"""Read-only evidence about the Pod serving a benchmark; never reads Secrets."""
import hashlib
import json
import re
import subprocess
import time

from client import request


def canonical_json_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def text_sha256(text):
    """Hash UTF-8 text with LF line endings, independent of the checkout platform."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def kube_command(context):
    return ["kubectl", "--context", context, "-n", "safeboard-llm"]


def server_snapshot(context):
    kube = kube_command(context)

    def get(*args):
        return json.loads(subprocess.check_output(kube + ["get", *args, "-o", "json"]))

    pods = get("pods", "-l", "app=safeboard-vllm")["items"]
    ready = [pod for pod in pods if not pod["metadata"].get("deletionTimestamp") and any(
        condition["type"] == "Ready" and condition["status"] == "True"
        for condition in pod.get("status", {}).get("conditions", []))]
    if len(ready) != 1:
        raise RuntimeError("Benchmark evidence requires exactly one ready inference Pod")
    pod = ready[0]
    container = next(c for c in pod["spec"]["containers"] if c["name"] == "inference")
    status = next(c for c in pod["status"]["containerStatuses"] if c["name"] == "inference")
    settings_name = next(v["configMap"]["name"] for v in pod["spec"]["volumes"] if v["name"] == "settings")
    settings = get("configmap", settings_name)["data"]
    env_maps = {ref["configMapRef"]["name"]: get("configmap", ref["configMapRef"]["name"])["data"]
                for ref in container.get("envFrom", []) if "configMapRef" in ref}
    return {"context": context, "namespace": "safeboard-llm",
            "pod_name": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "image": container["image"], "image_id": status["imageID"],
            "restart_count": status["restartCount"], "resources": container["resources"],
            "serve_config_sha256": text_sha256(settings["serve.yaml"]),
            "model_manifest_sha256": canonical_json_sha256(json.loads(settings["model.lock.json"])),
            "config_maps_sha256": canonical_json_sha256({"settings": settings, "env": env_maps}),
            "container_spec_sha256": canonical_json_sha256(container)}


def bind_endpoint(base, token, snapshot):
    """Match an endpoint-generated request ID to an audit record in the selected Pod."""
    with request(base, "/v1/models", token=token) as response:
        request_id = response.headers.get("X-Request-Id", "")
        json.load(response)
    if not re.fullmatch(r"[a-f0-9]{32}", request_id):
        raise RuntimeError("Endpoint did not return a verifiable request ID")
    for attempt in range(5):
        logs = subprocess.check_output(kube_command(snapshot["context"]) + [
            "logs", snapshot["pod_name"], "-c", "inference", "--since=1m", "--tail=2000"], text=True)
        for line in logs.splitlines():
            try:
                item = json.loads(line[line.index("{"):])
            except (ValueError, json.JSONDecodeError):
                continue
            if (item.get("event") == "model_request" and item.get("request_id") == request_id
                    and item.get("status") == 200 and item.get("operation") == "models"):
                return request_id
        if attempt < 4:
            time.sleep(0.1)
    raise RuntimeError("Endpoint could not be linked to the selected inference Pod")
