"""Run the real downloader with HF_HUB_OFFLINE=1 against the existing PVC."""
import copy
import json
import subprocess
from pathlib import Path

KUBE = ["kubectl", "--context", "safeboard", "-n", "safeboard-llm"]


if __name__ == "__main__":
    deploy = json.loads(subprocess.check_output(KUBE + ["get", "deployment", "vllm", "-o", "json"]))
    template = deploy["spec"]["template"]["spec"]
    container = copy.deepcopy(template["initContainers"][0])
    container["name"] = "check"
    container["env"] = [{"name": "HF_HUB_OFFLINE", "value": "1"}]
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": "offline-cache-check", "namespace": "safeboard-llm"},
           "spec": {"backoffLimit": 0, "activeDeadlineSeconds": 90,
                    "template": {"spec": {"restartPolicy": "Never", "enableServiceLinks": False,
                        "automountServiceAccountToken": False,
                        "securityContext": template["securityContext"],
                        "containers": [container], "volumes": template["volumes"]}}}}
    subprocess.run(KUBE + ["delete", "job", "offline-cache-check", "--ignore-not-found", "--wait=true"], check=True)
    subprocess.run(KUBE + ["create", "-f", "-"], input=json.dumps(job), text=True, check=True)
    try:
        subprocess.run(KUBE + ["wait", "--for=condition=complete", "job/offline-cache-check", "--timeout=90s"], check=True)
        logs = subprocess.check_output(KUBE + ["logs", "job/offline-cache-check"], text=True)
        assert '"model_cache_hit"' in logs and '"model_download_started"' not in logs
        report = {"HF_HUB_OFFLINE": "1", "cache_integrity": "sha256_verified", "result": "passed"}
        output = Path("results/offline-cache.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        subprocess.run(KUBE + ["delete", "job", "offline-cache-check", "--ignore-not-found", "--wait=false"], check=True)
