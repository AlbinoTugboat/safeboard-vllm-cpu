"""Create a local API token without printing it or passing it in process arguments."""
import argparse
import base64
import json
import os
import secrets
import subprocess


def create_secret(context="safeboard"):
    # The deployment target is explicit; an unrelated environment must not select it.
    kube = ["kubectl", "--context", context, "-n", "safeboard-llm"]
    existing = subprocess.run(kube + ["get", "secret", "vllm-tokens", "--ignore-not-found", "-o", "name"],
                              capture_output=True, text=True, check=True)
    if existing.stdout.strip():
        print("Existing vllm-tokens Secret retained")
    else:
        encode = lambda value: base64.b64encode(value.encode()).decode()
        obj = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
               "metadata": {"name": "vllm-tokens", "namespace": "safeboard-llm"},
               "data": {"api-token": encode(secrets.token_urlsafe(36)),
                        "hf-token": encode(os.getenv("HF_TOKEN", ""))}}
        subprocess.run(kube + ["create", "-f", "-"], input=json.dumps(obj), text=True, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="safeboard")
    create_secret(parser.parse_args().context)
