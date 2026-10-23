$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
Push-Location $projectRoot
try {
    docker info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop first' }
    $profileJson = minikube status -p safeboard --output=json 2>$null
    $profileReady = $false
    if ($LASTEXITCODE -eq 0) {
        $profileStatus = $profileJson | ConvertFrom-Json
        $profileReady = $profileStatus.Host -eq 'Running' -and $profileStatus.Kubelet -eq 'Running' -and $profileStatus.APIServer -eq 'Running'
    }
    if (-not $profileReady) {
        minikube start -p safeboard --driver=docker --container-runtime=containerd --cpus=12 --memory=16384 --kubernetes-version=v1.36.5 --listen-address=127.0.0.1
        if ($LASTEXITCODE -ne 0) { throw 'Minikube start failed' }
    }
    docker build -t safeboard-vllm:0.32.0 -f Dockerfile .
    if ($LASTEXITCODE -ne 0) { throw 'vLLM build failed' }
    docker build -t safeboard-downloader:1.3 -f Dockerfile.downloader .
    if ($LASTEXITCODE -ne 0) { throw 'Downloader build failed' }
    minikube image load safeboard-vllm:0.32.0 safeboard-downloader:1.3 -p safeboard
    if ($LASTEXITCODE -ne 0) { throw 'Image import failed' }
    kubectl --context safeboard apply -f k8s/namespace.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Namespace setup failed' }
    python scripts/create_secret.py --context safeboard
    if ($LASTEXITCODE -ne 0) { throw 'Secret setup failed' }
    $existingDeployment = kubectl --context safeboard -n safeboard-llm get deployment vllm --ignore-not-found -o name
    if ($LASTEXITCODE -ne 0) { throw 'Deployment lookup failed' }
    kubectl --context safeboard apply -k .
    if ($LASTEXITCODE -ne 0) { throw 'Kustomize deployment failed' }
    if ($existingDeployment) {
        kubectl --context safeboard -n safeboard-llm rollout restart deployment/vllm
        if ($LASTEXITCODE -ne 0) { throw 'Deployment restart failed' }
    }
    kubectl --context safeboard -n safeboard-llm rollout status deployment/vllm --timeout=15m
    if ($LASTEXITCODE -ne 0) { throw 'Deployment is not ready; inspect init container and inference logs' }
} finally {
    Pop-Location
}
