# Local Docker Desktop workaround: its synthetic DNS address is translated in
# the node OUTPUT chain, which ordinary Pod-network traffic does not traverse.
$ErrorActionPreference = 'Stop'
$patch = '{"spec":{"template":{"spec":{"hostNetwork":true,"dnsPolicy":"Default"}}}}'
$patchFile = [IO.Path]::GetTempFileName()
try {
    [IO.File]::WriteAllText($patchFile, $patch, [Text.UTF8Encoding]::new($false))
    kubectl --context safeboard -n kube-system patch deployment coredns --type merge --patch-file $patchFile
    if ($LASTEXITCODE -ne 0) { throw 'CoreDNS patch failed' }
} finally {
    Remove-Item -LiteralPath $patchFile
}
kubectl --context safeboard -n kube-system rollout status deployment/coredns --timeout=60s
if ($LASTEXITCODE -ne 0) { throw 'CoreDNS is not ready' }
