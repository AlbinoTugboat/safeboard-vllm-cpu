"""Sustained synthetic load with bounded workers and content-free evidence."""
import argparse
import concurrent.futures
import datetime
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import uuid
from pathlib import Path

from benchmark import one_request, percentile95
from client import api_token, request
from evidence import bind_endpoint, kube_command, server_snapshot, text_sha256


CASES = (("short", 64), ("short", 256), ("long", 64), ("long", 128), ("long_context", 128))
RESOURCE_CODE = '''import json,pathlib
p=pathlib.Path('/sys/fs/cgroup')
def pairs(name): return {k:int(v) for k,v in (line.split() for line in (p/name).read_text().splitlines())}
print(json.dumps({'memory_current_bytes':int((p/'memory.current').read_text()),'memory_peak_bytes':int((p/'memory.peak').read_text()),'memory_max_bytes':(p/'memory.max').read_text().strip(),'memory_stat':{k:v for k,v in pairs('memory.stat').items() if k in ('anon','file','inactive_file','active_file','kernel')},'memory_events':pairs('memory.events'),'cpu_stat':pairs('cpu.stat')}))'''


def summarize(measurements, elapsed):
    successes = [row for row in measurements if row.get('success')]
    result = {'requests': len(measurements), 'successful': len(successes),
              'failed': len(measurements) - len(successes), 'wall_s': elapsed,
              'output_tokens': sum(row['output_tokens'] for row in successes),
              'output_tokens_per_s': sum(row['output_tokens'] for row in successes) / elapsed if elapsed > 0 else 0}
    if successes:
        prompt_tokens = [row['prompt_tokens'] for row in successes]
        result['prompt_tokens'] = {'mean': statistics.mean(prompt_tokens), 'min': min(prompt_tokens), 'max': max(prompt_tokens)}
        for key in ('ttft_s', 'latency_s', 'tpot_s'):
            values = [row[key] for row in successes]
            result[key] = {'mean': statistics.mean(values), 'p95': percentile95(values), 'max': max(values)}
    return result


def prompt_text(kind):
    if kind == 'short':
        return 'List simple facts about the ocean.'
    return 'The local service processes synthetic benchmark data. ' * (460 if kind == 'long_context' else 210)


def resource_snapshot(context, pod):
    kube = kube_command(context)
    info = json.loads(subprocess.check_output(kube + ['get', 'pod', pod, '-o', 'json'], text=True, timeout=10))
    status = next(c for c in info['status']['containerStatuses'] if c['name'] == 'inference')
    resources = json.loads(subprocess.check_output(kube + ['exec', pod, '-c', 'inference', '--',
                                                        'python', '-c', RESOURCE_CODE], text=True, timeout=15))
    return {'pod_uid': info['metadata']['uid'], 'ready': status['ready'],
            'restart_count': status['restartCount'], **resources}


def run(base, duration_seconds=1200, concurrency=4, context='safeboard', sample_interval=15, checkpoint_path=None):
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    token = api_token(context=context)
    before = server_snapshot(context)
    config_hash = text_sha256((Path(__file__).resolve().parents[1] / 'k8s/serve.yaml').read_text(encoding='utf-8'))
    if before['serve_config_sha256'] != config_hash:
        raise RuntimeError('Pod settings differ from local serve.yaml')
    binding_before = bind_endpoint(base, token, before)
    marker = 'SOAK_PRIVATE_' + uuid.uuid4().hex
    # Warm every workload before the measured window; use a separate cache salt.
    for kind, tokens in CASES:
        text = prompt_text(kind)
        one_request(base, token, marker + ' Warmup. ' + text, min(tokens, 32), cache_salt=uuid.uuid4().hex)
    initial_resource = resource_snapshot(context, before['pod_name'])
    measurements, samples = [], []
    lock = threading.Lock()
    stop = threading.Event()
    started = time.perf_counter()
    deadline = started + duration_seconds
    counter = 0

    def checkpoint():
        if checkpoint_path is None:
            return
        with lock:
            partial_rows, partial_samples = list(measurements), list(samples)
        partial = {'status': 'running', 'started_at_utc': started_at,
                   'duration_requested_s': duration_seconds, 'concurrency': concurrency,
                   'kubernetes_evidence': {'before': before},
                   'summary': summarize(partial_rows, time.perf_counter() - started),
                   'resource_samples': partial_samples, 'measurements': partial_rows}
        temporary = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
        temporary.write_text(json.dumps(partial, indent=2) + '\n', encoding='utf-8', newline='\n')
        temporary.replace(checkpoint_path)

    checkpoint()

    def worker():
        nonlocal counter
        while time.perf_counter() < deadline:
            with lock:
                index = counter
                counter += 1
            kind, tokens = CASES[index % len(CASES)]
            text = prompt_text(kind)
            dispatched = time.perf_counter() - started
            row = {'index': index, 'prompt_kind': kind, 'requested_output_tokens': tokens, 'dispatched_s': dispatched}
            try:
                row.update(one_request(base, token, marker + f' Case {index:06d}. ' + text,
                                       tokens, cache_salt=uuid.uuid4().hex), success=True)
            except Exception as exc:
                row.update(success=False, error_type=type(exc).__name__)
                if isinstance(exc, urllib.error.HTTPError):
                    row['http_status'] = exc.code
                    exc.close()
                # Avoid an unbounded error storm when the endpoint is unavailable.
                time.sleep(1)
            row['finished_s'] = time.perf_counter() - started
            with lock:
                measurements.append(row)

    def monitor():
        while not stop.is_set():
            sample = {'elapsed_s': time.perf_counter() - started}
            try:
                sample.update(resource_snapshot(context, before['pod_name']))
                with request(base, '/health', timeout=5) as response:
                    sample['health_status'] = response.status
            except Exception as exc:
                sample['error_type'] = type(exc).__name__
            with lock:
                samples.append(sample)
            stop.wait(sample_interval)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker) for _ in range(concurrency)]
            last_progress = started
            while not all(future.done() for future in futures):
                now = time.perf_counter()
                if now - last_progress >= 60:
                    with lock:
                        progress = summarize(list(measurements), now - started)
                    print(json.dumps({'event': 'soak_progress', **progress}), flush=True)
                    checkpoint()
                    last_progress = now
                time.sleep(1)
            for future in futures:
                future.result()
    finally:
        stop.set()
        watcher.join(timeout=max(45, sample_interval + 5))
        if watcher.is_alive():
            raise RuntimeError('Resource monitor did not stop')
    elapsed = time.perf_counter() - started
    final_resource = resource_snapshot(context, before['pod_name'])
    samples.append({'elapsed_s': elapsed, **final_resource})
    after = server_snapshot(context)
    if after != before:
        raise RuntimeError('Pod or configuration changed during the sustained test')
    binding_after = bind_endpoint(base, token, after)
    logs = subprocess.check_output(kube_command(context) + ['logs', before['pod_name'], '-c', 'inference'], text=True)
    privacy_passed = marker not in logs and token not in logs
    summary = summarize(measurements, elapsed)
    health_passed = all(sample.get('health_status', 200) == 200 and sample.get('ready') is True
                        and 'error_type' not in sample for sample in samples)
    restarts_passed = all(sample['restart_count'] == before['restart_count'] for sample in samples if 'restart_count' in sample)
    oom_delta = {key: final_resource['memory_events'][key] - initial_resource['memory_events'][key]
                 for key in ('oom', 'oom_kill')}
    passed = summary['failed'] == 0 and health_passed and restarts_passed and privacy_passed and not any(oom_delta.values())
    return {'started_at_utc': started_at, 'finished_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
            'status': 'passed' if passed else 'failed', 'duration_requested_s': duration_seconds,
            'concurrency': concurrency, 'resource_sample_interval_s': sample_interval,
            'request_scheduling': 'closed loop; no new dispatch after duration; in-flight requests drained',
            'privacy_marker_absent_from_logs': privacy_passed, 'api_token_absent_from_logs': token not in logs,
            'health_samples_passed': health_passed, 'restart_count_unchanged': restarts_passed,
            'oom_events_delta': oom_delta, 'summary': summary,
            'cases': [{'prompt': kind, 'output_tokens_each': tokens,
                       **summarize([m for m in measurements if m['prompt_kind'] == kind and m['requested_output_tokens'] == tokens], elapsed)}
                      for kind, tokens in CASES],
            'local_config_sha256': config_hash,
            'kubernetes_evidence': {'status': 'pod_verified', 'before': before, 'after': after,
                                    'binding_request_ids': [binding_before, binding_after]},
            'resources_before': initial_resource, 'resource_samples': samples,
            'measurements': sorted(measurements, key=lambda row: row['index']),
            'notes': 'Synthetic prompts only; unique cache_salt for each request; no response content stored. Cgroup memory includes file cache. Per-case throughput is contribution over the whole mixed window, not an isolated workload benchmark. p95 nearest rank; no retry of failed requests.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--duration-seconds', type=int, default=1200)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--kube-context', default='safeboard')
    parser.add_argument('--output', default='results/soak-test.json')
    args = parser.parse_args()
    if not 60 <= args.duration_seconds <= 7200 or not 1 <= args.concurrency <= 8:
        parser.error('Use a duration of 60..7200 seconds and concurrency of 1..8')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_name(output.stem + '.in-progress.json')
    try:
        report = run(args.url, args.duration_seconds, args.concurrency, args.kube_context, checkpoint_path=checkpoint_path)
    except Exception as exc:
        partial = json.loads(checkpoint_path.read_text(encoding='utf-8')) if checkpoint_path.exists() else {}
        partial.update(status='aborted', error_type=type(exc).__name__)
        output.write_text(json.dumps(partial, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(json.dumps({'event': 'soak_aborted', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({'event': 'soak_complete', 'status': report['status'], **report['summary']}), flush=True)
    raise SystemExit(0 if report['status'] == 'passed' else 1)
