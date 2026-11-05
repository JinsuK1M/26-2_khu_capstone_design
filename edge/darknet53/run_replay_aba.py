"""동일 입력 JPEG → FCM → JPEG 순차 비교. 네트워크 설정은 변경하지 않습니다."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows, mode):
    fields = {'latency_ms': 'total_ms' if mode == 'jpeg' else 'e2e_ms',
              'payload_bytes': 'jpeg_bytes' if mode == 'jpeg' else 'payload_bytes',
              'server_ms': 'server_ms'}
    return {name: {'mean': statistics.mean(values), 'median': statistics.median(values),
                   'p95': percentile(values, .95)}
            for name, field in fields.items()
            for values in [[float(row[field]) for row in rows]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=9997)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds는 1 이상이어야 합니다')
    root = Path(__file__).resolve().parent
    inputs = args.inputs.expanduser().resolve()
    manifest = inputs / 'manifest.json'
    data = json.loads(manifest.read_text())
    frames = data['frames']
    if data.get('state') != 'completed' or not frames:
        raise ValueError('완료된 입력 세트가 필요합니다')
    expected = [(Path(f['file']).name, f['sha256']) for f in frames]
    for frame in frames:
        path = (inputs / frame['file']).resolve()
        if not path.is_relative_to(inputs) or hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('입력 파일 경로 또는 SHA256 오류')
    for name in ('jpeg_send.py', 'replay_fcm_server.py'):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    out = root / 'replay_aba_runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['FCM_HM_DIR'] = str(Path.home() / 'yolox_project/fcm_l37/fcm_bundle/hm')
    results = []
    report = {'state': 'running', 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'inputs': str(inputs), 'runs': results,
              'note': 'PNG 파일 읽기와 워밍업 제외. 센서 촬영 E2E 또는 mAP가 아닙니다. 대역폭 설정 변경 없음.'}
    try:
        for number in range(1, args.rounds + 1):
            for label, mode in [('A_before', 'jpeg'), ('B', 'fcm'), ('A_after', 'jpeg')]:
                name = f'{number:02d}_{label}'
                common = ['--host', args.host, '--port', str(args.port), '--warmup', '5']
                before = set((root / 'replay_fcm_runs').glob('*'))
                if mode == 'jpeg':
                    run_dir = out / name
                    command = [sys.executable, '-u', str(root / 'jpeg_send.py'), str(manifest),
                               *common, '--quality', '90', '--strict', '--out', str(run_dir)]
                else:
                    command = [sys.executable, '-u', str(root / 'replay_fcm_server.py'),
                               '--inputs', str(inputs), *common]
                print(f'\n=== {name} 시작 ===', flush=True)
                with (out / f'{name}.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True)
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        if process.wait() != 0:
                            raise RuntimeError(f'{name} 실행 실패. 로그 확인 필요')
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                if mode == 'fcm':
                    added = set((root / 'replay_fcm_runs').glob('*')) - before
                    if len(added) != 1:
                        raise RuntimeError('FCM 결과 폴더를 유일하게 식별하지 못했습니다')
                    run_dir = added.pop()
                    state = json.loads((run_dir / 'report.json').read_text())
                    if state['state'] != 'completed':
                        raise RuntimeError('FCM 미완료')
                with (run_dir / 'frames.csv').open() as file:
                    rows = list(csv.DictReader(file))
                observed = [(Path(r['file']).name, r['sha256']) for r in rows]
                if observed != expected:
                    raise RuntimeError(f'{name}: 입력 순서·개수·SHA256 불일치')
                result = dict(name=name, mode=mode, frames=len(rows), directory=str(run_dir),
                              measured=summarize(rows, mode))
                results.append(result)
                (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        report['state'] = 'completed'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print('\n=== A/B/A SUMMARY ===', flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('저장:', out, flush=True)


if __name__ == '__main__':
    main()
