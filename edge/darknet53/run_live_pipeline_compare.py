"""Board-only live B/A comparison; no persistent setting changes."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    live = root / 'live_dark3_fcm_fenet_compare.py'
    pipeline = root / 'live_dark3_fcm_pipeline.py'
    verify = Path.home() / 'yolox_project/fcm_l37/verify_live_fcm.py'
    assert all(p.is_file() for p in (live,pipeline,verify)), '비교 또는 복원 검사 파일이 없습니다.'
    target = root / 'live_pipeline_comparisons' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    target.mkdir(parents=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    reports = []
    for name, use_pipeline in [('sequential_before',False), ('pipeline',True), ('sequential_after',False)]:
        directory = target / name
        print(f'\n=== {name}: 60초 라이브 측정 ===', flush=True)
        command = [sys.executable,'-u',str(pipeline if use_pipeline else live),
                   '--seconds','60','--run-dir',str(directory)]
        if not use_pipeline:
            command += ['--qp','20','--threads','4','--hm-profile','combined4','--fenet-backend','npu']
        subprocess.run(command,
                       env=env, check=True, timeout=300)
        report = json.loads((directory / 'report.json').read_text())
        assert report['state'] == 'completed' and report['processed'] > 1
        reports.append(dict(name=name,directory=str(directory),report=report))
        (target/'comparison.json').write_text(json.dumps(reports,indent=2))
    # Decode after all timed runs so decoding does not interrupt the A/B/A sequence.
    for item in reports:
        directory = Path(item['directory'])
        samples = sorted((directory/'samples').glob('*')) + [directory]
        item['decode_checks'] = []
        for sample in samples:
            result = subprocess.run([sys.executable,'-u',str(verify),'--run',str(sample)],
                                    env=env,capture_output=True,text=True,timeout=180)
            (sample/'decode_check.log').write_text(result.stdout + result.stderr)
            ok = result.returncode == 0
            check = dict(sample=str(sample),ok=ok)
            if ok:
                check['vs_original'] = json.loads((sample/'decode_report.json').read_text())['comparison_to_precompression_npu_dark3']
            item['decode_checks'].append(check)
            print(f"복원 {'PASS' if ok else 'FAIL'}: {sample}",flush=True)
    (target/'comparison.json').write_text(json.dumps(reports,indent=2))
    print('\n=== LIVE PIPELINE COMPARISON ===',flush=True)
    for item in reports:
        report = item['report']
        stats = report['after_first_frame']
        print(json.dumps(dict(name=item['name'],processed=report['processed'],
            fenet_backend=report['fenet_backend'],hm_profile=report['hm_profile'],
            loop_fps=report['last_loop_avg_fps'],
            npu_median_ms=stats['npu_ms']['median'],
            fenet_median_ms=stats['fenet_ms']['median'],
            hm_median_ms=stats['hm_ms']['median'],
            queue_wait_median_ms=stats.get('queue_wait_ms',{}).get('median',0),
            queue_wait_p95_ms=stats.get('queue_wait_ms',{}).get('p95',0),
            selected_median_ms=stats['selected_process_ms']['median'],
            selected_p95_ms=stats['selected_process_ms']['p95'],
            mean_payload_bytes=stats['payload_bytes']['mean'],
            unfinished_jobs=report.get('unfinished_jobs',0),
            skipped=report['skipped_before_selected'],
            decode_pass=sum(x['ok'] for x in item['decode_checks']),
            decode_total=len(item['decode_checks'])),ensure_ascii=False),flush=True)
    print('저장:', target)
    print('선택한 프레임의 처리 시간입니다. 센서 촬영부터의 E2E나 mAP가 아닙니다.')


if __name__ == '__main__':
    main()
