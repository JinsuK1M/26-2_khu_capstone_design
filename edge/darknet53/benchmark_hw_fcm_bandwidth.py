"""HW FCM만 5/3/1/0.5Mbps에서 동일 PNG60장 측정. 기존 경로는 실행하지 않습니다."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_bandwidth import tc, qdiscs, shape, restore
from run_replay_aba import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,default=9997,choices=[9997])
    parser.add_argument('--rates',nargs='+',default=['5','3','1','0.5'],choices=['5','3','1','0.5'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    script=root/'replay_fcm_hw_server.py'
    if not script.is_file():raise FileNotFoundError(script)
    inputs=args.inputs.expanduser().resolve();manifest=inputs/'manifest.json'
    data=json.loads(manifest.read_text());frames=data['frames']
    if data.get('state')!='completed' or len(frames)!=60:raise ValueError('완료된 60장 입력 필요')
    expected=[(Path(f['file']).name,f['sha256']) for f in frames]
    for frame in frames:
        path=(inputs/frame['file']).resolve()
        if not path.is_relative_to(inputs) or hashlib.sha256(path.read_bytes()).hexdigest()!=frame['sha256']:
            raise ValueError('입력 경로/SHA256 오류')
    subprocess.run(['sudo','-v'],check=True)
    original=qdiscs()
    if len(original)!=1 or original[0].get('kind')!='fq_codel':
        raise RuntimeError('기본 fq_codel 상태가 아닙니다. 설정은 변경하지 않았습니다.')
    route=subprocess.check_output(['ip','route','get',args.host],text=True)
    if 'dev tailscale0' not in route:raise RuntimeError('tailscale0 경로 아님')
    out=root/'hw_fcm_bandwidth_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
    report=dict(state='running',mode='fcm_hw_v1',runs=[],original_qdisc=original,route=route,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        conditions=dict(qp=20,format='8bit NV12 all-I',warmup=5,threads=4,
                        shaping='server TCP upload only; TBF burst16kb latency5s'),
        note='HW FCM만 추가 측정. 과거 JPEG/HEVC/SW 결과와 실행 시점이 다름. PNG읽기/초기화/워밍업 제외, 압축 포함 결과반환까지. 센서 E2E/mAP 아님.')
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    try:
        for rate in args.rates:
            folder=out/rate;folder.mkdir()
            shape(rate,args.host,args.port)
            (folder/'qdisc_before.txt').write_text(tc('-s','qdisc','show','dev','tailscale0',capture=True))
            (folder/'filters.txt').write_text(tc('filter','show','dev','tailscale0',capture=True))
            before=set((root/'replay_fcm_hw_server_runs').glob('*'))
            command=[sys.executable,'-u',str(script),'--inputs',str(inputs),'--host',args.host,
                     '--port',str(args.port),'--warmup','5']
            print(f'\n=== HW FCM / 업로드 {rate}Mbps ===',flush=True)
            with (folder/'execution.log').open('w') as log:
                child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                try:
                    for line in child.stdout:
                        print(line,end='',flush=True);log.write(line);log.flush()
                    if child.wait()!=0:raise RuntimeError(f'{rate}Mbps 실행 실패')
                finally:
                    if child.poll() is None:
                        child.terminate();child.wait()
            added=set((root/'replay_fcm_hw_server_runs').glob('*'))-before
            if len(added)!=1:raise RuntimeError('결과 폴더 식별 실패')
            directory=added.pop();result=json.loads((directory/'report.json').read_text())
            if result['state']!='completed' or result['completed_frames']!=60:raise RuntimeError('60장 미완료')
            with (directory/'frames.csv').open() as file:rows=list(csv.DictReader(file))
            if [(Path(r['file']).name,r['sha256']) for r in rows]!=expected:raise RuntimeError('입력 순서/해시 불일치')
            report['runs'].append(dict(rate_mbps=rate,frames=60,directory=str(directory),
                                       measured=result['measured']))
            (folder/'qdisc_after.txt').write_text(tc('-s','qdisc','show','dev','tailscale0',capture=True))
            save()
        report['state']='completed'
    except BaseException as error:
        report['state']='failed';report['error']=repr(error);raise
    finally:
        try:
            restore();report['restored']=True
            (out/'qdisc_restored.txt').write_text(tc('-s','qdisc','show','dev','tailscale0',capture=True))
        except BaseException as error:
            report['restored']=False;report['restore_error']=repr(error)
        save()
        print('\n=== HW FCM BANDWIDTH SUMMARY ===',flush=True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);print('저장:',out,flush=True)


if __name__=='__main__':main()
