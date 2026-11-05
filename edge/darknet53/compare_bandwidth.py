"""JPEG/HEVC/FCM 동일 입력 비교. tailscale0의 지정 서버 TCP 업로드만 제한하고 종료 시 복구."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_replay_aba import summarize


def tc(*args, capture=False):
    return subprocess.run(['sudo', 'tc', *args], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def qdiscs():
    return json.loads(tc('-j', 'qdisc', 'show', 'dev', 'tailscale0', capture=True))


def restore():
    current = qdiscs()
    # 기본 상태이면 불필요한 replace를 하지 않습니다.
    if len(current) == 1 and current[0].get('kind') == 'fq_codel':
        return
    # 기존 prio의 handle 1:과 충돌하지 않는 별도 handle을 명시합니다.
    tc('qdisc', 'replace', 'dev', 'tailscale0', 'root', 'handle', '7ffe:', 'fq_codel',
       'limit', '10240', 'flows', '1024', 'quantum', '1280', 'target', '5ms',
       'interval', '100ms', 'memory_limit', '33554432', 'ecn')
    after = qdiscs()
    if len(after) != 1 or after[0].get('kind') != 'fq_codel':
        raise RuntimeError('fq_codel 복구 후 상태 확인 실패')


def shape(rate, host, port):
    restore()
    if rate == 'unlimited':
        return
    tc('qdisc', 'replace', 'dev', 'tailscale0', 'root', 'handle', '1:',
       'prio', 'bands', '3', 'priomap', *(['1'] * 16))
    tc('qdisc', 'add', 'dev', 'tailscale0', 'parent', '1:1', 'handle', '10:',
       'tbf', 'rate', rate+'mbit', 'burst', '16kb', 'latency', '5s')
    tc('filter', 'add', 'dev', 'tailscale0', 'protocol', 'ip', 'parent', '1:',
       'pref', '10', 'flower', 'ip_proto', 'tcp', 'dst_ip', host,
       'dst_port', str(port), 'classid', '1:1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=9997, choices=[9997])
    parser.add_argument('--rates', nargs='+', default=['unlimited','5','2','1','0.5'],
                        choices=['unlimited','5','3','2','1','0.5'])
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds는 1 이상')
    root = Path(__file__).resolve().parent
    inputs = args.inputs.expanduser().resolve()
    manifest = inputs / 'manifest.json'
    data = json.loads(manifest.read_text())
    if data.get('state') != 'completed' or len(data['frames']) != 60:
        raise ValueError('완료된 PNG 60장 입력 세트 필요')
    expected = [(Path(f['file']).name, f['sha256']) for f in data['frames']]
    for frame in data['frames']:
        path = (inputs / frame['file']).resolve()
        if not path.is_relative_to(inputs) or hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('입력 경로/SHA256 불일치')
    for name in ('jpeg_send.py', 'replay_hevc_live_encode.py', 'replay_fcm_server.py', 'hevc_send.py'):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    subprocess.run(['sudo', '-v'], check=True)
    original = json.loads(tc('-j', 'qdisc', 'show', 'dev', 'tailscale0', capture=True))
    if len(original) != 1 or original[0].get('kind') != 'fq_codel':
        raise RuntimeError('예상한 기본 fq_codel 상태가 아닙니다. 기존 설정을 변경하지 않습니다.')
    route = subprocess.check_output(['ip','route','get',args.host], text=True)
    if 'dev tailscale0' not in route:
        raise RuntimeError('서버 경로가 tailscale0가 아닙니다')
    out = root / 'bandwidth_comparisons' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               FCM_HM_DIR=str(Path.home()/'yolox_project/fcm_l37/fcm_bundle/hm'))
    report = dict(state='running', runs=[], original_qdisc=original, route=route,
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  conditions=dict(jpeg_quality=90,hevc_bitrate=1000000,hevc_gop=30,hevc_fps_metadata=1,
                                  fcm_qp=20,hm='combined4',fenet='npu',warmup=5,
                                  shaping='tailscale0; server TCP upload only; TBF burst16kb latency5s'),
                  note='동일 PNG60장, 파일읽기 제외, 압축 포함 결과반환까지. 카메라 E2E/mAP 아님. 제한값은 VPN 전 IPv4 업로드 상한이며 실측 가용 대역폭과 다름.')
    def save():
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    try:
        for rate_index, rate in enumerate(args.rates):
            shape(rate,args.host,args.port)
            rate_dir = out / rate
            rate_dir.mkdir()
            (rate_dir/'qdisc_before.txt').write_text(tc('-s','qdisc','show','dev','tailscale0',capture=True))
            (rate_dir/'filters.txt').write_text(tc('filter','show','dev','tailscale0',capture=True))
            for repeat in range(args.rounds):
                # 순서 편향을 줄이기 위해 조건/반복마다 실행 순서를 회전합니다.
                modes = ['jpeg','hevc','fcm']
                shift = (rate_index + repeat) % 3
                modes = modes[shift:] + modes[:shift]
                for mode in modes:
                    name=f'{repeat+1:02d}_{mode}'
                    common=['--host',args.host,'--port',str(args.port),'--warmup','5']
                    folder_name={'hevc':'replay_hevc_encode_runs','fcm':'replay_fcm_runs'}.get(mode)
                    before=set((root/folder_name).glob('*')) if folder_name else set()
                    if mode=='jpeg':
                        directory=rate_dir/name
                        command=[sys.executable,'-u',str(root/'jpeg_send.py'),str(manifest),*common,
                                 '--quality','90','--strict','--out',str(directory)]
                    else:
                        script='replay_hevc_live_encode.py' if mode=='hevc' else 'replay_fcm_server.py'
                        command=[sys.executable,'-u',str(root/script),'--inputs',str(inputs),*common]
                    print(f'\n=== 업로드 {rate} Mbps / {name} ===',flush=True)
                    with (rate_dir/(name+'.log')).open('w') as log:
                        process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT,text=True)
                        try:
                            for line in process.stdout:
                                print(line,end='',flush=True)
                                log.write(line); log.flush()
                            if process.wait()!=0:
                                raise RuntimeError(f'{rate}/{name} 실패')
                        finally:
                            if process.poll() is None:
                                process.terminate(); process.wait()
                    if folder_name:
                        added=set((root/folder_name).glob('*'))-before
                        if len(added)!=1:
                            raise RuntimeError('결과 폴더 식별 실패')
                        directory=added.pop()
                        child=json.loads((directory/'report.json').read_text())
                        if child['state']!='completed':
                            raise RuntimeError('하위 실험 미완료')
                    with (directory/'frames.csv').open() as file:
                        rows=list(csv.DictReader(file))
                    if [(Path(r['file']).name,r['sha256']) for r in rows]!=expected:
                        raise RuntimeError('입력 순서/개수/SHA256 불일치')
                    result=dict(rate_mbps=rate,repeat=repeat+1,mode=mode,frames=len(rows),
                                directory=str(directory),measured=summarize(rows,'jpeg' if mode=='jpeg' else 'fcm'))
                    report['runs'].append(result); save()
                    print('조건 결과:',json.dumps(result,ensure_ascii=False),flush=True)
            (rate_dir/'qdisc_after.txt').write_text(tc('-s','qdisc','show','dev','tailscale0',capture=True))
        report['state']='completed'
    except BaseException as error:
        report['state']='failed'; report['error']=repr(error)
        raise
    finally:
        try:
            restore()
            report['restored']=True
        except BaseException as error:
            report['restored']=False; report['restore_error']=repr(error)
            print('복구 실패: 수동 fq_codel 복구가 필요합니다.',flush=True)
        save()
        print('\n=== BANDWIDTH COMPARISON SUMMARY ===',flush=True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        print('저장:',out,flush=True)


if __name__=='__main__':
    main()
