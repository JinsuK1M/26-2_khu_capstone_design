"""보드 전용 HW FCM→서버 탐지. 기존 SW와 별도 fcm_hw_v1 모드, 28B 헤더."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import struct
import sys
import time
import cv2
import numpy as np
import torch
from rknnlite.api import RKNNLite
from compare_fcm_hw import Hardware,Gst
from live_dark3_fcm_server import call_fenet_npu,validate_server_result
from replay_fcm_server import validate_inputs,prepare_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,default=9997)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--warmup',type=int,default=5)
    args=p.parse_args()
    if min(args.limit,args.warmup)<0:p.error('limit/warmup은 0 이상')
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root.parent/'network_test'))
    from board_send import ServerLink
    bundle=root.parent/'fcm_l37/fcm_bundle';sys.path.insert(0,str(bundle))
    from fcm_common import quantize_feature
    inputs=args.inputs.expanduser().resolve();manifest,frames=validate_inputs(inputs)
    frames=frames[:args.limit or None]
    torch.set_num_threads(4);Gst.init(None)
    out=root/'replay_fcm_hw_server_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
    backbone=RKNNLite();fenet=RKNNLite();hw=None;link=None;rows=[];warm=[]
    report=dict(state='running',expected_frames=len(frames),mode='fcm_hw_v1',qp=20,codec='8bit NV12 all-I',
        manifest_sha256=__import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),
        note='PNG 읽기 제외, 보드 전처리/dark3/FENet/양자화/10→8변환/HW압축/헤더/전송/서버복원탐지/결과반환 포함. 초기화와 워밍업 제외. 센서 E2E/mAP 아님.')
    try:
        for runtime,path in [(backbone,root/'dark3_only/dark3_fp.rknn'),(fenet,bundle/'fenet_npu_candidate/fenet_fp.rknn')]:
            if runtime.load_rknn(str(path))!=0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
                raise RuntimeError('RKNN 초기화 오류')
        hw=Hardware();link=ServerLink(args.host,args.port,timeout=30)
        link.base={'mode':'fcm_hw_v1','ratio':.5,'qp':20}
        def process(frame,fid):
            image=cv2.imread(str(inputs/frame['file']))
            if image is None or image.shape!=(960,1280,3):raise ValueError('이미지 오류')
            start=time.perf_counter();inp=prepare_image(image);prep=time.perf_counter()
            values=backbone.inference(inputs=[inp]);back=time.perf_counter()
            if values is None or len(values)!=1 or values[0].shape!=(1,256,80,80):raise ValueError('dark3 오류')
            x=np.ascontiguousarray(values[0],dtype=np.float32)
            y=call_fenet_npu(fenet,torch.from_numpy(x));fen=time.perf_counter()
            if not torch.isfinite(y).all() or float(y.max())<=float(y.min()):raise ValueError('FENet 오류')
            q=quantize_feature(y);quant=time.perf_counter()
            bits=hw.encode(q);encoded=time.perf_counter()
            blob=b'FHW1'+struct.pack('<ddHHI',float(y.std()),float(y.mean()),320,640,len(bits))+bits
            edge=time.perf_counter();result=link.send(blob,frame_id=fid);done=time.perf_counter()
            validate_server_result(result,fid)
            if result.get('mode')!='fcm_hw_v1':raise RuntimeError('서버 HW 모드 확인 실패')
            row=dict(frame_id=fid,file=frame['file'],sha256=frame['sha256'],preprocess_ms=(prep-start)*1000,
                npu_ms=(back-prep)*1000,fenet_ms=(fen-back)*1000,quantize_ms=(quant-fen)*1000,
                hw_codec_ms=(encoded-quant)*1000,edge_ms=(edge-start)*1000,e2e_ms=(done-start)*1000,
                payload_bytes=len(blob),rtt_ms=result['rtt_ms'],server_ms=result['server_ms'],
                network_residual_ms=result['rtt_ms']-result['server_ms'],detection_count=len(result['dets']))
            return row,result,blob
        for i in range(args.warmup):
            row,_,_=process(frames[0],-i-1);warm.append(row)
        for i,frame in enumerate(frames):
            row,result,blob=process(frame,frame['frame_id']);rows.append(row)
            (out/f"frame_{i+1:04d}.json").write_text(json.dumps(dict(row=row,response=result),ensure_ascii=False))
            if i in (0,29,59) or i==len(frames)-1:(out/f"frame_{i+1:04d}.bin").write_bytes(blob)
            with (out/'frames.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            print(f"{i+1}/{len(frames)} edge={row['edge_ms']:.1f}ms E2E={row['e2e_ms']:.1f}ms bytes={len(blob)}",flush=True)
        report['state']='completed'
    except BaseException as error:
        report['state']='failed';report['error']=repr(error);raise
    finally:
        if link is not None:link.close()
        if hw is not None:hw.close()
        backbone.release();fenet.release()
        report['completed_frames']=len(rows);report['warmups']=warm
        report['measured']={key:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)))
            for key in ('npu_ms','fenet_ms','quantize_ms','hw_codec_ms','edge_ms','e2e_ms','payload_bytes','rtt_ms','server_ms','network_residual_ms')
            for v in [[row[key] for row in rows]] if v}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('\n=== HW FCM SERVER SUMMARY ===',flush=True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);print('저장:',out,flush=True)


if __name__=='__main__':main()
