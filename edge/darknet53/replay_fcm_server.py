"""Replay a hashed PNG input manifest through the actual board FCM/server path.

Every saved frame is measured once after separate warmups. No live capture or
frame skipping. Disk loading is measured separately and excluded from E2E.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import torch
from rknnlite.api import RKNNLite
from live_dark3_fcm_server import call_fenet_npu, sha, validate_server_result

OPTIONS=['--RDOQ=0','--RDOQTS=0','--TransformSkip=0','--TransformSkipFast=0',
         '--QuadtreeTUMaxDepthIntra=1','--SAO=0']


def prepare_image(image):
    if image is None or image.shape != (960,1280,3) or image.dtype != np.uint8:
        raise ValueError('Expected original 1280x960 BGR uint8 PNG')
    canvas=np.full((640,640,3),114,np.uint8)
    canvas[:480]=cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    return canvas[None]


def validate_inputs(root):
    manifest_file=root/'manifest.json'
    manifest=json.loads(manifest_file.read_text())
    frames=manifest.get('frames',[])
    if manifest.get('state')!='completed' or manifest.get('count')!=len(frames) or not frames:
        raise ValueError('Input manifest incomplete')
    ids=[f['frame_id'] for f in frames]
    if any(type(i) is not int or i<1 for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('Expected unique increasing positive frame IDs')
    if manifest.get('width')!=1280 or manifest.get('height')!=960:
        raise ValueError('Unexpected input dimensions')
    for frame in frames:
        path=(root/frame['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=frame['sha256']:
            raise ValueError('Missing, changed or invalid frame: '+str(path))
    return manifest_file,frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,default=9997)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=30)
    args=parser.parse_args()
    if args.warmup<0 or not 1<=args.port<=65535 or args.timeout<=0:
        parser.error('Invalid warmup, port or timeout')
    root=Path(__file__).resolve().parent
    input_root=args.inputs.expanduser().resolve()
    manifest_file,frames=validate_inputs(input_root)
    bundle=Path.home()/'yolox_project/fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_edge
    sys.path.insert(0,str(root.parent/'network_test'))
    from board_send import ServerLink
    torch.set_num_threads(4)
    backbone_model=root/'dark3_only/dark3_fp.rknn'
    fenet_model=bundle/'fenet_npu_candidate/fenet_fp.rknn'
    build=json.loads((fenet_model.parent/'build.json').read_text())
    if sha(fenet_model)!=build['rknn_sha256']:
        raise ValueError('FENet model hash mismatch')
    for path in (backbone_model,bundle/'fenet_l37.pt',Path(fcm_edge.HM_DIR)/'bin/TAppEncoderStatic'):
        if not path.is_file():raise FileNotFoundError(path)
    out=root/'replay_fcm_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    (out/'payloads').mkdir(parents=True)
    clock={}
    original_quant,original_hm,original_subprocess=fcm_edge.quantize_feature,fcm_edge._hm_encode,fcm_edge.subprocess
    def quantize(y):
        tick=time.perf_counter();value=original_quant(y)
        clock['quantize_ms']=(time.perf_counter()-tick)*1000;return value
    def encode(frame,qp):
        tick=time.perf_counter();value=original_hm(frame,qp)
        clock['hm_ms']=(time.perf_counter()-tick)*1000;return value
    def run_hm(command,**kw):
        kw.setdefault('timeout',120)
        return original_subprocess.run(list(command)+OPTIONS,**kw)
    fcm_edge.quantize_feature,fcm_edge._hm_encode=quantize,encode
    fcm_edge.subprocess=types.SimpleNamespace(run=run_hm)
    backbone,fenet=RKNNLite(),RKNNLite()
    link=None
    rows=[]
    state='starting'
    failure=None
    replay_start=None
    last_done=None
    warmups=[]
    try:
        if backbone.load_rknn(str(backbone_model))!=0 or backbone.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
            raise RuntimeError('Backbone NPU initialization failed')
        if fenet.load_rknn(str(fenet_model))!=0 or fenet.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
            raise RuntimeError('FENet NPU initialization failed')
        link=ServerLink(args.host,args.port,qp=20,hm='combined4',ratio=.5,timeout=args.timeout)
        class Encoder:
            def __call__(self,x):
                tick=time.perf_counter();y=call_fenet_npu(fenet,x)
                clock['fenet_ms']=(time.perf_counter()-tick)*1000
                if not torch.isfinite(y).all() or float(y.max())<=float(y.min()):
                    raise RuntimeError('Invalid FENet output')
                return y
        def process(image,frame_id):
            begin=time.perf_counter()
            inputs=prepare_image(image)
            prep=(time.perf_counter()-begin)*1000
            tick=time.perf_counter();values=backbone.inference(inputs=[inputs])
            npu=(time.perf_counter()-tick)*1000
            if values is None or len(values)!=1 or values[0].shape!=(1,256,80,80):
                raise RuntimeError('Invalid dark3 shape')
            dark3=np.ascontiguousarray(values[0],dtype=np.float32)
            if not np.isfinite(dark3).all():raise RuntimeError('Nonfinite dark3')
            clock.clear()
            tick=time.perf_counter();blob=fcm_edge.compress(Encoder(),torch.from_numpy(dark3),qp=20)
            compress=(time.perf_counter()-tick)*1000
            edge_done=time.perf_counter()
            result=link.send(blob,frame_id=frame_id)
            done=time.perf_counter()
            net=validate_server_result(result,frame_id)
            request_header=json.dumps(dict(link.base,frame_id=frame_id)).encode()
            row=dict(frame_id=frame_id,preprocess_ms=prep,npu_ms=npu,**clock,
                     compress_ms=compress,edge_ms=(edge_done-begin)*1000,**net,
                     e2e_ms=(done-begin)*1000,payload_bytes=len(blob),
                     application_request_bytes=8+len(request_header)+len(blob))
            return row,blob,result,done
        print('입력 확인:',input_root,'frames=',len(frames),flush=True)
        print('출력:',out,flush=True)
        for index in range(args.warmup):
            image=cv2.imread(str(input_root/frames[0]['file']))
            row,_,_,_=process(image,-index)
            warmups.append(row)
            print('Warmup complete:',index+1,flush=True)
        with (out/'frames.csv').open('w',newline='') as csvfile,(out/'detections.jsonl').open('w') as detfile:
            writer=None
            replay_start=time.perf_counter()
            for frame in frames:
                failure=dict(frame_id=frame['frame_id'],file=frame['file'])
                tick=time.perf_counter()
                image=cv2.imread(str(input_root/frame['file']))
                file_load=(time.perf_counter()-tick)*1000
                row,blob,result,last_done=process(image,frame['frame_id'])
                row.update(file=frame['file'],sha256=frame['sha256'],file_load_ms=file_load)
                if writer is None:
                    writer=csv.DictWriter(csvfile,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);csvfile.flush()
                rows.append(row)
                (out/'payloads'/f"frame_{frame['frame_id']:04d}.bin").write_bytes(blob)
                detfile.write(json.dumps(dict(file=frame['file'],sha256=frame['sha256'],**result),ensure_ascii=False)+'\n');detfile.flush()
                print(f"{len(rows)}/{len(frames)} {frame['file']} edge={row['edge_ms']:.1f}ms "
                      f"RTT={row['rtt_ms']:.1f}ms E2E={row['e2e_ms']:.1f}ms "
                      f"bytes={row['payload_bytes']} dets={row['detection_count']}",flush=True)
                failure=None
        state='completed'
    except KeyboardInterrupt:
        state='interrupted'
    except Exception as error:
        state='failed: '+repr(error)
        raise
    finally:
        if link is not None:link.close()
        backbone.release();fenet.release()
        fcm_edge.quantize_feature,fcm_edge._hm_encode,fcm_edge.subprocess=original_quant,original_hm,original_subprocess
        keys=('file_load_ms','preprocess_ms','npu_ms','fenet_ms','hm_ms','compress_ms',
              'edge_ms','rtt_ms','server_ms','network_residual_ms','e2e_ms',
              'payload_bytes','application_request_bytes','detection_count')
        stats={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),
                      p95=float(np.percentile([r[k] for r in rows],95))) for k in keys} if rows else {}
        wall=last_done-replay_start if last_done is not None and replay_start is not None else None
        report=dict(state=state,mode='B_FCM',input_root=str(input_root),manifest_sha256=sha(manifest_file),
            expected_frames=len(frames),completed_frames=len(rows),failure=failure,
            frame_ids=[r['frame_id'] for r in rows],input_sha256=[r['sha256'] for r in rows],
            qp=20,hm_profile='combined4',hm_options=OPTIONS,fenet_backend='npu',threads=4,
            host=args.host,port=args.port,ratio=.5,warmups=warmups,measured=stats,
            replay_wall_seconds=wall,replay_fps=len(rows)/wall if wall else None,
            total_payload_bytes=sum(r['payload_bytes'] for r in rows),
            total_application_request_bytes=sum(r['application_request_bytes'] for r in rows),
            backbone_sha256=sha(backbone_model),fenet_sha256=sha(fenet_model),
            note='All saved frames in manifest order, no frame skipping, no real-time pacing. Warmups separate; all measured frames retained. Per-frame E2E starts after file loading and ends at matched server result. Replay FPS includes intermediate I/O; not live camera FPS. Application bytes exclude TCP/IP/VPN overhead. No sensor acquisition E2E or mAP.')
        (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print('\n=== FIXED INPUT FCM REPLAY SUMMARY ===',flush=True)
        summary={key:report[key] for key in ('state','expected_frames','completed_frames',
            'manifest_sha256','replay_fps','total_payload_bytes','measured')}
        print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
        print('저장:',out,flush=True)


if __name__=='__main__':
    main()
