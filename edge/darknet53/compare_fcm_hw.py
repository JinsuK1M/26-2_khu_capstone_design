"""동일 PNG의 동일 NPU FENet 출력에서 SW combined4 vs HW 8bit all-I 후보 비교. 서버 전송 없음."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import cv2
import numpy as np
import torch
from rknnlite.api import RKNNLite
from test_fcm_hw_candidate import metrics, Gst
from live_dark3_fcm_server import call_fenet_npu
from replay_fcm_server import validate_inputs, prepare_image, OPTIONS


class Hardware:
    def __init__(self):
        self.p=Gst.parse_launch('appsrc name=src is-live=true format=time block=true caps=video/x-raw,format=NV12,width=640,height=320,framerate=1/1 '
            '! mpph265enc rc-mode=fixqp qp-init=20 qp-min=20 qp-max=20 qp-min-i=20 qp-max-i=20 gop=1 header-mode=each-idr max-pending=1 '
            '! appsink name=sink sync=false async=false max-buffers=1 drop=false')
        self.src=self.p.get_by_name('src');self.sink=self.p.get_by_name('sink');self.i=0
        if self.p.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:raise RuntimeError('HW 시작 실패')
    def encode(self,q):
        q8=np.rint(q.astype(np.float64)*255/1023).clip(0,255).astype(np.uint8)
        raw=q8.tobytes()+np.full((160,640),128,dtype=np.uint8).tobytes()
        b=Gst.Buffer.new_allocate(None,len(raw),None);b.fill(0,raw);b.pts=self.i*Gst.SECOND;b.duration=Gst.SECOND
        if self.src.emit('push-buffer',b)!=Gst.FlowReturn.OK:raise RuntimeError('HW 입력 실패')
        s=self.sink.emit('try-pull-sample',10*Gst.SECOND)
        if s is None:raise RuntimeError('HW 출력 시간 초과')
        o=s.get_buffer()
        if o.pts!=b.pts:raise RuntimeError('HW PTS 불일치')
        bits=o.extract_dup(0,o.get_size());self.i+=1
        return bits
    def close(self):self.p.set_state(Gst.State.NULL)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.limit<0:ap.error('--limit는 0 이상')
    inputs=args.inputs.expanduser().resolve()
    manifest,frames=validate_inputs(inputs)
    frames=frames[:args.limit or None]
    root=Path(__file__).resolve().parent
    bundle=Path.home()/'yolox_project/fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_edge,fcm_server
    from fcm_common import quantize_feature,dequantize_frame
    torch.set_num_threads(4);Gst.init(None)
    out=root/'fcm_hw_comparisons'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
    backbone=RKNNLite();fenet=RKNNLite();hw=None
    original=fcm_edge.subprocess
    def combined(command,**kwargs):
        return original.run(list(command)+OPTIONS,**kwargs)
    fcm_edge.subprocess=types.SimpleNamespace(run=combined)
    rows=[]
    report=dict(state='running',expected_frames=len(frames),inputs=str(inputs),
        settings='SW combined4 QP20 10bit mono vs HW fixedQP20 8bit NV12 all-I; 동일 QP는 동일 품질 아님',
        note='warmup3 제외. codec_ms는 q10 입력부터 비트스트림 반환까지; HW 10→8변환/패킹 포함. 보드 합산시간은 공통 전처리+dark3+FENet+양자화와 각 codec 시간의 합, 연속 E2E 아님. 복원 CPU시간/서버/네트워크/mAP 제외. bytes는 두 방식 모두 코덱 비트스트림만 포함하며 헤더/메타데이터 제외.')
    try:
        for runtime,path in [(backbone,root/'dark3_only/dark3_fp.rknn'),(fenet,bundle/'fenet_npu_candidate/fenet_fp.rknn')]:
            if runtime.load_rknn(str(path))!=0:raise RuntimeError('RKNN 로딩 실패')
            if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:raise RuntimeError('NPU 초기화 실패')
        decoder=fcm_server.load_decoder();hw=Hardware()
        def common(image):
            tick=time.perf_counter();v=backbone.inference(inputs=[prepare_image(image)])
            if v is None or len(v)!=1 or v[0].shape!=(1,256,80,80):raise RuntimeError('dark3 출력 오류')
            x=np.ascontiguousarray(v[0],dtype=np.float32)
            y=call_fenet_npu(fenet,torch.from_numpy(x))
            if not torch.isfinite(y).all() or float(y.max())<=float(y.min()):raise RuntimeError('FENet 출력 오류')
            q=quantize_feature(y)
            return x,y,q,(time.perf_counter()-tick)*1000
        first=cv2.imread(str(inputs/frames[0]['file']))
        for _ in range(3):
            _,_,q,_=common(first);fcm_edge._hm_encode(q,20);hw.encode(q)
        for i,frame in enumerate(frames):
            image=cv2.imread(str(inputs/frame['file']))
            x,y,q,shared=common(image)
            blobs={};timings={}
            for mode in (['sw','hw'] if i%2==0 else ['hw','sw']):
                tick=time.perf_counter()
                blobs[mode]=fcm_edge._hm_encode(q,20) if mode=='sw' else hw.encode(q)
                timings[mode]=(time.perf_counter()-tick)*1000
            swq=fcm_server._hm_decode(blobs['sw'],320,640)
            hfile=out/'decode_input.hevc';rawfile=out/'decode_y.raw';hfile.write_bytes(blobs['hw'])
            subprocess.run(['ffmpeg','-v','error','-i',str(hfile),'-vf','extractplanes=y','-frames:v','1',
                '-pix_fmt','gray','-f','rawvideo','-y',str(rawfile)],check=True,timeout=30)
            raw=np.fromfile(rawfile,dtype=np.uint8)
            if raw.size!=320*640:raise RuntimeError('HW Y 복원 크기 오류')
            hwq=raw.reshape(320,640).astype(np.float64)*1023/255
            with torch.inference_mode():
                swrec=decoder(dequantize_frame(swq,128,40,40,float(y.std()),float(y.mean()))).numpy()
                hwrec=decoder(dequantize_frame(hwq,128,40,40,float(y.std()),float(y.mean()))).numpy()
            row=dict(frame_id=frame['frame_id'],file=frame['file'],sha256=frame['sha256'],shared_ms=shared,
                sw_codec_ms=timings['sw'],hw_codec_ms=timings['hw'],sw_board_sum_ms=shared+timings['sw'],
                hw_board_sum_ms=shared+timings['hw'],sw_bytes=len(blobs['sw']),hw_bytes=len(blobs['hw']),
                sw_relative_l2=metrics(swrec,x)['relative_l2'],hw_relative_l2=metrics(hwrec,x)['relative_l2'],
                hw_vs_sw_relative_l2=metrics(hwrec,swrec)['relative_l2'])
            rows.append(row)
            with (out/'frames.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            if i in (0,29,59) or i==len(frames)-1:
                sample=out/f"sample_{i+1:04d}";sample.mkdir(exist_ok=True)
                (sample/'hw.hevc').write_bytes(blobs['hw']);np.save(sample/'hw_dark3.npy',hwrec)
                np.save(sample/'sw_dark3.npy',swrec);np.save(sample/'original_dark3.npy',x)
                (sample/'metadata.json').write_text(json.dumps(dict(frame=frame,std=float(y.std()),mean=float(y.mean()),shape=[320,640],codec='hw8_nv12_y_v1'),indent=2))
            print(f"{i+1}/{len(frames)} SW={timings['sw']:.1f}ms HW={timings['hw']:.1f}ms SW/HW bytes={row['sw_bytes']}/{row['hw_bytes']}",flush=True)
        report['state']='completed'
    except BaseException as error:
        report['state']='failed';report['error']=repr(error);raise
    finally:
        if hw is not None:hw.close()
        backbone.release();fenet.release();fcm_edge.subprocess=original
        report['completed_frames']=len(rows)
        report['measured']={key:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)))
            for key in ('shared_ms','sw_codec_ms','hw_codec_ms','sw_board_sum_ms','hw_board_sum_ms','sw_bytes','hw_bytes','sw_relative_l2','hw_relative_l2','hw_vs_sw_relative_l2')
            for v in [[row[key] for row in rows]] if v}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('\n=== FCM SW/HW SAME INPUT SUMMARY ===',flush=True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);print('저장:',out,flush=True)


if __name__=='__main__':main()
