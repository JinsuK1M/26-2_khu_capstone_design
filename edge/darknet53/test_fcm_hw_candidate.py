"""FCM HW 후보: q10→q8→NV12 Y→HW all-I HEVC→Y 복원→q10 스케일→DRNet.
기존 FCM payload/서버와 호환되지 않는 별도 실험입니다. 원본 모듈은 수정하지 않습니다.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
try:
    import gi
except ModuleNotFoundError:
    sys.path.append('/usr/lib/python3/dist-packages')
    import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


def metrics(a,b):
    a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel()
    return dict(mse=float(np.mean((a-b)**2)),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20)),
                cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-20)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--qp',type=int,default=20)
    args=parser.parse_args()
    if not 0<=args.qp<=51: parser.error('QP 범위는 0..51')
    bundle=Path.home()/'yolox_project/fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_edge, fcm_server
    from fcm_common import quantize_feature,dequantize_frame
    torch.set_num_threads(4)
    x=np.load(args.input.expanduser(),allow_pickle=False)
    if x.shape!=(1,256,80,80) or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('유한 float32 dark3 (1,256,80,80) 필요')
    out=Path(__file__).resolve().parent/'fcm_hw_candidates'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    enc=fcm_edge.load_encoder(); dec=fcm_server.load_decoder()
    with torch.inference_mode(): y=enc(torch.from_numpy(x))
    q10=quantize_feature(y)
    h,w=q10.shape
    if (h,w)!=(320,640): raise ValueError('타일 크기 확인 필요')
    q8=np.rint(q10.astype(np.float64)*255/1023).clip(0,255).astype(np.uint8)
    nv12=q8.tobytes()+np.full((h//2,w),128,dtype=np.uint8).tobytes()
    np.save(out/'q10.npy',q10);np.save(out/'q8.npy',q8)
    print('SW HM 기준 압축·복원 중...',flush=True)
    tick=time.perf_counter(); sw_bits=fcm_edge._hm_encode(q10,args.qp)
    sw_ms=(time.perf_counter()-tick)*1000
    sw_q=fcm_server._hm_decode(sw_bits,h,w)
    (out/'sw_hm.bin').write_bytes(sw_bits)
    Gst.init(None)
    pipeline=Gst.parse_launch(
        f'appsrc name=src is-live=true format=time block=true caps=video/x-raw,format=NV12,width={w},height={h},framerate=1/1 '
        f'! mpph265enc rc-mode=fixqp qp-init={args.qp} qp-min={args.qp} qp-max={args.qp} qp-min-i={args.qp} qp-max-i={args.qp} '
        'gop=1 header-mode=each-idr max-pending=1 '
        '! appsink name=sink sync=false async=false max-buffers=1 drop=false')
    src=pipeline.get_by_name('src');sink=pipeline.get_by_name('sink')
    samples=[];bits=None
    try:
        if pipeline.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('HW 파이프라인 시작 실패')
        for i in range(8):
            tick=time.perf_counter()
            buf=Gst.Buffer.new_allocate(None,len(nv12),None);buf.fill(0,nv12)
            buf.pts=i*Gst.SECOND;buf.duration=Gst.SECOND
            if src.emit('push-buffer',buf)!=Gst.FlowReturn.OK: raise RuntimeError('입력 실패')
            sample=sink.emit('try-pull-sample',10*Gst.SECOND)
            if sample is None:
                msg=pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                raise RuntimeError(str(msg.parse_error()) if msg else 'HW 출력 10초 시간 초과')
            output=sample.get_buffer()
            if output.pts!=buf.pts: raise RuntimeError('PTS 불일치')
            bits=output.extract_dup(0,output.get_size())
            duration=(time.perf_counter()-tick)*1000
            if i>=3:samples.append(dict(ms=duration,bytes=len(bits)))
        src.emit('end-of-stream')
        msg=pipeline.get_bus().timed_pop_filtered(10*Gst.SECOND,Gst.MessageType.EOS|Gst.MessageType.ERROR)
        if msg is None or msg.type==Gst.MessageType.ERROR: raise RuntimeError('EOS 종료 실패')
    finally:
        pipeline.set_state(Gst.State.NULL)
    (out/'hw.hevc').write_bytes(bits)
    # Y를 직접 추출: 회색조 색 변환/limited→full 변환을 추가하지 않습니다.
    command=['ffmpeg','-v','error','-i',str(out/'hw.hevc'),'-vf','extractplanes=y',
             '-frames:v','1','-pix_fmt','gray','-f','rawvideo','-y',str(out/'hw_y.raw')]
    subprocess.run(command,check=True)
    raw=np.fromfile(out/'hw_y.raw',dtype=np.uint8)
    if raw.size!=h*w: raise RuntimeError('HW 복원 Y 크기 불일치')
    hw8=raw.reshape(h,w)
    hw_q=hw8.astype(np.float64)*(1023/255)
    std=float(y.std());mean=float(y.mean())
    with torch.inference_mode():
        def reconstruct(q):
            return dec(dequantize_frame(q,128,40,40,std,mean)).numpy()
        sw_recon=reconstruct(sw_q)
        hw_recon=reconstruct(hw_q)
        quant_only=reconstruct(q8.astype(np.float64)*(1023/255))
    np.save(out/'hw_dark3.npy',hw_recon);np.save(out/'sw_dark3.npy',sw_recon)
    report=dict(state='completed',source=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        qp=args.qp,torch=torch.__version__,gst=Gst.version_string(),
        sw=dict(encode_ms=sw_ms,bytes=len(sw_bits),settings='bundle original HM; combined4 미적용'),
        hw=dict(encode_ms_median=float(np.median([s['ms'] for s in samples])),samples=samples,
                settings='8bit NV12, neutral UV, all-I, fixed QP, 3회 워밍업/5회 측정'),
        q8_rounding_vs_q10=metrics(q8.astype(float)*1023/255,q10),
        hw_y_vs_input_q8=metrics(hw8,q8),
        quantization_only_dark3_vs_original=metrics(quant_only,x),
        sw_dark3_vs_original=metrics(sw_recon,x),hw_dark3_vs_original=metrics(hw_recon,x),
        hw_dark3_vs_sw=metrics(hw_recon,sw_recon),
        note='별도 비호환 후보. HW는 8bit 4:2:0, HM은 10bit 4:0:0. 동일 QP는 동일 품질이 아님. HW 시간은 이미 변환한 NV12 버퍼 복사+인코딩; FENet/10→8변환/초기화/전송/복원 제외. SW는 프로세스·I/O 포함 단일 호출. mAP 아님.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('저장:',out,flush=True)


if __name__=='__main__': main()
