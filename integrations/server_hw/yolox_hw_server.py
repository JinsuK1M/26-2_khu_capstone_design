"""기존 yolox_server.py와 같은 폴더에서 실행. fcm_hw_v1 모드만 추가하는 서버 진입점."""
import math
import os
import struct
import sys
import time
import numpy as np
import torch
import yolox_server as base
from yolox_split import server_part, to_detections

MAGIC=b'FHW1'
HEADER='<ddHHI'


def unpack(blob):
    size=4+struct.calcsize(HEADER)
    if len(blob)<size or blob[:4]!=MAGIC:raise ValueError('HW FCM magic/header 오류')
    std,mean,h,w,n=struct.unpack_from(HEADER,blob,4)
    if (h,w)!=(320,640) or n!=len(blob)-size or n<=0:
        raise ValueError('HW FCM 크기/길이 오류')
    if not math.isfinite(std) or std<=0 or not math.isfinite(mean):
        raise ValueError('HW FCM std/mean 오류')
    return std,mean,blob[size:]


class HardwareServer(base.Server):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.hw_decoder=None
    @torch.no_grad()
    def handle(self,hdr,payload,state=None):
        if hdr.get('mode')!='fcm_hw_v1':return super().handle(hdr,payload,state)
        start=time.perf_counter()
        std,mean,bits=unpack(payload)
        if self.hw_decoder is None:
            sys.path.insert(0,os.path.expanduser(os.environ.get('FCM_BUNDLE_DIR','~/fcm_bundle')))
            import fcm_server
            self.hw_decoder=fcm_server.load_decoder()
        import av
        from fcm_common import dequantize_frame
        tick=time.perf_counter()
        # HW all-I 스트림: 매 payload가 독립 복원 가능해야 합니다.
        ctx=av.CodecContext.create('hevc','r');ctx.thread_count=1
        frames=ctx.decode(av.Packet(bits))+ctx.decode(None)
        if len(frames)!=1:raise ValueError('HW FCM payload는 정확히 한 프레임이어야 합니다')
        frame=frames[0]
        if frame.width!=640 or frame.height!=320 or frame.format.name not in ('yuv420p','yuvj420p'):
            raise ValueError('HW FCM은 640x320 8bit planar 420이어야 합니다')
        plane=frame.planes[0]
        # RGB/gray 색 변환 없이 Y 평면을 직접 읽고 stride padding을 제거합니다.
        y8=np.frombuffer(plane,dtype=np.uint8).reshape(plane.height,plane.line_size)[:320,:640].copy()
        decoded=time.perf_counter()
        q10=y8.astype(np.float64)*(1023/255)
        latent=dequantize_frame(q10,128,40,40,std,mean)
        dark3=self.hw_decoder(latent)
        if tuple(dark3.shape)!=(1,256,80,80) or not torch.isfinite(dark3).all():
            raise ValueError('HW FCM 복원 dark3 오류')
        restored=time.perf_counter()
        pred=server_part(self.model,{'dark3':dark3});inferred=time.perf_counter()
        dets=to_detections(pred,hdr.get('ratio',.5),conf=self.conf)
        end=time.perf_counter()
        t=dict(hevc_decode_ms=(decoded-tick)*1000,drnet_ms=(restored-decoded)*1000,
               decode_ms=(restored-tick)*1000,infer_ms=(inferred-restored)*1000,post_ms=(end-inferred)*1000)
        return dict(frame_id=hdr.get('frame_id'),mode='fcm_hw_v1',dets=dets,t=t,
                    server_ms=(end-start)*1000,payload_bytes=len(payload),
                    unpack_ms=t['decode_ms'],tail_ms=t['infer_ms'])


if __name__=='__main__':
    base.Server=HardwareServer
    base.main()
