"""
FCM 팀 번들(fcm_bundle)을 서버 코드에 연결하는 파일
  dark3 (1,256,80,80) -> [Edge] FENet -> 10비트 정수화 -> HM 인코딩 -> bytes
                      -> [Server] HM 디코딩 -> DRNet -> 복원 dark3 (1,256,80,80)

HM 설정 (hm=...):
  default   : 학습할 때 쓴 원래 설정 (encoder_intra_main_rext.cfg 그대로)
  combined4 : Edge 보드 최종 설정 (docs/edge_server_contract.md)
              RDOQ·RDOQTS·TransformSkip·TransformSkipFast 끔, QuadtreeTUMaxDepthIntra=1, SAO=0
  -> 디코더(서버)는 설정과 상관없이 똑같이 풀 수 있다.

필요한 것:
  FCM_BUNDLE_DIR : fcm_bundle 폴더 위치 (기본 ~/fcm_bundle)
  FCM_HM_DIR     : HM 폴더 (bin/TAppEncoderStatic, bin/TAppDecoderStatic). 기본은 fcm_bundle/hm
"""
import os
import struct
import subprocess
import sys
import tempfile

BUNDLE_DIR = os.path.expanduser(os.environ.get("FCM_BUNDLE_DIR", "~/fcm_bundle"))
sys.path.insert(0, BUNDLE_DIR)

import fcm_edge  # noqa: E402
import fcm_server  # noqa: E402

HM_PRESETS = {
    "default": [],
    "combined4": ["--RDOQ=0", "--RDOQTS=0", "--TransformSkip=0", "--TransformSkipFast=0",
                  "--QuadtreeTUMaxDepthIntra=1", "--SAO=0"],
}


def _hm_encode(frame, qp, extra):
    """fcm_edge._hm_encode 와 같고, HM 옵션(extra)만 뒤에 덧붙인다 (번들 파일은 수정하지 않음)"""
    h, w = frame.shape
    hm = fcm_edge.HM_DIR
    with tempfile.TemporaryDirectory() as d:
        yuv, binf = os.path.join(d, "in.yuv"), os.path.join(d, "out.bin")
        frame.astype("<u2").tofile(yuv)
        cmd = [os.path.join(hm, "bin", "TAppEncoderStatic"),
               "-c", os.path.join(hm, "cfg", "encoder_intra_main_rext.cfg"),
               "--Level=6.2", "--InputBitDepth=10", "--InternalBitDepth=10",
               "--InputChromaFormat=400", "--ConformanceWindowMode=1",
               "-i", yuv, "-f", "1", "-fr", "1", "-wdt", str(w), "-hgt", str(h),
               "-q", str(qp), "-b", binf] + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("HM 인코더 실패:\n" + r.stdout[-500:] + r.stderr[-500:])
        return open(binf, "rb").read()


class FCM:
    """yolox_eval / yolox_client / yolox_server 에서 쓰는 FCM 묶음"""

    def __init__(self, qp=20, hm="default"):
        if hm not in HM_PRESETS:
            raise ValueError(f"hm 은 {list(HM_PRESETS)} 중 하나여야 해요: {hm}")
        self.qp, self.hm = qp, hm
        self.enc = fcm_edge.load_encoder()
        self.dec = fcm_server.load_decoder()

    def compress(self, dark3):  # [Edge] (1,256,80,80) -> bytes
        import torch
        with torch.no_grad():
            y = self.enc(dark3)
        frame = fcm_edge.quantize_feature(y)
        bits = _hm_encode(frame, self.qp, HM_PRESETS[self.hm])
        h, w = frame.shape
        return struct.pack(fcm_edge.HDR, float(y.std()), float(y.mean()), h, w, len(bits)) + bits

    def decompress(self, blob):  # [Server] bytes -> (1,256,80,80)
        return fcm_server.decompress(self.dec, blob)

    def __call__(self, feats):
        """평가용: {'dark3': (B,256,80,80)} -> 압축·복원한 {'dark3': ...}, 장당 전송 바이트 수"""
        import torch
        outs, nbytes = [], 0
        for i in range(feats["dark3"].shape[0]):
            blob = self.compress(feats["dark3"][i:i + 1].contiguous())
            outs.append(self.decompress(blob))
            nbytes += len(blob)
        return {"dark3": torch.cat(outs)}, nbytes / feats["dark3"].shape[0]


def load_fcm(qp=20, hm="default"):
    return FCM(qp, hm)
