"""
YOLOX-Darknet53 을 Edge / Server 로 나누는 공통 코드
  Edge  : 사진 -> Darknet53 backbone -> dark3(256x80x80), dark4(512x40x40), dark5(512x20x20)
  Server: dark3, dark4, dark5 -> FPN(neck) -> head -> 탐지 결과
다른 스크립트(yolox_check / yolox_server / yolox_client / yolox_eval)가 이 파일을 불러서 쓴다.
"""
import os
import sys

import cv2
import numpy as np
import torch

YOLOX_DIR = os.path.expanduser(os.environ.get("YOLOX_DIR", "~/YOLOX"))
sys.path.append(YOLOX_DIR)

from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

# 클래스 이름 (pycocotools 설치 없이 쓰려고 파일에서 직접 읽음)
_ns = {}
exec(open(os.path.join(YOLOX_DIR, "yolox/data/datasets/coco_classes.py")).read(), _ns)
COCO_CLASSES = _ns["COCO_CLASSES"]

FEATURES = ["dark3", "dark4", "dark5"]


def load_model(weights="yolox_darknet.pth"):
    exp = get_exp(os.path.join(YOLOX_DIR, "exps/default/yolov3.py"), None)
    model = exp.get_model()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.eval()


def preprocess(img_bgr, size=640):
    """YOLOX 공식 전처리: 비율 유지 리사이즈, 오른쪽/아래 114 채움, BGR 0~255 (정규화 없음)"""
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    r = min(size / img_bgr.shape[0], size / img_bgr.shape[1])
    h, w = int(img_bgr.shape[0] * r), int(img_bgr.shape[1] * r)
    padded[:h, :w] = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(padded.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    return x, r


@torch.no_grad()
def edge_part(model, x, split="dark345"):
    """[Edge] backbone 실행
    split='dark345': backbone 전체 -> {'dark3','dark4','dark5'}
    split='l37'    : stem -> dark2 -> dark3 까지만 -> {'dark3'}  (FCM 압축 지점)"""
    bb = model.backbone.backbone
    if split == "l37":
        return {"dark3": bb.dark3(bb.dark2(bb.stem(x)))}
    out = bb(x)
    return {k: out[k] for k in FEATURES}


@torch.no_grad()
def server_part(model, feats):
    """[Server] (dark3 -> dark4 -> dark5) -> FPN -> head  (YOLOFPN.forward 의 뒷부분과 동일)"""
    fpn = model.backbone
    if "dark4" not in feats:  # l37 분할: dark3 만 받았으면 dark4, dark5 를 서버에서 계산
        bb = fpn.backbone
        d4 = bb.dark4(feats["dark3"])
        feats = {"dark3": feats["dark3"], "dark4": d4, "dark5": bb.dark5(d4)}
    x2, x1, x0 = feats["dark3"], feats["dark4"], feats["dark5"]
    x1_in = torch.cat([fpn.upsample(fpn.out1_cbl(x0)), x1], 1)
    out_dark4 = fpn.out1(x1_in)
    x2_in = torch.cat([fpn.upsample(fpn.out2_cbl(out_dark4)), x2], 1)
    out_dark3 = fpn.out2(x2_in)
    return model.head((out_dark3, out_dark4, x0))  # [B, 8400, 85] = (cx, cy, w, h, obj, 80 class)


def to_detections(pred, r, conf=0.25, nms=0.45):
    """head 출력 -> [{'box':[x1,y1,x2,y2], 'conf':..., 'cls':...}] (원본 사진 좌표)"""
    out = postprocess(pred.clone(), 80, conf, nms, class_agnostic=False)[0]
    if out is None:
        return []
    dets = []
    for x1, y1, x2, y2, obj, cls_conf, c in out.tolist():
        dets.append({"box": [round(v / r, 1) for v in (x1, y1, x2, y2)],
                     "conf": round(obj * cls_conf, 3), "cls": COCO_CLASSES[int(c)]})
    return dets


def draw(img, dets, text=""):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{d["cls"]} {d["conf"]:.2f}', (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if text:
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img
