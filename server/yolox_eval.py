"""
YOLOX-Darknet53 정확도 평가 (Server 파트)
  사진 -> [Edge] backbone 앞부분 -> 특징 -> [압축 -> 복원] -> [Server] 나머지 -> mAP
  채점은 yolov5 폴더의 val.py 를 그대로 빌려 쓴다 (그래서 yolov5 폴더에서 실행).
  전처리는 YOLOX 공식 방식(오른쪽·아래 114 패딩)으로 맞췄다. FCM 학습 조건과 같다.

분할 (--split):
  l37    (기본) Edge: stem~dark3, 보내는 것: dark3 1개 (256x80x80)  <- FCM 압축 지점
  dark345       Edge: backbone 전체, 보내는 것: dark3·dark4·dark5

비교하는 것:
  A. YOLOX 원본 (기준)
  B. 분할만 (fp32)           -> A 와 같아야 정상
  C. 분할 + uint8            -> 단순 압축 기준선
  D. 분할 + FCM (QP별)       -> fcm_plugin_yolox.py + fcm_bundle + HM 이 있어야 측정 (l37 전용)

사용법 (yolov5 폴더에서):
  python yolox_eval.py                                   # COCO128, l37, A B C D(QP20)
  python yolox_eval.py --qp 20,25                        # D 를 QP20, QP25 둘 다
  python yolox_eval.py --only AD                         # 원하는 것만
  python yolox_eval.py --data data/coco.yaml --limit 500 # COCO val2017 앞 500장 (FCM 은 느려서)
  python yolox_eval.py --only AD --hm default,combined4  # 학습 때 HM 설정 vs 보드 최종 설정(combined4)
"""
import argparse
import time

import cv2
import torch
import torch.nn as nn
import val as yolo_val
from torch.utils.data import DataLoader, Dataset
from utils.dataloaders import LoadImagesAndLabels, create_dataloader
from utils.general import check_dataset

from yolox_split import COCO_CLASSES, edge_part, load_model, preprocess, server_part


# ---------------------------------------------------------------- YOLOX 공식 전처리로 사진을 주는 데이터 묶음
class YoloxStyleSet(Dataset):
    def __init__(self, base, size, limit=0):
        n = limit or len(base.im_files)
        self.files, self.labels, self.size = base.im_files[:n], base.labels[:n], size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        h0, w0 = img.shape[:2]
        x, r = preprocess(img, self.size)                 # BGR 0~255, 오른쪽·아래 114 패딩
        im = x[0][[2, 1, 0]].round().clamp(0, 255).byte()  # val.py 형식(RGB uint8)으로
        lab = torch.from_numpy(self.labels[i]).float()    # [cls, x, y, w, h] (원본 기준 0~1)
        t = torch.zeros((len(lab), 6))
        if len(lab):
            t[:, 1] = lab[:, 0]
            t[:, 2] = lab[:, 1] * w0 * r / self.size
            t[:, 3] = lab[:, 2] * h0 * r / self.size
            t[:, 4] = lab[:, 3] * w0 * r / self.size
            t[:, 5] = lab[:, 4] * h0 * r / self.size
        return im, t, self.files[i], ((h0, w0), ((r, r), (0.0, 0.0)))


# ---------------------------------------------------------------- 압축 방법들: feats -> (feats, 장당 바이트)
def no_codec(feats):
    return feats, sum(f[0].numel() * 4 for f in feats.values())


def uint8_codec(feats):
    out, nbytes = {}, 0
    for k, f in feats.items():
        mn, mx = f.amin(dim=(1, 2, 3), keepdim=True), f.amax(dim=(1, 2, 3), keepdim=True)
        scale = (mx - mn).clamp(min=1e-8) / 255
        out[k] = torch.round((f - mn) / scale).clamp(0, 255) * scale + mn
        nbytes += f[0].numel()
    return out, nbytes


class Pipeline(nn.Module):
    def __init__(self, model, split, codec=None):
        super().__init__()
        self.model, self.split, self.codec = model, split, codec
        self.names = dict(enumerate(COCO_CLASSES))
        self.bytes_sum, self.images = 0.0, 0

    def forward(self, x, augment=False):
        x = x[:, [2, 1, 0]] * 255  # val.py 입력(RGB 0~1) -> YOLOX 입력(BGR 0~255)
        if self.codec is None:
            return self.model(x)
        feats, nbytes = self.codec(edge_part(self.model, x, self.split))
        self.bytes_sum += nbytes * x.shape[0]
        self.images += x.shape[0]
        return server_part(self.model, feats)  # (cx, cy, w, h, obj, cls) = yolov5 와 같은 형식


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolox_darknet.pth")
    ap.add_argument("--data", default="data/coco128.yaml")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="l37", choices=["l37", "dark345"])
    ap.add_argument("--qp", default="20", help="FCM QP, 여러 개는 쉼표로 (학습 범위 20, 25)")
    ap.add_argument("--hm", default="default", help="HM 설정: default, combined4 (여러 개는 쉼표로)")
    ap.add_argument("--only", default="ABCD", help="일부만 돌리기, 예: --only AD")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 몇 장만 (0 = 전부)")
    args = ap.parse_args()

    model = load_model(args.weights)
    data = check_dataset(args.data)
    base = create_dataloader(data["val"], args.img, args.batch, 32, pad=0.0, rect=False, workers=0, prefix="val: ")[1]
    dl = DataLoader(YoloxStyleSet(base, args.img, args.limit), batch_size=args.batch, shuffle=False,
                    collate_fn=LoadImagesAndLabels.collate_fn)

    runs = []
    if "A" in args.only:
        runs.append(("A. YOLOX 원본", None))
    if "B" in args.only:
        runs.append(("B. 분할만 (fp32)", no_codec))
    if "C" in args.only:
        runs.append(("C. 분할 + uint8", uint8_codec))
    if "D" in args.only:
        if args.split != "l37":
            print("[D 건너뜀] FCM 은 l37(dark3) 분할 전용이에요. --split l37 로 실행하세요.")
        else:
            try:
                from fcm_plugin_yolox import load_fcm
                for hm in args.hm.split(","):
                    for q in args.qp.split(","):
                        tag = "" if hm == "default" else f" {hm}"
                        runs.append((f"D. FCM QP{q}{tag}", load_fcm(int(q), hm)))
            except Exception as e:  # 번들·HM 이 없을 때
                print(f"[D 건너뜀] FCM 을 불러오지 못했어요: {e}")

    results = []
    for name, codec in runs:
        print(f"\n--- {name} ---")
        pipe = Pipeline(model, args.split, codec).eval()
        t0 = time.time()
        with torch.no_grad():
            (mp, mr, m50, m, *_), _, _ = yolo_val.run(data=data, model=pipe, dataloader=dl, imgsz=args.img,
                                                      plots=False, compute_loss=None)
        size = f"{pipe.bytes_sum / max(pipe.images, 1) / 1024:9.2f} KB" if pipe.images else "   (image)  "
        results.append((name, size, m50, m, time.time() - t0))

    n = len(dl.dataset)
    print(f"\n===== YOLOX-Darknet53 정확도 평가 (입력 {args.img}, 분할 {args.split}, {args.data}, {n}장) =====")
    print(f"{'방식':<26}{'전송량/장':>13}{'mAP50':>9}{'mAP50-95':>10}{'걸린 시간':>10}")
    for name, size, m50, m, sec in results:
        print(f"{name:<26}{size:>13}{m50:>9.3f}{m:>10.3f}{sec:>9.0f}s")


if __name__ == "__main__":
    main()
