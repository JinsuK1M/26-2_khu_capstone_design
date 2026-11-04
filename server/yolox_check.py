"""
YOLOX-Darknet53 분할 검증
  1) 사진 폴더:  원본 YOLOX 통째로 vs (Edge backbone -> Server FPN+head) 결과 비교
  2) --npz   :  Orange Pi 에서 뽑은 feature(npu_features.npz)를 서버 부분에 넣어 탐지

사용법 (yolov5 폴더 또는 아무 데서나):
  python yolox_check.py                                   # data/images 의 사진들로 검증
  python yolox_check.py --npz npu_features.npz --image 원본사진.jpg
결과 이미지: runs/yolox_check/
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from yolox_split import FEATURES, draw, edge_part, load_model, preprocess, server_part, to_detections


def load_npz(path):
    z = np.load(path)
    print(f"[npz] {path} 안의 항목:")
    for k in z.files:
        print(f"   {k:<20} shape={z[k].shape} dtype={z[k].dtype}")
    feats = {}
    for name, shape in zip(FEATURES, [(256, 80, 80), (512, 40, 40), (512, 20, 20)]):
        key = name if name in z.files else next((k for k in z.files if tuple(z[k].shape[-3:]) == shape), None)
        if key is None:
            raise SystemExit(f"[npz] {name} {shape} 에 해당하는 항목을 못 찾았어요. 위 목록을 보내 주세요.")
        a = z[key].astype(np.float32)
        feats[name] = torch.from_numpy(a if a.ndim == 4 else a[None])
        print(f"   -> {name} = '{key}'")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolox_darknet.pth")
    ap.add_argument("--source", default="data/images")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--npz", default="")
    ap.add_argument("--image", default="", help="npz 를 뽑은 원본 사진 (있으면 박스를 그려서 비교)")
    args = ap.parse_args()

    model = load_model(args.weights)
    out_dir = "runs/yolox_check"
    os.makedirs(out_dir, exist_ok=True)

    if args.npz:  # ---------------- 보드 feature 로 서버 부분 실행
        feats = load_npz(args.npz)
        pred = server_part(model, feats)
        r = 1.0
        if args.image:
            img = cv2.imread(args.image)
            x, r = preprocess(img, args.img)
            ref = to_detections(model(x), r)
        dets = to_detections(pred, r)
        print(f"\n[서버] 보드 feature 로 탐지한 물체 {len(dets)}개")
        for d in dets:
            print(f"   {d['cls']:<14} {d['conf']:.2f}  box={d['box']}")
        if args.image:
            print(f"[비교] 원본 사진을 노트북 YOLOX 에 통째로 넣으면 {len(ref)}개")
            cv2.imwrite(os.path.join(out_dir, "npz_result.jpg"), draw(img.copy(), dets, "server(npz feature)"))
            print(f"결과 이미지: {out_dir}/npz_result.jpg")
        return

    paths = sorted(p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.source, e)))
    for p in paths:  # ---------------- 노트북 안에서 분할 검증
        img = cv2.imread(p)
        x, r = preprocess(img, args.img)
        with torch.no_grad():
            full = model(x)
            feats = edge_part(model, x)
            split = server_part(model, feats)
        diff = float((full - split).abs().max())
        d_full, d_split = to_detections(full, r), to_detections(split, r)
        sizes = " + ".join(f"{k}{tuple(v.shape[1:])}" for k, v in feats.items())
        kb = sum(v.numel() for v in feats.values()) * 4 / 1024
        print(f"{os.path.basename(p)}: {sizes} = {kb:.0f}KB(fp32) | 원본 {len(d_full)}개, 분할 {len(d_split)}개 "
              f"| 출력 최대 차이 {diff:.6f}")
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), draw(img.copy(), d_split, "YOLOX split"))
    print(f"결과 이미지: {out_dir}/")


if __name__ == "__main__":
    main()
