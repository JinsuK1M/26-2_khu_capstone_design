"""
보드(NPU)에서 뽑은 dark3 를 받아 서버에서 탐지까지 해 보는 확인용 코드
  1) 원본 YOLOX 전체 모델           (기준)
  2) NPU dark3 -> 서버              (압축 없이)
  3) NPU dark3 -> FCM -> 서버       (QP20, 학습 설정 / combined4)

사용법 (yolov5 폴더에서):
  python yolox_npu_check.py --npu zidane_dark3_npu.npy --image data/images/zidane.jpg
  python yolox_npu_check.py --npu zidane_dark3_npu.npy --pytorch zidane_dark3_pytorch.npy --input input_bgr_uint8_nhwc.npy
"""
import argparse
import os

import cv2
import numpy as np
import torch

from yolox_split import draw, edge_part, load_model, preprocess, server_part, to_detections


def show(name, dets):
    print(f"\n[{name}] {len(dets)}개")
    for d in dets:
        print(f"   {d['cls']:<12} {d['conf']:.3f}  box={d['box']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", required=True, help="보드 NPU dark3 (.npy, 1x256x80x80)")
    ap.add_argument("--image", default="data/images/zidane.jpg")
    ap.add_argument("--pytorch", default="", help="(선택) 보드 쪽 PyTorch dark3 .npy")
    ap.add_argument("--input", default="", help="(선택) 보드 전처리 입력 .npy (1x640x640x3)")
    ap.add_argument("--qp", type=int, default=20)
    ap.add_argument("--no-fcm", action="store_true")
    args = ap.parse_args()

    model = load_model()
    img = cv2.imread(args.image)
    x, r = preprocess(img)

    print("===== 입력·특징 맞는지 확인 =====")
    if args.input:
        xin = torch.from_numpy(np.load(args.input)[0].transpose(2, 0, 1).copy()).float()[None]
        print(f"보드 전처리 입력 vs 내 전처리 최대 차이: {float((xin - x).abs().max()):.4g}  (0 이면 같음)")
    mine = edge_part(model, x, "l37")["dark3"]
    if args.pytorch:
        pt = torch.from_numpy(np.load(args.pytorch))
        print(f"보드 쪽 PyTorch dark3 vs 내 dark3 최대 차이: {float((pt - mine).abs().max()):.3g}")
    npu = torch.from_numpy(np.load(args.npu).astype(np.float32))
    a, b = npu.flatten().double(), mine.flatten().double()
    print(f"NPU dark3 vs 내 dark3: 코사인 {float(a @ b / (a.norm() * b.norm())):.7f}, "
          f"상대 L2 {float((a - b).norm() / b.norm()) * 100:.3f}%")

    print("\n===== 탐지 결과 비교 =====")
    with torch.no_grad():
        ref = to_detections(model(x), r)
    show("1. 원본 전체 모델", ref)
    d_npu = to_detections(server_part(model, {"dark3": npu}), r)
    show("2. NPU dark3 -> 서버", d_npu)
    os.makedirs("runs/npu_check", exist_ok=True)
    cv2.imwrite("runs/npu_check/npu_server.jpg", draw(img.copy(), d_npu, "NPU dark3 -> server"))

    if not args.no_fcm:
        from fcm_plugin_yolox import load_fcm
        for hm in ("default", "combined4"):
            f = load_fcm(args.qp, hm)
            blob = f.compress(npu)
            d = to_detections(server_part(model, {"dark3": f.decompress(blob)}), r)
            show(f"3. NPU dark3 -> FCM QP{args.qp} {hm} ({len(blob) / 1024:.1f}KB) -> 서버", d)
            cv2.imwrite(f"runs/npu_check/npu_fcm_{hm}.jpg", draw(img.copy(), d, f"NPU+FCM {hm}"))
    print("\n그림 저장: runs/npu_check/")


if __name__ == "__main__":
    main()
