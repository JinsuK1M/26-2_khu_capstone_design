"""
보드가 만든 진짜 FCM 압축 파일(.bin)을 서버에서 풀고 탐지해 보는 확인용 코드
  1) 원본 YOLOX 전체 모델로 사진(.png) 탐지          (기준)
  2) 보드 payload(.bin) -> 서버 복원 -> 탐지        (보드 최종 경로: NPU FENet + combined4)
  결과 그림: runs/payload_check/ 에 왼쪽(원본) | 오른쪽(보드 payload) 로 저장

사용법 (yolov5 폴더에서):
  python yolox_payload_check.py --payload last_payload.bin --image last_frame.png --meta last_frame.json
"""
import argparse
import json
import os
import struct
import time

import cv2
import torch

from yolox_split import draw, load_model, preprocess, server_part, to_detections


def show(name, dets):
    print(f"\n[{name}] {len(dets)}개")
    for d in sorted(dets, key=lambda d: -d["conf"]):
        print(f"   {d['cls']:<14} {d['conf']:.3f}  box={d['box']}")


def match(ref, test, thr=0.5):
    """원본 결과와 같은 물체(같은 클래스, 박스 겹침 IoU>=0.5)를 몇 개 찾았는지"""
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
        return inter / (area(a) + area(b) - inter + 1e-9)
    used, hit = set(), 0
    for r in ref:
        for j, t in enumerate(test):
            if j not in used and t["cls"] == r["cls"] and iou(r["box"], t["box"]) >= thr:
                used.add(j)
                hit += 1
                break
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="last_payload.bin")
    ap.add_argument("--image", default="last_frame.png")
    ap.add_argument("--meta", default="last_frame.json")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    blob = open(args.payload, "rb").read()
    std, mean, h, w, n = struct.unpack_from("<ddHHI", blob)
    print("===== 보드 payload 확인 =====")
    print(f"파일 크기 {len(blob):,} bytes = 헤더 24 + HM {n:,}  ({'OK' if len(blob) == 24 + n else '길이 안 맞음!'})")
    print(f"타일 {h}x{w}, std {std:.4f}, mean {mean:.4f}")
    if os.path.exists(args.meta):
        m = json.load(open(args.meta))
        print(f"보드 설정: FENet={m.get('fenet_backend')}, HM={m.get('hm_profile')}, QP={m.get('qp')}, "
              f"보드 처리 {m.get('selected_process_ms', 0):.0f}ms")

    img = cv2.imread(args.image)
    r = 640 / max(img.shape[:2])  # 1280x960 -> 640x480 이면 0.5
    model = load_model()
    from fcm_plugin_yolox import load_fcm
    fcm = load_fcm(20)

    t0 = time.perf_counter()
    dark3 = fcm.decompress(blob)
    t1 = time.perf_counter()
    d_board = to_detections(server_part(model, {"dark3": dark3}), r, conf=args.conf)
    t2 = time.perf_counter()

    x, _ = preprocess(img)
    with torch.no_grad():
        d_ref = to_detections(model(x), r, conf=args.conf)

    print("\n===== 탐지 결과 =====")
    show("1. 원본 전체 모델 (사진 그대로)", d_ref)
    show("2. 보드 payload -> 서버 복원 -> 탐지", d_board)
    print(f"\n서버 시간: 복원 {(t1 - t0) * 1000:.0f}ms + 탐지 {(t2 - t1) * 1000:.0f}ms "
          f"(첫 실행이라 실제보다 느릴 수 있음)")
    print(f"원본이 찾은 {len(d_ref)}개 중 보드 경로도 찾은 것: {match(d_ref, d_board)}개 (같은 종류, 박스 50% 이상 겹침)")

    os.makedirs("runs/payload_check", exist_ok=True)
    left = draw(img.copy(), d_ref, f"original YOLOX: {len(d_ref)} obj")
    right = draw(img.copy(), d_board, f"board payload {len(blob) / 1024:.1f}KB -> server: {len(d_board)} obj")
    cv2.imwrite("runs/payload_check/compare.jpg", cv2.hconcat([left, right]))
    print("그림 저장: runs/payload_check/compare.jpg (왼쪽 원본 | 오른쪽 보드 payload)")


if __name__ == "__main__":
    main()
