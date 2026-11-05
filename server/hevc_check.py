"""
보드에서 만든 HEVC 파일을 서버(Mac)에서 디코딩해 확인하는 코드
  - 프레임 수, 크기, 프레임 종류(I/P/B, PyAV 가 있을 때), 프레임별 크기
  - 모든 프레임 PNG 저장 + 60장을 한 장에 모은 그림
  - (선택) 몇 프레임에 YOLOX 탐지

사용법 (yolov5 폴더에서):
  python hevc_check.py test_gstreamer.hevc
  python hevc_check.py test_gstreamer.hevc --detect 0,20,28,49 --out hevc_frames
결과: <out>/frame_000.png ..., <out>/all_frames.jpg, <out>/report.json
"""
import argparse
import hashlib
import json
import os

import cv2
import numpy as np


def decode_frames(path):
    """(BGR 프레임 목록, 프레임 정보 목록). PyAV 가 있으면 프레임 종류·패킷 크기도 기록"""
    try:
        import av
        frames, info = [], []
        with av.open(path) as c:
            for pkt in c.demux(video=0):
                for fr in pkt.decode():
                    frames.append(fr.to_ndarray(format="bgr24"))
                    pt = getattr(fr.pict_type, "name", None) or {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), str(fr.pict_type))
                    info.append({"pict_type": pt, "key_frame": bool(fr.key_frame),
                                 "packet_bytes": pkt.size})
        return frames, info, f"pyav {av.__version__}"
    except ImportError:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        return frames, [{} for _ in frames], f"opencv {cv2.__version__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hevc")
    ap.add_argument("--out", default="hevc_frames")
    ap.add_argument("--detect", default="", help="탐지해 볼 프레임 번호, 예: 0,20,28")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = open(args.hevc, "rb").read()
    frames, info, decoder = decode_frames(args.hevc)
    if not frames:
        raise SystemExit("디코딩된 프레임이 없어요. 'pip install av' 후 다시 실행해 보세요.")

    thumbs, per_frame = [], []
    for i, (f, inf) in enumerate(zip(frames, info)):
        cv2.imwrite(os.path.join(args.out, f"frame_{i:03d}.png"), f)
        t = cv2.resize(f, (256, 192))
        cv2.putText(t, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        thumbs.append(t)
        per_frame.append({"index": i, "shape": list(f.shape), "mean_bgr": [round(float(v), 1) for v in f.reshape(-1, 3).mean(0)],
                          **inf})
    blank = np.zeros_like(thumbs[0])
    rows = [cv2.hconcat((thumbs[r:r + 10] + [blank] * 10)[:10]) for r in range(0, len(thumbs), 10)]
    cv2.imwrite(os.path.join(args.out, "all_frames.jpg"), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

    diffs = [float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean()) for i in range(1, len(frames))]
    types = [p.get("pict_type") for p in per_frame if p.get("pict_type")]
    report = {
        "file": os.path.basename(args.hevc), "file_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "decoder": decoder, "frames": len(frames), "shape": list(frames[0].shape),
        "all_same_shape": all(f.shape == frames[0].shape for f in frames),
        "avg_bytes_per_frame": len(data) / len(frames),
        "pict_types": {t: types.count(t) for t in sorted(set(types))} if types else "PyAV 없음",
        "key_frames": [p["index"] for p in per_frame if p.get("key_frame")],
        "consecutive_mean_abs_diff": {"min": min(diffs), "median": float(np.median(diffs)), "max": max(diffs)} if diffs else None,
        "per_frame": per_frame,
    }

    if args.detect:
        import torch
        from yolox_split import load_model, preprocess, to_detections
        model = load_model()
        report["detections"] = {}
        for i in [int(x) for x in args.detect.split(",")]:
            x, r = preprocess(frames[i])
            with torch.no_grad():
                d = to_detections(model(x), r)
            report["detections"][i] = d
            print(f"[탐지] frame {i}: " + ", ".join(f"{t['cls']} {t['conf']:.2f}" for t in d[:6]))

    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"디코더 {decoder} | 프레임 {report['frames']}장 | 크기 {report['shape']} | 모두 같은 크기 {report['all_same_shape']}")
    print(f"프레임 종류 {report['pict_types']} | key frame {report['key_frames']} | 평균 {report['avg_bytes_per_frame']/1024:.1f}KB/프레임")
    print(f"저장: {args.out}/frame_*.png, all_frames.jpg, report.json")


if __name__ == "__main__":
    main()
