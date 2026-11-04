"""
가짜 Edge (Orange Pi 대신 노트북이 backbone 을 실행)
  사진 -> Darknet53 backbone -> dark3, dark4, dark5 -> 서버로 전송 -> 탐지 결과 받기

사용법:
  python yolox_client.py --fcm --check                    # dark3 를 FCM 으로 압축해서 전송 (QP20)
  python yolox_client.py --fcm --qp 25 --check            # QP25
  python yolox_client.py --check                          # 압축 없이 dark3·dark4·dark5 전송
  python yolox_client.py --dtype uint8                    # 숫자를 1바이트로 줄여서 전송
  python yolox_client.py --fcm --hm combined4 --check     # 보드 최종 HM 설정으로 압축
  python yolox_client.py --payload frame_0001.bin         # 보드가 만든 FCM bytes 파일을 그대로 전송
  python yolox_client.py --npz npu_features.npz           # 보드 feature 파일을 그대로 전송
  python yolox_client.py --source ~/datasets/coco128/images/train2017 --noshow
"""
import argparse
import csv
import glob
import json
import os
import socket
import struct
import time

import cv2
import numpy as np
import torch

from yolox_split import FEATURES, draw, edge_part, load_model, preprocess, to_detections


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed")
        buf += chunk
    return buf


def recv_msg(sock):
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    return recv_exact(sock, n)


def send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


def pack(name, f, dtype):
    a = f.numpy()
    t = {"name": name, "shape": list(a.shape), "dtype": dtype}
    if dtype == "fp32":
        p = a.astype(np.float32).tobytes()
    elif dtype == "fp16":
        p = a.astype(np.float16).tobytes()
    else:
        mn, mx = float(a.min()), float(a.max())
        scale = (mx - mn) / 255 if mx > mn else 1.0
        p = np.clip(np.round((a - mn) / scale), 0, 255).astype(np.uint8).tobytes()
        t.update(scale=scale, zero=mn)
    t["nbytes"] = len(p)
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--weights", default="yolox_darknet.pth")
    ap.add_argument("--source", default="data/images")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "uint8"])
    ap.add_argument("--npz", default="", help="보드 npu_features.npz 를 그대로 보내기")
    ap.add_argument("--check", action="store_true", help="원본 YOLOX 통째로 실행한 결과와 비교")
    ap.add_argument("--fcm", action="store_true", help="dark3 를 FCM 으로 압축해서 전송")
    ap.add_argument("--qp", type=int, default=20)
    ap.add_argument("--hm", default="default", choices=["default", "combined4"], help="HM 설정")
    ap.add_argument("--payload", nargs="*", default=[], help="보드가 저장한 FCM bytes 파일(.bin)들")
    ap.add_argument("--split", default="dark345", choices=["dark345", "l37"], help="FCM 없이 보낼 때 분할 위치")
    ap.add_argument("--noshow", action="store_true")
    args = ap.parse_args()

    model = load_model(args.weights)
    fcm = None
    if args.fcm:
        from fcm_plugin_yolox import load_fcm
        fcm = load_fcm(args.qp, args.hm)
    sock = socket.create_connection((args.host, args.port))

    if args.payload:  # 보드에서 압축한 bytes 를 그대로 서버로 (서버 복원·탐지 확인용)
        for i, p in enumerate(args.payload):
            blob = open(os.path.expanduser(p), "rb").read()
            t0 = time.perf_counter()
            send_msg(sock, json.dumps({"fcm": True, "qp": args.qp, "ratio": 1.0, "frame_id": i}).encode())
            send_msg(sock, blob)
            resp = json.loads(recv_msg(sock))
            print(f"[payload] {os.path.basename(p)} {len(blob)/1024:.1f}KB | frame_id {resp.get('frame_id')} | "
                  f"server {resp['server_ms']:.0f}ms | E2E {(time.perf_counter()-t0)*1000:.0f}ms | "
                  f"{len(resp['dets'])} objects (box 좌표는 640 입력 기준)")
            for d in resp["dets"][:10]:
                print(f"   {d['cls']:<14} {d['conf']:.2f}  box={d['box']}")
        return

    if args.npz:  # Orange Pi 에서 뽑은 feature 를 그대로 전송
        z = np.load(args.npz)
        tensors, payload = [], b""
        for k in FEATURES:
            f = torch.from_numpy(z[k].astype(np.float32))
            f = f if f.ndim == 4 else f[None]
            t, p = pack(k, f, args.dtype)
            tensors.append(t)
            payload += p
        t0 = time.perf_counter()
        send_msg(sock, json.dumps({"tensors": tensors, "ratio": 1.0}).encode())
        send_msg(sock, payload)
        resp = json.loads(recv_msg(sock))
        print(f"[npz] sent {len(payload)/1024:.0f}KB | E2E {(time.perf_counter()-t0)*1000:.0f}ms | "
              f"{len(resp['dets'])} objects (box 좌표는 640 입력 기준)")
        for d in resp["dets"]:
            print(f"   {d['cls']:<14} {d['conf']:.2f}  box={d['box']}")
        return

    mode = f"fcm_qp{args.qp}" + ("" if args.hm == "default" else f"_{args.hm}") if args.fcm else f"{args.split}_{args.dtype}"
    out_dir = f"runs/yolox_{mode}"
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(p for e in ("*.jpg", "*.jpeg", "*.png")
                   for p in glob.glob(os.path.join(os.path.expanduser(args.source), e)))
    rows = []
    for fid, p in enumerate(paths):
        img = cv2.imread(p)
        t0 = time.perf_counter()
        x, r = preprocess(img, args.img)
        if fcm:  # dark3 -> FCM bytes
            payload = fcm.compress(edge_part(model, x, "l37")["dark3"])
            hdr = {"fcm": True, "qp": args.qp, "ratio": r, "frame_id": fid}
        else:
            feats = edge_part(model, x, args.split)
            tensors, payload = [], b""
            for k, f in feats.items():
                t, b = pack(k, f, args.dtype)
                tensors.append(t)
                payload += b
            hdr = {"tensors": tensors, "ratio": r, "frame_id": fid}
        t1 = time.perf_counter()
        send_msg(sock, json.dumps(hdr).encode())
        send_msg(sock, payload)
        resp = json.loads(recv_msg(sock))
        t2 = time.perf_counter()
        assert resp.get("frame_id") == fid, f"보낸 frame_id {fid} 와 받은 결과 {resp.get('frame_id')} 가 달라요"
        row = {"frame_id": fid, "image": os.path.basename(p), "sent_KB": len(payload) / 1024, "edge_ms": (t1 - t0) * 1000,
               "server_ms": resp["server_ms"], "net_ms": (t2 - t1) * 1000 - resp["server_ms"],
               "e2e_ms": (t2 - t0) * 1000, "num_dets": len(resp["dets"])}
        msg = (f"{row['image']}: sent {row['sent_KB']:.1f}KB | edge {row['edge_ms']:.0f}ms | "
               f"server {row['server_ms']:.0f}ms | E2E {row['e2e_ms']:.0f}ms | {row['num_dets']} obj")
        if args.check:
            with torch.no_grad():
                ref = to_detections(model(x), r)
            row["ref_dets"] = len(ref)
            msg += f" | full-model {len(ref)} obj"
        rows.append(row)
        print(msg)
        vis = draw(img.copy(), resp["dets"], f"YOLOX {mode} {row['sent_KB']:.1f}KB")
        cv2.imwrite(os.path.join(out_dir, row["image"]), vis)
        if not args.noshow:
            cv2.imshow("YOLOX split (q: quit)", vis)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
    sock.close()
    cv2.destroyAllWindows()
    if rows:
        with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        avg = lambda k: sum(r[k] for r in rows) / len(rows)
        print(f"\n===== Average (YOLOX-Darknet53, input {args.img}, {mode}) =====")
        print(f"images      : {len(rows)}")
        print(f"sent size   : {avg('sent_KB'):.1f} KB")
        print(f"edge        : {avg('edge_ms'):.0f} ms  (backbone + 압축)")
        print(f"server      : {avg('server_ms'):.0f} ms")
        print(f"network     : {avg('net_ms'):.1f} ms")
        print(f"End-to-End  : {avg('e2e_ms'):.0f} ms")
        if args.check:
            print(f"same #objects as full model: {sum(r['ref_dets'] == r['num_dets'] for r in rows)}/{len(rows)}")
        print(f"log saved   : {out_dir}/log.csv")


if __name__ == "__main__":
    main()
