"""
비교군 A: 보드에서 PNG -> JPEG 인코딩 -> 서버 전송 -> 서버가 YOLOX 전체로 탐지 -> 결과 수신
(보드에서 실행. 모델 추론 없음. 필요한 패키지: python3, numpy, opencv-python)

manifest.json 순서대로 PNG 를 한 장씩 읽어 JPEG 로 바꿔 보내고, 결과를 받은 뒤 다음 장을 보낸다 (건너뛰기 없음).
워밍업 요청은 측정 로그에서 분리한다.

측정 경계 (B 경로와 같게 맞추기 위한 기준)
  png_read_ms : PNG 파일 읽기 + 디코딩       -> 따로 기록, total 에서 제외
  encode_ms   : JPEG 인코딩 시작 ~ 끝         (보드)
  rtt_ms      : 요청 전송 시작 ~ 결과 수신 끝   (보드)
  total_ms    : JPEG 인코딩 시작 ~ 결과 수신 끝 = encode_ms + rtt_ms   (보드)
  server_*    : 서버가 잰 시간 (payload 다 받은 뒤 ~ 결과 만들기 전)
  net_ms      : rtt_ms - server_ms   (전송 + 응답 + 직렬화, 두 시계를 섞지 않음)

사용법
  python3 jpeg_send.py <manifest.json> --host 100.90.83.29
  python3 jpeg_send.py <manifest.json> --host 100.90.83.29 --quality 80 --warmup 5 --out runs/jpeg_q80
결과: <out>/frames.csv (프레임별), <out>/frames.json (탐지 결과 포함 전체), <out>/summary.json
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import statistics
import struct
import time

import cv2
import numpy as np


# ---------------------------------------------------------------- 통신 (서버 규격: docs/server_protocol.md)
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("server closed")
        buf += chunk
    return bytes(buf)


def request(sock, header, payload):
    """헤더와 payload 를 보내고 JSON 결과를 받는다. (결과, 왕복 ms)"""
    h = json.dumps(header).encode()
    msg = struct.pack(">I", len(h)) + h + struct.pack(">I", len(payload)) + payload
    t0 = time.perf_counter()
    sock.sendall(msg)
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    res = json.loads(_recv_exact(sock, n))
    return res, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- manifest
def load_manifest(path):
    """manifest.json 에서 (파일 경로, 기록된 sha256) 목록을 순서대로 꺼낸다. 여러 형식을 허용."""
    m = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    items = m
    if isinstance(m, dict):
        items = None
        for k in ("frames", "images", "files", "items", "samples", "entries"):
            if isinstance(m.get(k), list):
                items = m[k]
                break
        if items is None:
            items = next((v for v in m.values() if isinstance(v, list)), None)
    if not items:
        raise ValueError("manifest 에서 이미지 목록을 찾지 못했어요")
    out = []
    for k, e in enumerate(items):
        if isinstance(e, str):
            name, sha, order = e, None, k
        else:
            name = next((e[x] for x in ("file", "filename", "name", "path", "image", "png") if x in e), None)
            sha = e.get("sha256") or e.get("sha")
            order = e.get("index", e.get("order", e.get("frame_id", k)))
        if name is None:
            raise ValueError(f"manifest 항목 {k} 에 파일 이름이 없어요: {e}")
        p = name if os.path.isabs(name) else os.path.join(base, name)
        if not os.path.exists(p):
            p = os.path.join(base, os.path.basename(name))
        out.append((order, p, sha))
    if all(isinstance(o, (int, float)) for o, _, _ in out):
        out.sort(key=lambda x: x[0])
    return [(p, sha) for _, p, sha in out]


def read_png(path):
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"PNG 읽기 실패: {path}")
    return img, sha


def stats(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return {"median": statistics.median(vals), "mean": statistics.fmean(vals), "p95": p95,
            "min": vals[0], "max": vals[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--quality", type=int, default=90, help="JPEG 품질 (0~100)")
    ap.add_argument("--warmup", type=int, default=5, help="측정 전에 보내고 버릴 요청 수 (첫 이미지 사용)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 몇 장만 (0 = 전부)")
    ap.add_argument("--out", default="", help="결과 폴더 (기본 runs/jpeg_q<품질>_<시각>)")
    ap.add_argument("--strict", action="store_true", help="SHA256 이 manifest 와 다르면 중단")
    args = ap.parse_args()

    frames = load_manifest(args.manifest)
    if args.limit:
        frames = frames[:args.limit]
    out = args.out or time.strftime(f"runs/jpeg_q{args.quality}_%Y%m%d-%H%M%S")
    os.makedirs(out, exist_ok=True)
    print(f"[manifest] {len(frames)}장, 첫 파일 {frames[0][0]}")
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    sock = socket.create_connection((args.host, args.port), timeout=30)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[연결] {args.host}:{args.port}  JPEG quality {args.quality}")

    # 워밍업: 서버·연결 준비 시간을 측정에서 분리
    img0, _ = read_png(frames[0][0])
    warm = []
    for i in range(args.warmup):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img0, enc_params)
        res, rtt = request(sock, {"mode": "jpeg", "frame_id": f"warmup-{i}", "warmup": True}, buf.tobytes())
        warm.append({"total_ms": (time.perf_counter() - t0) * 1000, "rtt_ms": rtt, "server_ms": res.get("server_ms")})
    if warm:
        print(f"[워밍업] {len(warm)}회, 마지막 왕복 {warm[-1]['rtt_ms']:.0f}ms")

    rows = []
    for fid, (path, sha_expected) in enumerate(frames):
        t = time.perf_counter()
        img, sha = read_png(path)
        png_read_ms = (time.perf_counter() - t) * 1000
        sha_ok = (sha == sha_expected) if sha_expected else None
        if sha_ok is False:
            msg = f"SHA256 불일치: {path}"
            if args.strict:
                raise SystemExit(msg)
            print("[경고] " + msg)

        t0 = time.perf_counter()  # ---- 측정 시작: JPEG 인코딩
        ok, buf = cv2.imencode(".jpg", img, enc_params)
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        payload = buf.tobytes()
        t1 = time.perf_counter()
        res, rtt = request(sock, {"mode": "jpeg", "frame_id": fid, "quality": args.quality}, payload)
        t2 = time.perf_counter()  # ---- 측정 끝: 결과 수신
        if res.get("error"):
            raise RuntimeError(f"서버 오류 (frame {fid}): {res['error']}")
        if res.get("frame_id") != fid:
            raise RuntimeError(f"frame_id 불일치: 보냄 {fid}, 받음 {res.get('frame_id')}")
        st = res.get("t", {})
        row = {
            "frame_id": fid, "file": os.path.basename(path), "sha256": sha, "sha256_match": sha_ok,
            "width": img.shape[1], "height": img.shape[0], "jpeg_quality": args.quality,
            "png_read_ms": png_read_ms, "encode_ms": (t1 - t0) * 1000, "jpeg_bytes": len(payload),
            "rtt_ms": rtt, "server_ms": res["server_ms"],
            "server_decode_ms": st.get("decode_ms"), "server_preprocess_ms": st.get("preprocess_ms"),
            "server_infer_ms": st.get("infer_ms"), "server_post_ms": st.get("post_ms"),
            "net_ms": rtt - res["server_ms"], "total_ms": (t2 - t0) * 1000,
            "num_dets": len(res["dets"]), "dets": res["dets"],
        }
        rows.append(row)
        top = ", ".join(f"{d['cls']} {d['conf']:.2f}" for d in res["dets"][:3]) or "-"
        print(f"frame {fid:3d} {row['file']}: {len(payload)/1024:6.1f}KB | 인코딩 {row['encode_ms']:5.1f} + "
              f"왕복 {rtt:6.1f} (서버 {res['server_ms']:6.1f}, 네트워크 {row['net_ms']:5.1f}) = 전체 {row['total_ms']:6.1f}ms | "
              f"{row['num_dets']}개 ({top})")
    sock.close()

    # ---- 저장
    cols = [k for k in rows[0] if k != "dets"] + ["dets_json"]
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: v for k, v in r.items() if k != "dets"}, "dets_json": json.dumps(r["dets"])})
    meta = {"mode": "jpeg", "host": args.host, "port": args.port, "jpeg_quality": args.quality,
            "manifest": os.path.abspath(args.manifest), "frames": len(rows), "warmup": warm,
            "opencv": cv2.__version__, "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "boundary": "total_ms = JPEG encode start -> result received (PNG read excluded)"}
    json.dump({"meta": meta, "frames": rows}, open(os.path.join(out, "frames.json"), "w"), ensure_ascii=False, indent=1)
    keys = ["png_read_ms", "encode_ms", "jpeg_bytes", "rtt_ms", "server_ms", "server_decode_ms",
            "server_preprocess_ms", "server_infer_ms", "server_post_ms", "net_ms", "total_ms"]
    summary = {"meta": meta, **{k: stats([r[k] for r in rows if r[k] is not None]) for k in keys}}
    summary["sha256_mismatch"] = sum(r["sha256_match"] is False for r in rows)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n===== JPEG q{args.quality}, {len(rows)}장 (워밍업 {len(warm)}회 제외) — 중앙값 / p95 =====")
    for k, label in [("jpeg_bytes", "JPEG 크기(B)"), ("encode_ms", "보드 인코딩"), ("rtt_ms", "왕복"),
                     ("server_ms", "서버 전체"), ("server_decode_ms", " ㄴ JPEG 디코딩"),
                     ("server_preprocess_ms", " ㄴ 전처리"), ("server_infer_ms", " ㄴ 모델"),
                     ("server_post_ms", " ㄴ 후처리"), ("net_ms", "네트워크"), ("total_ms", "전체(인코딩~결과)")]:
        s = summary[k]
        print(f"{label:<16} {s['median']:10.1f} / {s['p95']:10.1f}")
    print(f"SHA256 불일치: {summary['sha256_mismatch']}장 | 저장: {out}/frames.csv, frames.json, summary.json")


if __name__ == "__main__":
    main()
