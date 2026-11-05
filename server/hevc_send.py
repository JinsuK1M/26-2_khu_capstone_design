"""
비교군 A': 보드에서 HEVC Access Unit(프레임 1장)을 하나씩 서버로 보내고 탐지 결과를 받는다.
(보드에서 실행. 파이썬 기본 라이브러리만 사용)

서버(yolox_server.py, mode="hevc")는 연결 하나에 HEVC 디코더 하나를 유지한다.
  - 같은 연결로 AU 를 순서대로 보내야 한다 (P 프레임이 앞 프레임을 참조).
  - 첫 AU 는 VPS/SPS/PPS + I(IDR) 프레임이어야 한다.
  - AU 1개 -> 디코딩 프레임 1개 -> 결과 1개. 결과의 frame_id 로 짝을 맞춘다.

1) 저장된 .hevc 파일로 시험 (파일을 AU 단위로 잘라 보냄)
   python3 hevc_send.py test_gstreamer.hevc --host 100.90.83.29
   python3 hevc_send.py test_gstreamer.hevc --host 100.90.83.29 --warmup 5 --out runs/hevc_file

2) GStreamer 실시간 연결: appsink 샘플 하나 = AU 하나가 되도록
     ... ! mpph265enc ... ! h265parse config-interval=-1 ! video/x-h265,stream-format=byte-stream,alignment=au ! appsink
   from hevc_send import HevcLink
   link = HevcLink("100.90.83.29")                 # 연결 = 새 디코더
   res = link.send(au_bytes, frame_id=i)            # 결과 dict + rtt_ms, net_ms
   (alignment=nal 로 받는 경우에는 split_access_units() 로 묶어서 보내면 된다)

로그 (CSV/JSON): frame_id, AU 바이트, 프레임 종류(I/P), rtt_ms, 서버 시간(디코딩·전처리·모델·후처리·전체), net_ms, 탐지 결과
"""
import argparse
import csv
import json
import os
import socket
import statistics
import struct
import time


# ---------------------------------------------------------------- HEVC Annex-B -> Access Unit
def _nal_units(data):
    """Annex-B 바이트열 -> [(시작 위치, 끝 위치, NAL type, first_slice_segment_in_pic_flag)] (시작 코드 포함 범위)"""
    starts, i, n = [], 0, len(data)
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0:
            break
        s = i - 1 if i > 0 and data[i - 1] == 0 else i  # 4바이트 시작 코드
        starts.append((s, i + 3))
        i += 3
    out = []
    for k, (s, h) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else n
        if h + 2 >= e:
            continue
        nal_type = (data[h] >> 1) & 0x3F
        first_slice = bool(data[h + 2] & 0x80) if nal_type < 32 else False
        out.append((s, e, nal_type, first_slice))
    return out


def split_access_units(data):
    """HEVC Annex-B 스트림을 Access Unit(프레임 1장 단위) bytes 목록으로 자른다.
    새 AU 시작: 이미 VCL(슬라이스)을 담은 AU 다음에 VPS/SPS/PPS/AUD/prefix SEI 가 오거나,
               first_slice_segment_in_pic_flag=1 인 슬라이스가 올 때"""
    aus, cur_start, has_vcl = [], None, False
    for s, e, t, first in _nal_units(data):
        new_au = has_vcl and (t in (32, 33, 34, 35, 39) or (t < 32 and first))
        if cur_start is None:
            cur_start = s
        elif new_au:
            aus.append(data[cur_start:s])
            cur_start, has_vcl = s, False
        if t < 32:
            has_vcl = True
    if cur_start is not None and has_vcl:
        aus.append(data[cur_start:])
    return aus


# ---------------------------------------------------------------- 통신
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("server closed")
        buf += chunk
    return bytes(buf)


class HevcLink:
    """서버 연결 하나 = 서버 HEVC 디코더 하나. AU 를 순서대로 send() 한다."""

    def __init__(self, host, port=9997, timeout=30):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, au, frame_id, extra=None):
        h = json.dumps(dict({"mode": "hevc", "frame_id": frame_id}, **(extra or {}))).encode()
        t0 = time.perf_counter()
        self.sock.sendall(struct.pack(">I", len(h)) + h + struct.pack(">I", len(au)) + au)
        (n,) = struct.unpack(">I", _recv_exact(self.sock, 4))
        res = json.loads(_recv_exact(self.sock, n))
        rtt = (time.perf_counter() - t0) * 1000
        if res.get("error"):
            raise RuntimeError(f"서버 오류 (frame {frame_id}): {res['error']}")
        if res.get("frame_id") != frame_id:
            raise RuntimeError(f"frame_id 불일치: 보냄 {frame_id}, 받음 {res.get('frame_id')}")
        res["rtt_ms"] = rtt
        res["net_ms"] = rtt - res["server_ms"]
        return res

    def close(self):
        self.sock.close()


def stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return {"median": statistics.median(vals), "mean": statistics.fmean(vals), "p95": p95, "min": vals[0], "max": vals[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hevc", help="Annex-B HEVC 파일 (.hevc / .h265)")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--warmup", type=int, default=5, help="별도 연결로 앞 AU 몇 개를 먼저 보내고 버림 (서버 준비용)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = open(args.hevc, "rb").read()
    aus = split_access_units(data)
    if args.limit:
        aus = aus[:args.limit]
    print(f"[파일] {args.hevc}: {len(data):,} bytes -> AU {len(aus)}개")
    out = args.out or time.strftime("runs/hevc_%Y%m%d-%H%M%S")
    os.makedirs(out, exist_ok=True)

    warm = []
    if args.warmup:
        link = HevcLink(args.host, args.port)  # 워밍업 전용 연결 (끝나면 서버 디코더도 버려짐)
        for i, au in enumerate(aus[:args.warmup]):
            r = link.send(au, f"warmup-{i}", {"warmup": True})
            warm.append({"rtt_ms": r["rtt_ms"], "server_ms": r["server_ms"]})
        link.close()
        print(f"[워밍업] {len(warm)}회 (별도 연결), 마지막 왕복 {warm[-1]['rtt_ms']:.0f}ms")

    link = HevcLink(args.host, args.port)  # 측정 연결 = 새 디코더, AU 0 부터
    print(f"[연결] {args.host}:{args.port}")
    rows = []
    for fid, au in enumerate(aus):
        r = link.send(au, fid)
        st = r.get("t", {})
        rows.append({"frame_id": fid, "au_bytes": len(au), "frame_type": r.get("frame_type"),
                     "decoded_index": r.get("decoded_index"), "rtt_ms": r["rtt_ms"], "server_ms": r["server_ms"],
                     "server_decode_ms": st.get("decode_ms"), "server_preprocess_ms": st.get("preprocess_ms"),
                     "server_infer_ms": st.get("infer_ms"), "server_post_ms": st.get("post_ms"),
                     "net_ms": r["net_ms"], "num_dets": len(r["dets"]), "dets": r["dets"]})
        top = ", ".join(f"{d['cls']} {d['conf']:.2f}" for d in r["dets"][:3]) or "-"
        print(f"frame {fid:3d} {r.get('frame_type', '?')}: {len(au)/1024:6.1f}KB | 왕복 {r['rtt_ms']:6.1f} "
              f"(서버 {r['server_ms']:6.1f}: 디코딩 {st.get('decode_ms', 0):5.1f}, 모델 {st.get('infer_ms', 0):6.1f} | "
              f"네트워크 {r['net_ms']:5.1f}) | {len(r['dets'])}개 ({top})")
    link.close()

    cols = [k for k in rows[0] if k != "dets"] + ["dets_json"]
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: v for k, v in r.items() if k != "dets"}, "dets_json": json.dumps(r["dets"])})
    meta = {"mode": "hevc", "file": os.path.abspath(args.hevc), "file_bytes": len(data), "access_units": len(rows),
            "host": args.host, "port": args.port, "warmup": warm, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "boundary": "rtt_ms = AU send start -> result received (encoding time not included for file input)"}
    json.dump({"meta": meta, "frames": rows}, open(os.path.join(out, "frames.json"), "w"), ensure_ascii=False, indent=1)
    keys = ["au_bytes", "rtt_ms", "server_ms", "server_decode_ms", "server_preprocess_ms", "server_infer_ms",
            "server_post_ms", "net_ms"]
    summary = {"meta": meta, **{k: stats([r[k] for r in rows]) for k in keys},
               "frame_types": {t: sum(r["frame_type"] == t for r in rows) for t in sorted({r["frame_type"] for r in rows if r["frame_type"]})}}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n===== HEVC AU {len(rows)}개 (워밍업 {len(warm)}회 제외) — 중앙값 / p95 =====")
    for k, label in [("au_bytes", "AU 크기(B)"), ("rtt_ms", "왕복"), ("server_ms", "서버 전체"),
                     ("server_decode_ms", " ㄴ HEVC 디코딩"), ("server_preprocess_ms", " ㄴ 전처리"),
                     ("server_infer_ms", " ㄴ 모델"), ("server_post_ms", " ㄴ 후처리"), ("net_ms", "네트워크")]:
        s = summary[k]
        print(f"{label:<14} {s['median']:10.1f} / {s['p95']:10.1f}")
    print(f"프레임 종류 {summary['frame_types']} | 저장: {out}/frames.csv, frames.json, summary.json")


if __name__ == "__main__":
    main()
