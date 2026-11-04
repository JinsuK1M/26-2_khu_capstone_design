"""
보드 -> Mac 서버로 FCM payload 를 보내는 코드 (보드에서 실행, 파이썬 기본 라이브러리만 사용)

서버 수신 규격 (yolox_server.py, TCP 포트 9997)
  보낼 때: [4바이트 길이(big-endian)][JSON 헤더] 다음에 [4바이트 길이][payload bytes]
           JSON 헤더 = {"fcm": true, "qp": 20, "hm": "combined4", "ratio": 0.5, "frame_id": 번호}
           ratio = 원본 사진 -> 640 축소 비율 (1280x960 카메라면 0.5). 박스를 원본 좌표로 돌려받기 위함
  받을 때: [4바이트 길이][JSON 결과]
           오류 시 {"frame_id": 번호, "error": "이유", "dets": []}
           정상 {"frame_id": 번호, "dets": [{"box":[x1,y1,x2,y2], "conf":0.9, "cls":"person"}, ...], "server_ms": ...}

1) 저장된 파일로 연결 시험
   python board_send.py --host <Mac 주소> last_payload.bin
   python board_send.py --host <Mac 주소> last_payload.bin --repeat 20     # 같은 파일 20번 -> 네트워크 시간 통계

2) 실시간 코드(live_dark3_fcm.py)에 붙일 때
   from board_send import ServerLink
   link = ServerLink("<Mac 주소>")                       # 루프 시작 전 한 번
   result = link.send(payload, frame_id=processed)      # 압축 끝난 직후
   print(result["dets"], result["net_ms"])
"""
import argparse
import json
import socket
import statistics
import struct
import time


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed")
        buf += chunk
    return buf


def _send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    return _recv_exact(sock, n)


class ServerLink:
    def __init__(self, host, port=9997, qp=20, hm="combined4", ratio=0.5, timeout=10):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.base = {"fcm": True, "qp": qp, "hm": hm, "ratio": ratio}

    def send(self, payload, frame_id):
        """payload 보내고 탐지 결과 받기. 결과에 rtt_ms(왕복), net_ms(왕복 - 서버 처리) 추가"""
        hdr = dict(self.base, frame_id=frame_id)
        t0 = time.perf_counter()
        _send_msg(self.sock, json.dumps(hdr).encode())
        _send_msg(self.sock, payload)
        res = json.loads(_recv_msg(self.sock))
        rtt = (time.perf_counter() - t0) * 1000
        if res.get("frame_id") != frame_id:
            raise RuntimeError(f"frame_id 불일치: 보냄 {frame_id}, 받음 {res.get('frame_id')}")
        res["rtt_ms"] = rtt
        res["net_ms"] = rtt - res["server_ms"]
        return res

    def close(self):
        self.sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload", nargs="+", help="보낼 .bin 파일들")
    ap.add_argument("--host", required=True, help="Mac 서버 주소 (예: Tailscale 100.x.x.x)")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    t = time.perf_counter()
    link = ServerLink(args.host, args.port, ratio=args.ratio)
    print(f"[연결] {args.host}:{args.port} ({(time.perf_counter() - t) * 1000:.0f}ms)")
    rows, fid = [], 0
    for _ in range(args.repeat):
        for p in args.payload:
            blob = open(p, "rb").read()
            r = link.send(blob, fid)
            if r.get("error"):
                print(f"frame {fid}: 서버가 거절함 -> {r['error']}")
                fid += 1
                continue
            rows.append(r)
            top = ", ".join(f"{d['cls']} {d['conf']:.2f}" for d in r["dets"][:3]) or "없음"
            print(f"frame {fid}: {len(blob)/1024:.1f}KB | 왕복 {r['rtt_ms']:.0f}ms = 서버 {r['server_ms']:.0f}ms "
                  f"+ 네트워크 {r['net_ms']:.0f}ms | {len(r['dets'])}개 ({top})")
            fid += 1
    link.close()
    if len(rows) > 1:
        rest = rows[1:]  # 첫 번째는 서버 준비 시간 포함이라 제외
        med = lambda k: statistics.median(r[k] for r in rest)
        print(f"\n[첫 번째 제외 {len(rest)}개 중앙값] 왕복 {med('rtt_ms'):.0f}ms, "
              f"서버 {med('server_ms'):.0f}ms, 네트워크 {med('net_ms'):.0f}ms")


if __name__ == "__main__":
    main()
