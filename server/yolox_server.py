"""
YOLOX-Darknet53 서버 (Server 파트)
  Edge(Orange Pi 또는 가짜 Edge)가 보낸 dark3, dark4, dark5 를 받아 FPN + head 를 실행하고 탐지 결과를 돌려준다.
  FCM 모드: 클라이언트가 보낸 bytes(FCM 압축 결과)를 복원해 dark3 를 만들고 나머지를 실행한다.
  보내는 쪽이 헤더에 frame_id 를 넣으면 결과에 그대로 돌려준다 (어느 프레임 결과인지 짝 맞추기).

사용법:  python yolox_server.py
"""
import argparse
import json
import socket
import struct
import time

import numpy as np
import torch

from yolox_split import load_model, server_part, to_detections


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return buf


def recv_msg(sock):
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    if n > MAX_MSG:
        raise ConnectionError(f"메시지 길이 {n} 가 최대 {MAX_MSG} 초과")
    return recv_exact(sock, n)


def send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


MAX_MSG = 64 * 1024 * 1024  # 한 메시지 최대 64MB (비정상 길이 방어)


def check_fcm_payload(blob):
    """FCM payload 규격 검사: <ddHHI 헤더(24B) + HM 비트스트림 n 바이트"""
    if len(blob) < 24:
        raise ValueError(f"payload 가 너무 짧음: {len(blob)} bytes")
    std, mean, h, w, n = struct.unpack_from("<ddHHI", blob)
    if len(blob) != 24 + n:
        raise ValueError(f"길이 불일치: 전체 {len(blob)}, 헤더 기준 {24 + n}")
    if (h, w) != (320, 640) or not (np.isfinite(std) and np.isfinite(mean)):
        raise ValueError(f"헤더 값 이상: tile {h}x{w}, std {std}, mean {mean}")


def unpack(t, payload):
    if t["dtype"] == "fp32":
        a = np.frombuffer(payload, np.float32)
    elif t["dtype"] == "fp16":
        a = np.frombuffer(payload, np.float16).astype(np.float32)
    else:  # uint8 min-max
        a = np.frombuffer(payload, np.uint8).astype(np.float32) * t["scale"] + t["zero"]
    return torch.from_numpy(a.reshape(t["shape"]).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--weights", default="yolox_darknet.pth")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = load_model(args.weights)
    fcm = None  # FCM 모드 요청이 처음 오면 그때 불러온다
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[yolox-server] ready on port {args.port} (특징 또는 FCM bytes 를 기다리는 중)")

    while True:
        conn, addr = srv.accept()
        print(f"[yolox-server] client connected: {addr}")
        try:
            while True:
                hdr = json.loads(recv_msg(conn))
                payload = recv_msg(conn)
                t0 = time.perf_counter()
                if hdr.get("fcm"):  # FCM bytes -> 복원 dark3
                    if fcm is None:
                        from fcm_plugin_yolox import load_fcm
                        fcm = load_fcm(hdr.get("qp", 20))
                    try:
                        check_fcm_payload(payload)
                    except ValueError as e:  # 잘못된 payload 는 오류만 돌려주고 연결은 유지
                        send_msg(conn, json.dumps({"frame_id": hdr.get("frame_id"), "error": str(e),
                                                   "dets": [], "server_ms": 0.0}).encode())
                        print(f"[yolox-server] frame {hdr.get('frame_id')} payload 오류: {e}")
                        continue
                    feats = {"dark3": fcm.decompress(payload)}
                    kind = f"FCM QP{hdr.get('qp', 20)}"
                else:
                    feats, pos = {}, 0
                    for t in hdr["tensors"]:
                        feats[t["name"]] = unpack(t, payload[pos:pos + t["nbytes"]])
                        pos += t["nbytes"]
                    kind = "+".join(feats) + " " + hdr["tensors"][0]["dtype"]
                t1 = time.perf_counter()
                pred = server_part(model, feats)
                t2 = time.perf_counter()
                dets = to_detections(pred, hdr.get("ratio", 1.0), conf=args.conf)
                resp = {"frame_id": hdr.get("frame_id"), "dets": dets, "unpack_ms": (t1 - t0) * 1000, "tail_ms": (t2 - t1) * 1000,
                        "server_ms": (time.perf_counter() - t0) * 1000}
                send_msg(conn, json.dumps(resp).encode())
                print(f"[yolox-server] frame {hdr.get('frame_id')} | {len(payload)/1024:.1f}KB {kind}, {len(dets)} objects, "
                      f"복원 {resp['unpack_ms']:.0f}ms + 탐지 {resp['tail_ms']:.0f}ms")
        except ConnectionError:
            print("[yolox-server] client disconnected, waiting...")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
