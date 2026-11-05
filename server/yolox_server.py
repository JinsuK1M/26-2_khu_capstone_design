"""
YOLOX-Darknet53 서버 (Server 파트)

요청 하나 = [길이][JSON 헤더] + [길이][payload], 응답 = [길이][JSON 결과]. 형식은 docs/server_protocol.md

모드 (헤더의 "mode", 없으면 예전 방식으로 판단)
  fcm  : payload = FCM bytes (24B 헤더 + HM 비트스트림)
         -> HM 디코딩 -> DRNet -> dark3 -> dark4·dark5·neck·head -> 후처리      (비교군 B)
  jpeg : payload = JPEG bytes
         -> JPEG 디코딩 -> 640 전처리(비율 유지, 오른쪽·아래 114 패딩) -> YOLOX 전체 -> 후처리   (비교군 A)
  hevc : payload = HEVC Annex-B Access Unit 1개 (프레임 1장, 시작 코드 포함)
         -> 연결별 HEVC 디코더(상태 유지) -> 프레임 1장 -> 640 전처리 -> YOLOX 전체 -> 후처리 (비교군 A')
         첫 요청은 VPS/SPS/PPS + I 프레임이어야 한다. 새 연결 또는 헤더 "reset": true 면 디코더를 새로 만든다.
         PyAV 필요: pip install av
  tensor : 압축 없는 특징 텐서들 (Mac 실험용, 예전 방식)

두 모드 모두 같은 체크포인트, 같은 confidence(기본 0.25)·NMS(0.45)를 쓰고
박스는 원본 사진 좌표로 돌려준다. 헤더의 frame_id 는 결과에 그대로 돌려준다.
서버 시간은 payload 를 다 받은 직후부터 결과 JSON 을 만들기 직전까지 잰다.

사용법:  python yolox_server.py            (포트 9997)
"""
import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np
import torch

from yolox_split import load_model, preprocess, server_part, to_detections

MAX_MSG = 64 * 1024 * 1024  # 한 메시지 최대 64MB (비정상 길이 방어)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    if n > MAX_MSG:
        raise ConnectionError(f"메시지 길이 {n} 가 최대 {MAX_MSG} 초과")
    return recv_exact(sock, n)


def send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


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


def get_mode(hdr):
    if "mode" in hdr:
        return hdr["mode"]
    return "fcm" if hdr.get("fcm") else "tensor"


class HevcStream:
    """연결 하나의 HEVC 디코더. P 프레임이 앞 프레임을 참조하므로 연결이 끝날 때까지 유지한다."""

    def __init__(self):
        try:
            import av
        except ImportError as e:
            raise ValueError("HEVC 모드에는 PyAV 가 필요해요: pip install av") from e
        self.av = av
        self.ctx = av.CodecContext.create("hevc", "r")
        self.ctx.thread_count = 1  # 프레임 스레딩을 끄면 AU 하나 -> 프레임 하나가 바로 나온다 (지연 없음)
        self.received = 0
        self.decoded = 0

    def decode(self, au):
        """AU 1개 -> BGR 프레임 목록 (정상이면 1개)"""
        self.received += 1
        frames = self.ctx.decode(self.av.Packet(au))
        out = []
        for fr in frames:
            pt = getattr(fr.pict_type, "name", None) or {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), str(fr.pict_type))
            out.append((fr.to_ndarray(format="bgr24"), pt))
        self.decoded += len(out)
        return out


class Server:
    def __init__(self, weights, conf):
        self.model = load_model(weights)
        self.conf = conf
        self.fcm = None  # FCM 요청이 처음 오면 불러온다 (워밍업에서 처리)

    @torch.no_grad()
    def handle_jpeg(self, hdr, payload):
        t = {}
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("JPEG 디코딩 실패")
        t1 = time.perf_counter()
        x, r = preprocess(img, 640)
        t2 = time.perf_counter()
        pred = self.model(x)
        t3 = time.perf_counter()
        dets = to_detections(pred, r, conf=self.conf)
        t4 = time.perf_counter()
        t.update(decode_ms=(t1 - t0) * 1000, preprocess_ms=(t2 - t1) * 1000,
                 infer_ms=(t3 - t2) * 1000, post_ms=(t4 - t3) * 1000)
        return dets, t, {"image_hw": list(img.shape[:2]), "ratio": r}

    @torch.no_grad()
    def handle_fcm(self, hdr, payload):
        if self.fcm is None:
            from fcm_plugin_yolox import load_fcm
            self.fcm = load_fcm(hdr.get("qp", 20))
        check_fcm_payload(payload)
        t0 = time.perf_counter()
        dark3, td = self.fcm.decompress_timed(payload)
        t1 = time.perf_counter()
        pred = server_part(self.model, {"dark3": dark3})
        t2 = time.perf_counter()
        dets = to_detections(pred, hdr.get("ratio", 1.0), conf=self.conf)
        t3 = time.perf_counter()
        t = dict(td, decode_ms=(t1 - t0) * 1000, infer_ms=(t2 - t1) * 1000, post_ms=(t3 - t2) * 1000)
        return dets, t, {}

    @torch.no_grad()
    def handle_tensor(self, hdr, payload):
        t0 = time.perf_counter()
        feats, pos = {}, 0
        for tt in hdr["tensors"]:
            feats[tt["name"]] = unpack(tt, payload[pos:pos + tt["nbytes"]])
            pos += tt["nbytes"]
        t1 = time.perf_counter()
        pred = server_part(self.model, feats)
        t2 = time.perf_counter()
        dets = to_detections(pred, hdr.get("ratio", 1.0), conf=self.conf)
        t3 = time.perf_counter()
        return dets, {"decode_ms": (t1 - t0) * 1000, "infer_ms": (t2 - t1) * 1000,
                      "post_ms": (t3 - t2) * 1000}, {}

    @torch.no_grad()
    def handle_hevc(self, hdr, payload, state):
        if hdr.get("reset") or "hevc" not in state:
            state["hevc"] = HevcStream()
        hs = state["hevc"]
        t0 = time.perf_counter()
        try:
            frames = hs.decode(payload)
        except Exception as e:  # 디코더 오류 (예: 첫 AU 에 SPS/PPS/I 프레임이 없음)
            raise ValueError(f"HEVC 디코딩 실패: {e}")
        t1 = time.perf_counter()
        if len(frames) != 1:
            raise ValueError(f"AU 1개에서 프레임 {len(frames)}개가 나옴 (요청 {hs.received}, 디코딩 누적 {hs.decoded}). "
                             "첫 AU 가 VPS/SPS/PPS + I 프레임인지, Access Unit 단위(alignment=au)로 보내는지 확인해 주세요. "
                             "다시 시작하려면 새로 연결하거나 헤더에 reset: true")
        img, ptype = frames[0]
        x, r = preprocess(img, 640)
        t2 = time.perf_counter()
        pred = self.model(x)
        t3 = time.perf_counter()
        dets = to_detections(pred, r, conf=self.conf)
        t4 = time.perf_counter()
        t = {"decode_ms": (t1 - t0) * 1000, "preprocess_ms": (t2 - t1) * 1000,
             "infer_ms": (t3 - t2) * 1000, "post_ms": (t4 - t3) * 1000}
        return dets, t, {"image_hw": list(img.shape[:2]), "ratio": r, "frame_type": ptype,
                         "decoded_index": hs.decoded - 1}

    def handle(self, hdr, payload, state=None):
        state = {} if state is None else state
        mode = get_mode(hdr)
        t0 = time.perf_counter()
        if mode == "hevc":
            dets, t, extra = self.handle_hevc(hdr, payload, state)
        else:
            fn = {"jpeg": self.handle_jpeg, "fcm": self.handle_fcm, "tensor": self.handle_tensor}.get(mode)
            if fn is None:
                raise ValueError(f"알 수 없는 mode: {mode}")
            dets, t, extra = fn(hdr, payload)
        server_ms = (time.perf_counter() - t0) * 1000
        resp = {"frame_id": hdr.get("frame_id"), "mode": mode, "payload_bytes": len(payload),
                "dets": dets, "server_ms": server_ms, "t": t, **extra}
        # 예전 클라이언트 호환
        resp["unpack_ms"] = t.get("decode_ms", 0.0) + t.get("preprocess_ms", 0.0)
        resp["tail_ms"] = t.get("infer_ms", 0.0)
        return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--weights", default="yolox_darknet.pth")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--quiet", action="store_true", help="프레임마다 출력하지 않기")
    args = ap.parse_args()

    server = Server(args.weights, args.conf)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[yolox-server] ready on port {args.port} (mode: fcm / jpeg / hevc / tensor, conf {args.conf}, nms 0.45)")

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[yolox-server] client connected: {addr}")
        state = {}  # 연결별 상태 (HEVC 디코더). 연결이 끝나면 버린다
        try:
            while True:
                hdr = json.loads(recv_msg(conn))
                payload = recv_msg(conn)
                try:
                    resp = server.handle(hdr, payload, state)
                except ValueError as e:  # 잘못된 요청은 오류만 돌려주고 연결은 유지
                    send_msg(conn, json.dumps({"frame_id": hdr.get("frame_id"), "mode": get_mode(hdr),
                                               "error": str(e), "dets": [], "server_ms": 0.0}).encode())
                    print(f"[yolox-server] frame {hdr.get('frame_id')} 오류: {e}")
                    continue
                send_msg(conn, json.dumps(resp).encode())
                if not args.quiet:
                    t = resp["t"]
                    parts = " + ".join(f"{k[:-3]} {v:.0f}" for k, v in t.items() if k in
                                       ("hm_decode_ms", "drnet_ms", "decode_ms", "preprocess_ms", "infer_ms", "post_ms")
                                       and not (k == "decode_ms" and "hm_decode_ms" in t))
                    ft = f" {resp['frame_type']}" if "frame_type" in resp else ""
                    print(f"[yolox-server] {resp['mode']}{ft} frame {resp['frame_id']} | {len(payload)/1024:.1f}KB | "
                          f"{len(resp['dets'])} obj | server {resp['server_ms']:.0f}ms ({parts})")
        except ConnectionError:
            print("[yolox-server] client disconnected, waiting...")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
