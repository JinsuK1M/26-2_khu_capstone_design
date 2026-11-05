"""동일 PNG 입력을 지속 HEVC 인코더로 압축·전송. PNG 읽기 제외, 압축 포함 시간 측정."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime

import cv2
import numpy as np
# Ubuntu 시스템 GI를 기존 Python 3.10 가상환경에서 사용. pip로 gi를 설치하지 않습니다.
try:
    import gi
except ModuleNotFoundError:
    sys.path.append('/usr/lib/python3/dist-packages')
    import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from hevc_send import HevcLink, split_access_units


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Encoder:
    def __init__(self, bitrate, gop):
        self.pipeline = Gst.parse_launch(
            'appsrc name=source is-live=true format=time block=true '
            'caps=video/x-raw,format=BGR,width=1280,height=960,framerate=1/1 '
            '! videoconvert ! video/x-raw,format=NV12 '
            f'! mpph265enc rc-mode=cbr bps={bitrate} gop={gop} header-mode=each-idr max-pending=1 '
            '! appsink name=sink sync=false async=false max-buffers=1 drop=false')
        self.source = self.pipeline.get_by_name('source')
        self.sink = self.pipeline.get_by_name('sink')
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError('GStreamer 시작 실패')
        self.index = 0

    def encode(self, image):
        raw = image.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(raw), None)
        buffer.fill(0, raw)
        expected_pts = self.index * Gst.SECOND
        buffer.pts = expected_pts
        buffer.duration = Gst.SECOND
        if self.source.emit('push-buffer', buffer) != Gst.FlowReturn.OK:
            raise RuntimeError('인코더 입력 실패')
        sample = self.sink.emit('try-pull-sample', 10 * Gst.SECOND)
        if sample is None:
            msg = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            detail = str(msg.parse_error()) if msg else '10초 내 AU 출력 없음: 입력 대기/버퍼링 가능'
            raise RuntimeError(detail)
        caps = sample.get_caps().get_structure(0)
        if caps.get_string('alignment') != 'au' or caps.get_string('stream-format') != 'byte-stream':
            raise RuntimeError('출력은 byte-stream, alignment=au여야 합니다')
        output = sample.get_buffer()
        if output.pts != expected_pts:
            raise RuntimeError(f'PTS 불일치: expected={expected_pts}, actual={output.pts}')
        au = output.extract_dup(0, output.get_size())
        if len(split_access_units(au)) != 1:
            raise RuntimeError('출력 버퍼의 AU 수가 1이 아닙니다')
        self.index += 1
        return au, expected_pts

    def finish(self):
        if self.source.emit('end-of-stream') != Gst.FlowReturn.OK:
            raise RuntimeError('EOS 입력 실패')
        message = self.pipeline.get_bus().timed_pop_filtered(
            10 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if message is None:
            raise RuntimeError('EOS 종료 확인 시간 초과')
        if message.type == Gst.MessageType.ERROR:
            raise RuntimeError(str(message.parse_error()))

    def close(self):
        self.pipeline.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=9997)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--bitrate', type=int, default=1000000)
    parser.add_argument('--gop', type=int, default=30)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if min(args.bitrate, args.gop) <= 0 or min(args.warmup, args.limit) < 0:
        parser.error('옵션 범위 오류')
    Gst.init(None)
    inputs = args.inputs.expanduser().resolve()
    manifest = inputs / 'manifest.json'
    data = json.loads(manifest.read_text())
    if data.get('state') != 'completed':
        raise ValueError('완료된 입력 manifest 필요')
    frames = data['frames'][:args.limit or None]
    if not frames:
        raise ValueError('입력 없음')
    for frame in frames:
        path = (inputs / frame['file']).resolve()
        if not path.is_relative_to(inputs) or sha(path) != frame['sha256']:
            raise ValueError('입력 경로 또는 SHA256 불일치')
    out = Path(__file__).resolve().parent / 'replay_hevc_encode_runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    rows, warmups = [], []
    report = dict(state='running', manifest_sha256=sha(manifest), bitrate=args.bitrate,
                  gop=args.gop, fps_metadata=1, pacing=False, expected_frames=len(frames),
                  boundary='PNG 읽기 제외. BGR 버퍼 복사·색 변환·HEVC 압축 시작부터 서버 결과 수신까지. 센서 E2E/mAP 아님.',
                  warmups=warmups, gst_version=Gst.version_string(), opencv=cv2.__version__)
    try:
        for warm in ([True, False] if args.warmup else [False]):
            encoder = Encoder(args.bitrate, args.gop)
            link = None
            try:
                link = HevcLink(args.host, args.port)
                selected = [frames[0]] * args.warmup if warm else frames
                for index, frame in enumerate(selected):
                    path = inputs / frame['file']
                    load_start = time.perf_counter()
                    image = cv2.imread(str(path))
                    file_load_ms = (time.perf_counter() - load_start) * 1000
                    if image is None or image.shape != (960, 1280, 3):
                        raise ValueError('입력 크기 오류: 1280x960 BGR 이미지가 필요합니다')
                    start = time.perf_counter()
                    au, pts = encoder.encode(image)
                    encoded = time.perf_counter()
                    result = link.send(au, frame['frame_id'], {'warmup': warm})
                    end = time.perf_counter()
                    if result.get('decoded_index') != index:
                        raise RuntimeError('서버 디코딩 순번 불일치')
                    row = dict(frame_id=frame['frame_id'], file=frame['file'], sha256=frame['sha256'], pts=pts,
                               file_load_ms=file_load_ms, encode_ms=(encoded-start)*1000,
                               e2e_ms=(end-start)*1000, payload_bytes=len(au),
                               rtt_ms=result['rtt_ms'], server_ms=result['server_ms'],
                               network_residual_ms=result['net_ms'], frame_type=result.get('frame_type'))
                    if warm:
                        warmups.append(row)
                    else:
                        rows.append(row)
                        (out / f"frame_{index+1:04d}.json").write_text(json.dumps(dict(row=row, response=result), ensure_ascii=False))
                        with (out / 'stream.hevc').open('ab') as stream:
                            stream.write(au)
                        with (out / 'frames.csv').open('w', newline='') as file:
                            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                        print(f"{index+1}/{len(frames)} encode={row['encode_ms']:.1f}ms E2E={row['e2e_ms']:.1f}ms bytes={len(au)}", flush=True)
                encoder.finish()
            finally:
                if link is not None:
                    link.close()
                encoder.close()
        report['state'] = 'completed'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report['completed_frames'] = len(rows)
        report['measured'] = {key: dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                     p95=float(np.percentile(values, 95)))
                              for key in ('encode_ms','e2e_ms','payload_bytes','rtt_ms','server_ms','network_residual_ms')
                              for values in [[row[key] for row in rows]] if values}
        (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print('\n=== HEVC ENCODE-TO-RESULT SUMMARY ===', flush=True)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        print('저장:', out, flush=True)


if __name__ == '__main__':
    main()
