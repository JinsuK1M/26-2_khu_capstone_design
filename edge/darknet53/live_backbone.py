"""OV13855 NV12 continuous capture -> latest frame -> RKNN backbone.

Requires the previously verified 1280x960 NV12 device layout (stride 1280).
Optional browser preview. No FCM or detection head. Last processed frame saved.
"""
import argparse
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=30, help='0: run until Ctrl+C')
    parser.add_argument('--device', default='/dev/video11')
    parser.add_argument('--preview-port', type=int, default=0, help='0: disabled; otherwise loopback HTTP port')
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error('--seconds must be nonnegative')
    root = Path(__file__).resolve().parent
    width, height = 1280, 960
    frame_bytes = width * height * 3 // 2
    subprocess.run(['v4l2-ctl', '-d', args.device,
                    '--set-fmt-video=width=1280,height=960,pixelformat=NV12'], check=True)
    fmt = subprocess.check_output(['v4l2-ctl', '-d', args.device, '--get-fmt-video'], text=True)
    print(fmt, flush=True)
    # Refuse silently negotiated sizes/strides incompatible with raw parsing.
    import re
    for pattern in (r'Width/Height\s*:\s*1280/960', r"Pixel Format\s*:\s*'NV12'",
                    r'Bytes per Line\s*:\s*1280\b', r'Size Image\s*:\s*1843200\b'):
        if not re.search(pattern, fmt):
            raise RuntimeError('Unexpected camera layout; stopping before capture')
    lock = threading.Lock()
    latest = [None, 0, 0.0]
    ended = threading.Event()
    proc = None
    last_image = last_outputs = None
    processed = 0
    preview = None
    rknn = RKNNLite()
    try:
        if args.preview_port:
            from camera_preview import Preview
            preview = Preview(args.preview_port)
            print(f'Preview: http://127.0.0.1:{args.preview_port} (SSH tunnel required)', flush=True)
        if rknn.load_rknn(str(root / 'backbone_fp.rknn')) != 0:
            raise RuntimeError('RKNN load failed')
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('NPU initialization failed')
        with (root / 'live_capture.log').open('wb') as log:
            proc = subprocess.Popen(['v4l2-ctl', '-d', args.device, '--stream-mmap=4',
                                     '--stream-skip=10', '--stream-count=0', '--stream-to=-'],
                                    stdout=subprocess.PIPE, stderr=log, bufsize=0)

            def read_frames():
                try:
                    while True:
                        data = bytearray()
                        while len(data) < frame_bytes:
                            chunk = proc.stdout.read(frame_bytes - len(data))
                            if not chunk:
                                return
                            data.extend(chunk)
                        with lock:
                            latest[0] = bytes(data)
                            latest[1] += 1
                            latest[2] = time.monotonic()
                finally:
                    ended.set()

            threading.Thread(target=read_frames, daemon=True).start()
            start = last_received = time.monotonic()
            seen = 0
            print('LIVE START: Ctrl+C to stop', flush=True)
            while args.seconds == 0 or time.monotonic() - start < args.seconds:
                with lock:
                    data, number, received = latest
                if number == seen:
                    if ended.is_set():
                        raise RuntimeError('Camera stream ended; see live_capture.log')
                    if time.monotonic() - last_received > 15:
                        raise RuntimeError('No new camera frames for 15s; see live_capture.log')
                    time.sleep(0.005)
                    continue
                seen = number
                last_received = received
                image = cv2.cvtColor(np.frombuffer(data, np.uint8).reshape(height * 3 // 2, width),
                                     cv2.COLOR_YUV2BGR_NV12)
                resized = cv2.resize(image, (640, 480))
                canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
                canvas[:480] = resized
                tick = time.perf_counter()
                outputs = rknn.inference(inputs=[canvas[None]])
                api_ms = (time.perf_counter() - tick) * 1000
                expected = [(1, 256, 80, 80), (1, 512, 40, 40), (1, 512, 20, 20)]
                if outputs is None or len(outputs) != 3:
                    raise RuntimeError('Invalid backbone outputs')
                for output, shape in zip(outputs, expected):
                    if output.shape != shape or not np.isfinite(output).all():
                        raise RuntimeError('Invalid feature shape or values')
                processed += 1
                last_image, last_outputs = image, outputs
                elapsed = time.monotonic() - start
                ranges = ' '.join(f'{name}={v.min():.2f}..{v.max():.2f}'
                                  for name, v in zip(('dark3', 'dark4', 'dark5'), outputs))
                print(f'processed={processed} capture_id={number} api={api_ms:.1f}ms '
                      f'loop_avg={processed/elapsed:.2f}fps {ranges}', flush=True)
                if preview is not None:
                    ok, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError('Preview JPEG encoding failed')
                    preview.publish(encoded.tobytes(), {
                        'processed': processed, 'capture_id': number,
                        'backbone_api_ms': round(api_ms, 2),
                        'loop_avg_fps': round(processed / elapsed, 2),
                        'features': {name: {'shape': list(value.shape),
                                           'min': float(value.min()), 'max': float(value.max())}
                                     for name, value in zip(('dark3', 'dark4', 'dark5'), outputs)}
                    })
    except KeyboardInterrupt:
        print('\nStopped by user', flush=True)
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
        rknn.release()
        if preview is not None:
            preview.close()
        if last_outputs is not None:
            np.savez(root / 'live_last_features.npz',
                     **dict(zip(('dark3', 'dark4', 'dark5'), last_outputs)))
            if not cv2.imwrite(str(root / 'live_last.jpg'), last_image):
                raise RuntimeError('Failed to save last camera image')
            print('Saved live_last.jpg and live_last_features.npz', flush=True)


if __name__ == '__main__':
    main()
