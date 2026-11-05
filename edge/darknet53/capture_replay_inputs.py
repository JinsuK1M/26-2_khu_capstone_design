"""Capture 60 lossless camera samples once for identical offline A/B inputs.

No inference or network. Host receive timestamps are not sensor exposure times.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=60)
    parser.add_argument('--interval',type=float,default=1.0)
    parser.add_argument('--device',default='/dev/video11')
    args=parser.parse_args()
    if args.count<1 or args.interval<=0:
        parser.error('count and interval must be positive')
    out=Path(__file__).resolve().parent/'replay_inputs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    subprocess.run(['v4l2-ctl','-d',args.device,
        '--set-fmt-video=width=1280,height=960,pixelformat=NV12'],check=True)
    fmt=subprocess.check_output(['v4l2-ctl','-d',args.device,'--get-fmt-video'],text=True)
    (out/'camera_format.txt').write_text(fmt)
    for pattern in (r'Width/Height\s*:\s*1280/960',r"Pixel Format\s*:\s*'NV12'",
                    r'Bytes per Line\s*:\s*1280\b',r'Size Image\s*:\s*1843200\b'):
        if not re.search(pattern,fmt):
            raise RuntimeError('Unexpected camera format')
    stop,ended=threading.Event(),threading.Event()
    lock=threading.Lock()
    latest=[None,0,0.0]
    samples=[]
    state='starting'
    started=None
    with (out/'capture.log').open('wb') as log:
        proc=subprocess.Popen(['v4l2-ctl','-d',args.device,'--stream-mmap=4',
            '--stream-skip=10','--stream-count=0','--stream-to=-'],
            stdout=subprocess.PIPE,stderr=log,bufsize=0)
        def reader():
            try:
                while not stop.is_set():
                    data=bytearray()
                    while len(data)<1843200:
                        chunk=proc.stdout.read(1843200-len(data))
                        if not chunk:return
                        data.extend(chunk)
                    with lock:
                        latest[:]=[bytes(data),latest[1]+1,time.monotonic()]
            finally:ended.set()
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        try:
            print('저장 폴더:',out,flush=True)
            print('카메라가 준비되면 촬영을 시작합니다. 지금부터 물체를 움직여도 됩니다.',flush=True)
            wait_start=time.monotonic()
            while latest[1]==0:
                if ended.is_set() or time.monotonic()-wait_start>15:
                    raise RuntimeError('No camera frames; see capture.log')
                time.sleep(.01)
            started=time.monotonic()
            seen=0
            for index in range(args.count):
                deadline=started+index*args.interval
                if time.monotonic()<deadline:
                    time.sleep(deadline-time.monotonic())
                wait_start=time.monotonic()
                while True:
                    with lock:data,number,received=latest
                    if number!=seen and time.monotonic()-received<2:
                        break
                    if ended.is_set() or time.monotonic()-wait_start>5:
                        raise RuntimeError('No fresh camera frame')
                    time.sleep(.01)
                seen=number
                image=cv2.cvtColor(np.frombuffer(data,np.uint8).reshape(1440,1280),cv2.COLOR_YUV2BGR_NV12)
                filename=f'frame_{index+1:04d}.png'
                path=out/filename
                if not cv2.imwrite(str(path),image):
                    raise RuntimeError('PNG save failed: '+filename)
                samples.append(dict(frame_id=index+1,file=filename,capture_id=number,
                    host_receive_relative_ms=(received-started)*1000,
                    sample_relative_ms=(time.monotonic()-started)*1000,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                if index==0 or (index+1)%10==0 or index+1==args.count:
                    print(f'저장 {index+1}/{args.count}: {filename}',flush=True)
            # Retain the requested acquisition window through the final interval.
            remaining=started+args.count*args.interval-time.monotonic()
            if remaining>0:time.sleep(remaining)
            state='completed'
        except KeyboardInterrupt:
            state='interrupted'
        except Exception as error:
            state='failed: '+repr(error)
            raise
        finally:
            stop.set()
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
            thread.join(timeout=3)
            if proc.stdout:proc.stdout.close()
            manifest=dict(state=state,count=len(samples),requested_count=args.count,
                interval_seconds=args.interval,width=1280,height=960,color='BGR',
                format='lossless PNG after NV12 to BGR conversion; no resize or rotation',
                note='Identical saved input set for sequential A/B replay. Sampled camera stream, not every captured frame. Host receive times are not exposure times. File I/O and replay timing must be measured separately from live acquisition.',
                frames=samples)
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('완료:',state,'저장 이미지:',len(samples),flush=True)
    print('입력 세트:',out,flush=True)


if __name__=='__main__':
    main()
