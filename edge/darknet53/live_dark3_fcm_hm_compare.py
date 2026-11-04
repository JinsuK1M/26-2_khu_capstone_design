"""Camera -> dark3-only RKNN -> original FCM payload. No network or decoder.

Latest-frame policy; unique run directory; only last processed payload saved.
Receive timestamps are host pipe-read timestamps, NOT sensor exposure times.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import types

import cv2
import numpy as np
import torch
from rknnlite.api import RKNNLite


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_input(data):
    image = cv2.cvtColor(np.frombuffer(data, np.uint8).reshape(1440,1280), cv2.COLOR_YUV2BGR_NV12)
    canvas = np.full((640,640,3),114,dtype=np.uint8)
    canvas[:480] = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    return image, canvas[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=30,help='0: until Ctrl+C; checked between frames')
    parser.add_argument('--qp',type=int,choices=[20,25],default=20)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',default='/dev/video11')
    parser.add_argument('--hm-profile',choices=['baseline','combined4'],default='baseline')
    parser.add_argument('--run-dir',type=Path)
    args = parser.parse_args()
    if args.seconds < 0 or args.threads < 1:
        parser.error('Invalid duration or thread count')
    root = Path(__file__).resolve().parent
    bundle = Path.home()/'yolox_project/fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_edge
    torch.set_num_threads(args.threads)
    model = root/'dark3_only/dark3_fp.rknn'
    # Fail before opening camera if dependencies/models missing.
    hm = Path(fcm_edge.HM_DIR)
    for path in (model,bundle/'fenet_l37.pt',hm/'bin/TAppEncoderStatic',hm/'cfg/encoder_intra_main_rext.cfg'):
        if not path.is_file():
            raise FileNotFoundError(path)
    out = args.run_dir or root/'live_fcm_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    hm_options = [] if args.hm_profile=='baseline' else [
        '--RDOQ=0','--RDOQTS=0','--TransformSkip=0',
        '--TransformSkipFast=0','--QuadtreeTUMaxDepthIntra=1','--SAO=0']
    print('HM profile:',args.hm_profile,hm_options,flush=True)
    print('Run directory:',out,flush=True)
    subprocess.run(['v4l2-ctl','-d',args.device,'--set-fmt-video=width=1280,height=960,pixelformat=NV12'],check=True)
    fmt = subprocess.check_output(['v4l2-ctl','-d',args.device,'--get-fmt-video'],text=True)
    (out/'camera_format.txt').write_text(fmt)
    for pattern in (r'Width/Height\s*:\s*1280/960',r"Pixel Format\s*:\s*'NV12'",
                    r'Bytes per Line\s*:\s*1280\b',r'Size Image\s*:\s*1843200\b'):
        if not re.search(pattern,fmt):
            raise RuntimeError('Unexpected camera layout; see camera_format.txt')
    tick = time.perf_counter()
    encoder = fcm_edge.load_encoder(device='cpu')
    load_ms = (time.perf_counter()-tick)*1000
    clock = {}
    quantize, hm_encode, hm_subprocess = fcm_edge.quantize_feature, fcm_edge._hm_encode, fcm_edge.subprocess

    class TimedEncoder:
        def __call__(self,x):
            tick = time.perf_counter()
            y = encoder(x)
            clock['fenet_ms'] = (time.perf_counter()-tick)*1000
            if tuple(y.shape)!=(128,40,40) or not torch.isfinite(y).all() or float(y.max())<=float(y.min()):
                raise RuntimeError('Invalid or constant encoder output; cannot normalize')
            return y

    def timed_quantize(y):
        tick = time.perf_counter()
        value = quantize(y)
        clock['quantize_ms'] = (time.perf_counter()-tick)*1000
        return value

    def timed_hm(frame,qp):
        tick = time.perf_counter()
        value = hm_encode(frame,qp)
        clock['hm_ms'] = (time.perf_counter()-tick)*1000
        return value

    def bounded_hm_run(*pos,**kw):
        kw.setdefault('timeout',120)
        # Original encoder command remains intact; override only selected options.
        if hm_options:
            if pos:
                pos=(list(pos[0])+hm_options,)+pos[1:]
            else:
                kw['args']=list(kw['args'])+hm_options
        return hm_subprocess.run(*pos,**kw)

    # Only module references in memory change. Team source files stay untouched.
    fcm_edge.quantize_feature, fcm_edge._hm_encode = timed_quantize,timed_hm
    fcm_edge.subprocess = types.SimpleNamespace(run=bounded_hm_run)
    latest = [None,0,0.0]
    lock = threading.Lock()
    ended = threading.Event()
    runtime = RKNNLite()
    proc = None
    last = None
    rows = []
    samples = []
    state = 'starting'
    started = None
    try:
        if runtime.load_rknn(str(model)) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('NPU initialization failed')
        with (out/'capture.log').open('wb') as log, (out/'frames.csv').open('w',newline='') as csvfile:
            proc = subprocess.Popen(['v4l2-ctl','-d',args.device,'--stream-mmap=4','--stream-skip=10',
                                     '--stream-count=0','--stream-to=-'],stdout=subprocess.PIPE,stderr=log,bufsize=0)

            def reader():
                try:
                    while True:
                        data = bytearray()
                        while len(data)<1843200:
                            chunk=proc.stdout.read(1843200-len(data))
                            if not chunk:
                                return
                            data.extend(chunk)
                        with lock:
                            latest[0]=bytes(data)
                            latest[1]+=1
                            latest[2]=time.monotonic()
                finally:
                    ended.set()

            thread=threading.Thread(target=reader,daemon=True)
            thread.start()
            started=time.monotonic()
            seen=0
            writer=None
            print('LIVE CAMERA -> DARK3 -> FCM START; no server/network. Ctrl+C stops.',flush=True)
            while args.seconds==0 or time.monotonic()-started<args.seconds:
                with lock:
                    data,number,received=latest
                if number==seen:
                    if ended.is_set():
                        raise RuntimeError('Capture ended; see capture.log')
                    if time.monotonic()-(received or started)>15:
                        raise RuntimeError('No camera frames for 15s')
                    time.sleep(.005)
                    continue
                if time.monotonic()-received>15:
                    raise RuntimeError('Camera stopped providing fresh frames')
                skipped=number-seen-1
                seen=number
                begin=time.monotonic()
                tick=time.perf_counter()
                image,inputs=prepare_input(data)
                prep_ms=(time.perf_counter()-tick)*1000
                tick=time.perf_counter()
                outputs=runtime.inference(inputs=[inputs])
                npu_ms=(time.perf_counter()-tick)*1000
                if outputs is None or len(outputs)!=1 or outputs[0].shape!=(1,256,80,80):
                    raise RuntimeError('Expected dark3-only RKNN output')
                dark3=np.ascontiguousarray(outputs[0],dtype=np.float32)
                if not np.isfinite(dark3).all():
                    raise RuntimeError('Non-finite dark3')
                print(f'frame={number}: NPU complete; FCM encoding...',flush=True)
                clock.clear()
                tick=time.perf_counter()
                blob=fcm_edge.compress(TimedEncoder(),torch.from_numpy(dark3),qp=args.qp)
                compress_ms=(time.perf_counter()-tick)*1000
                done=time.monotonic()
                row={'processed':len(rows)+1,'capture_id':number,'skipped_before':skipped,
                     'preprocess_ms':prep_ms,'npu_ms':npu_ms,**clock,
                     'other_fcm_ms':compress_ms-sum(clock.values()),'compress_ms':compress_ms,
                     'selected_process_ms':(done-begin)*1000,
                     'host_receive_to_payload_ms':(done-received)*1000,
                     'payload_bytes':len(blob),'loop_avg_fps':(len(rows)+1)/(done-started)}
                if writer is None:
                    writer=csv.DictWriter(csvfile,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);csvfile.flush()
                rows.append(row)
                # Bounded snapshots; disk writes happen after camera measurement.
                if len(rows) in (1,10,20):
                    samples.append((dark3.copy(),blob,dict(row)))
                last=(image,inputs,dark3,blob,dict(row))
                print(f'processed={len(rows)} capture_id={number} skip={skipped} '
                      f'npu={npu_ms:.1f}ms FENet={clock["fenet_ms"]:.1f}ms '
                      f'HM={clock["hm_ms"]:.1f}ms FCM={compress_ms:.1f}ms '
                      f'payload={len(blob)/1024:.2f}KiB loop_avg={row["loop_avg_fps"]:.2f}fps',flush=True)
            state='completed'
    except KeyboardInterrupt:
        state='interrupted'
        print('\nStopped by user',flush=True)
    except Exception as error:
        state='failed: '+repr(error)
        raise
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
            if proc.stdout: proc.stdout.close()
        runtime.release()
        fcm_edge.quantize_feature, fcm_edge._hm_encode, fcm_edge.subprocess=quantize,hm_encode,hm_subprocess
        for sample_dark3,sample_blob,sample_info in samples:
            directory=out/'samples'/f"{sample_info['processed']:06d}"
            directory.mkdir(parents=True)
            np.save(directory/'last_dark3.npy',sample_dark3)
            (directory/'last_payload.bin').write_bytes(sample_blob)
            (directory/'last_frame.json').write_text(json.dumps(
                {**sample_info,'qp':args.qp,'hm_profile':args.hm_profile},indent=2))
        if last is not None:
            image,inputs,dark3,blob,info=last
            (out/'last_payload.bin').write_bytes(blob)
            np.save(out/'last_dark3.npy',dark3)
            np.save(out/'last_input_bgr.npy',inputs)
            if not cv2.imwrite(str(out/'last_frame.png'),image):
                print('WARNING: last_frame.png save failed',flush=True)
            (out/'last_frame.json').write_text(json.dumps({**info,'qp':args.qp,'hm_profile':args.hm_profile,
                'preprocessing':'BGR 0..255;1280x960->640x480 INTER_LINEAR;bottom pad114 to640x640'},indent=2))
        keys=('npu_ms','fenet_ms','quantize_ms','hm_ms','compress_ms','selected_process_ms','payload_bytes')
        # First-frame overhead retained separately; no automatic warmup frame discard.
        def summarize(values):
            return {k:{'mean':float(np.mean([r[k] for r in values])),
                       'median':float(np.median([r[k] for r in values])),
                       'p95':float(np.percentile([r[k] for r in values],95))} for k in keys} if values else {}
        report={'state':state,'qp':args.qp,'threads':args.threads,'processed':len(rows),
                'hm_profile':args.hm_profile,'hm_options':hm_options,
                'saved_sample_frames':[info['processed'] for _,_,info in samples],
                'received_host_frames':latest[1],'skipped_before_selected':sum(r['skipped_before'] for r in rows),
                'remaining_unprocessed_host_frames':latest[1]-rows[-1]['capture_id'] if rows else latest[1],
                'encoder_load_ms':load_ms,'rknn_sha256':sha(model),'encoder_sha256':sha(bundle/'fenet_l37.pt'),
                'all_frames':summarize(rows),'after_first_frame':summarize(rows[1:]),
                'first_frame':rows[0] if rows else None,
                'last_loop_avg_fps':rows[-1]['loop_avg_fps'] if rows else 0,
                'notes':'Same stream, latest-frame policy. Host pipe receive timestamps are not sensor exposure timestamps. No network/server/decoder/head/mAP/preview. HM includes subprocess and file I/O. Captured frames during encoding may be skipped.'}
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('SUMMARY:',json.dumps({'profile':args.hm_profile,
              'fps':report['last_loop_avg_fps'],
              'after_first_frame':report['after_first_frame']},indent=2),flush=True)
        print('Saved run:',out,flush=True)
        print('Completed payloads:',len(rows),'state:',state,flush=True)


if __name__=='__main__':
    main()
