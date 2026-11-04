"""Bounded two-stage pipeline: backbone/FENet NPU overlaps one CPU HM encoder.

One ready-job slot with backpressure; latest-camera-frame policy. No server.
Queue waiting is included in selected_process_ms, never hidden as service time.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
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
from live_dark3_fcm_fenet_compare import prepare_input, call_fenet_npu, sha


OPTIONS = ['--RDOQ=0','--RDOQTS=0','--TransformSkip=0',
           '--TransformSkipFast=0','--QuadtreeTUMaxDepthIntra=1','--SAO=0']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--device',default='/dev/video11')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    root = Path(__file__).resolve().parent
    bundle = Path.home()/'yolox_project/fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_edge
    torch.set_num_threads(4)
    model = root/'dark3_only/dark3_fp.rknn'
    fenet_model = bundle/'fenet_npu_candidate/fenet_fp.rknn'
    build = json.loads((fenet_model.parent/'build.json').read_text())
    assert sha(fenet_model)==build['rknn_sha256'], 'FENet model hash mismatch'
    for p in (model,Path(fcm_edge.HM_DIR)/'bin/TAppEncoderStatic',
              Path(fcm_edge.HM_DIR)/'cfg/encoder_intra_main_rext.cfg'):
        assert p.is_file(), str(p)
    out = args.run_dir or root/'live_pipeline_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    subprocess.run(['v4l2-ctl','-d',args.device,
                    '--set-fmt-video=width=1280,height=960,pixelformat=NV12'],check=True)
    fmt = subprocess.check_output(['v4l2-ctl','-d',args.device,'--get-fmt-video'],text=True)
    (out/'camera_format.txt').write_text(fmt)
    for pattern in (r'Width/Height\s*:\s*1280/960',r"Pixel Format\s*:\s*'NV12'",
                    r'Bytes per Line\s*:\s*1280\b',r'Size Image\s*:\s*1843200\b'):
        assert re.search(pattern,fmt), 'Unexpected camera format'
    clock = {}
    original_quant, original_hm, original_subprocess = fcm_edge.quantize_feature,fcm_edge._hm_encode,fcm_edge.subprocess
    def quantize(y):
        start=time.perf_counter()
        result=original_quant(y)
        clock['quantize_ms']=(time.perf_counter()-start)*1000
        return result
    def encode(frame,qp):
        start=time.perf_counter()
        result=original_hm(frame,qp)
        clock['hm_ms']=(time.perf_counter()-start)*1000
        return result
    def run_hm(command,**kwargs):
        kwargs.setdefault('timeout',120)
        return original_subprocess.run(list(command)+OPTIONS,**kwargs)
    fcm_edge.quantize_feature,fcm_edge._hm_encode = quantize,encode
    fcm_edge.subprocess = types.SimpleNamespace(run=run_hm)
    stop,ready,start_event,capture_ended = [threading.Event() for _ in range(4)]
    errors = queue.Queue()
    jobs = queue.Queue(maxsize=1)
    slot = threading.Semaphore(1)
    lock = threading.Lock()
    latest = [None,0,0.0]
    started = [None]
    produced = [0]
    proc = None
    rows,snapshots = [],[]
    last=None
    worker=None
    reader_thread=None
    state='starting'

    def produce():
        backbone,fenet = RKNNLite(),RKNNLite()
        seen=0
        try:
            if backbone.load_rknn(str(model))!=0 or backbone.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
                raise RuntimeError('Backbone NPU initialization failed')
            if fenet.load_rknn(str(fenet_model))!=0 or fenet.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
                raise RuntimeError('FENet NPU initialization failed')
            ready.set()
            while not stop.is_set() and not start_event.wait(.05):
                pass
            with torch.no_grad():
                while not stop.is_set():
                    if not slot.acquire(timeout=.05):
                        continue
                    if stop.is_set() or time.monotonic()-started[0]>=args.seconds:
                        slot.release()
                        break
                    # Reserve the one output slot before selecting a camera frame.
                    while not stop.is_set():
                        with lock:
                            data,number,received=latest
                        if number!=seen:
                            break
                        if capture_ended.is_set() or time.monotonic()-(received or started[0])>15:
                            raise RuntimeError('Camera stopped; see capture.log')
                        time.sleep(.005)
                    if stop.is_set():
                        slot.release()
                        break
                    if time.monotonic()-received>15:
                        raise RuntimeError('Stale camera stream')
                    skipped=number-seen-1
                    seen=number
                    begin=time.monotonic()
                    tick=time.perf_counter()
                    image,inputs=prepare_input(data)
                    prep_ms=(time.perf_counter()-tick)*1000
                    tick=time.perf_counter()
                    values=backbone.inference(inputs=[inputs])
                    npu_ms=(time.perf_counter()-tick)*1000
                    if values is None or len(values)!=1 or values[0].shape!=(1,256,80,80):
                        raise RuntimeError('Invalid backbone output')
                    dark3=np.array(values[0],dtype=np.float32,order='C',copy=True)
                    if not np.isfinite(dark3).all():
                        raise RuntimeError('Non-finite dark3')
                    tick=time.perf_counter()
                    y=call_fenet_npu(fenet,torch.from_numpy(dark3)).clone()
                    fenet_ms=(time.perf_counter()-tick)*1000
                    if not torch.isfinite(y).all() or float(y.max())<=float(y.min()):
                        raise RuntimeError('Invalid FENet output')
                    job=dict(image=image,inputs=inputs,dark3=dark3,y=y,
                             capture_id=number,skipped_before=skipped,received=received,
                             begin=begin,preprocess_ms=prep_ms,npu_ms=npu_ms,fenet_ms=fenet_ms)
                    job['ready_at']=time.monotonic()
                    jobs.put_nowait(job)
                    produced[0]+=1
        except Exception as exc:
            errors.put(repr(exc))
            ready.set()
        finally:
            backbone.release()
            fenet.release()

    try:
        worker=threading.Thread(target=produce,name='npu-producer',daemon=True)
        worker.start()
        if not ready.wait(60):
            raise RuntimeError('NPU initialization timeout')
        if not errors.empty():
            raise RuntimeError(errors.get())
        with (out/'capture.log').open('wb') as log,(out/'frames.csv').open('w',newline='') as stream:
            proc=subprocess.Popen(['v4l2-ctl','-d',args.device,'--stream-mmap=4',
                '--stream-skip=10','--stream-count=0','--stream-to=-'],
                stdout=subprocess.PIPE,stderr=log,bufsize=0)
            def read_camera():
                try:
                    while not stop.is_set():
                        data=bytearray()
                        while len(data)<1843200:
                            chunk=proc.stdout.read(1843200-len(data))
                            if not chunk:
                                return
                            data.extend(chunk)
                        with lock:
                            latest[:]=[bytes(data),latest[1]+1,time.monotonic()]
                except Exception as exc:
                    if not stop.is_set(): errors.put(repr(exc))
                finally:
                    capture_ended.set()
            started[0]=time.monotonic()
            reader_thread=threading.Thread(target=read_camera,daemon=True)
            reader_thread.start()
            start_event.set()
            writer=None
            print('PIPELINE: sequential backbone/FENet NPU overlaps one CPU HM; ready queue capacity=1',flush=True)
            while time.monotonic()-started[0]<args.seconds:
                if not errors.empty():
                    raise RuntimeError(errors.get())
                try:
                    job=jobs.get(timeout=.1)
                except queue.Empty:
                    if not worker.is_alive():
                        if not errors.empty(): raise RuntimeError(errors.get())
                        break
                    continue
                slot.release()
                hm_start=time.monotonic()
                clock.clear()
                tick=time.perf_counter()
                # FENet was already evaluated in the producer; original codec/header retained.
                blob=fcm_edge.compress(lambda _:job['y'],torch.from_numpy(job['dark3']),qp=20)
                encode_ms=(time.perf_counter()-tick)*1000
                done=time.monotonic()
                row=dict(processed=len(rows)+1,capture_id=job['capture_id'],
                    skipped_before=job['skipped_before'],preprocess_ms=job['preprocess_ms'],
                    npu_ms=job['npu_ms'],fenet_ms=job['fenet_ms'],**clock,
                    hm_stage_ms=encode_ms,queue_wait_ms=(hm_start-job['ready_at'])*1000,
                    selected_process_ms=(done-job['begin'])*1000,
                    host_receive_to_payload_ms=(done-job['received'])*1000,
                    payload_bytes=len(blob),loop_avg_fps=(len(rows)+1)/(done-started[0]))
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush()
                rows.append(row)
                last=(job,blob,dict(row))
                if len(rows) in (1,10,20):
                    snapshots.append((job['dark3'].copy(),blob,dict(row)))
                print(f"processed={len(rows)} frame={row['capture_id']} "
                      f"queue={row['queue_wait_ms']:.1f}ms selected={row['selected_process_ms']:.1f}ms "
                      f"HM={row['hm_ms']:.1f}ms fps={row['loop_avg_fps']:.2f}",flush=True)
            state='completed'
    except KeyboardInterrupt:
        state='interrupted'
    except Exception as exc:
        state='failed: '+repr(exc)
        raise
    finally:
        stop.set();start_event.set()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
        if reader_thread is not None: reader_thread.join(timeout=3)
        if proc is not None and proc.stdout: proc.stdout.close()
        if worker is not None:
            worker.join(timeout=30)
            if worker.is_alive(): state='failed: NPU worker shutdown timeout'
        if not errors.empty(): state='failed: '+errors.get()
        fcm_edge.quantize_feature,fcm_edge._hm_encode,fcm_edge.subprocess=original_quant,original_hm,original_subprocess
        def save_sample(directory,dark3,blob,info):
            directory.mkdir(parents=True,exist_ok=True)
            np.save(directory/'last_dark3.npy',dark3)
            (directory/'last_payload.bin').write_bytes(blob)
            (directory/'last_frame.json').write_text(json.dumps({**info,'qp':20,
                'hm_profile':'combined4','fenet_backend':'npu','pipeline':True},indent=2))
        for dark3,blob,info in snapshots:
            save_sample(out/'samples'/f"{info['processed']:06d}",dark3,blob,info)
        if last is not None:
            job,blob,info=last
            save_sample(out,job['dark3'],blob,info)
            np.save(out/'last_input_bgr.npy',job['inputs'])
            if not cv2.imwrite(str(out/'last_frame.png'),job['image']):
                print('WARNING: last image save failed',flush=True)
        keys=('npu_ms','fenet_ms','hm_ms','hm_stage_ms','queue_wait_ms',
              'selected_process_ms','host_receive_to_payload_ms','payload_bytes')
        def summarize(values):
            return {k:dict(mean=float(np.mean([r[k] for r in values])),
                           median=float(np.median([r[k] for r in values])),
                           p95=float(np.percentile([r[k] for r in values],95)))
                    for k in keys} if values else {}
        report=dict(state=state,pipeline=True,fenet_backend='npu',hm_profile='combined4',qp=20,threads=4,
            processed=len(rows),produced=produced[0],unfinished_jobs=produced[0]-len(rows),
            received_host_frames=latest[1],skipped_before_selected=sum(r['skipped_before'] for r in rows),
            remaining_unprocessed_host_frames=latest[1]-rows[-1]['capture_id'] if rows else latest[1],
            last_loop_avg_fps=rows[-1]['loop_avg_fps'] if rows else 0,
            all_frames=summarize(rows),after_first_frame=summarize(rows[1:]),first_frame=rows[0] if rows else None,
            rknn_sha256=sha(model),fenet_rknn_sha256=sha(fenet_model),
            hm_options=OPTIONS,notes='Latest camera, one ready-job slot/backpressure. No intermediate feature overwrites. May leave unfinished jobs at stop. Selected latency includes queue waiting; timestamps are host based, not sensor E2E. No network/head/mAP. Snapshots written after measurement.')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('Saved run:',out,'state:',state,flush=True)
        if state!='completed':
            raise RuntimeError(state)


if __name__=='__main__':
    main()
