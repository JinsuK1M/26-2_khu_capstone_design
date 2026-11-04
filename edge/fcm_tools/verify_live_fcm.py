"""Decode one saved real-camera FCM payload on board; no server/network/head."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from time import perf_counter
import types

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_header(blob):
    size = struct.calcsize('<ddHHI')
    if len(blob) <= size:
        raise ValueError('Payload too short')
    std, mean, height, width, n = struct.unpack_from('<ddHHI', blob)
    if (height, width) != (320, 640):
        raise ValueError(f'Unexpected tiled shape: {height}x{width}')
    if n <= 0 or len(blob) != size + n:
        raise ValueError('Bitstream length mismatch')
    if not np.isfinite(std) or not np.isfinite(mean) or std <= 0:
        raise ValueError('Invalid normalization statistics')
    return {'header_bytes': size, 'bitstream_bytes': n, 'std': std, 'mean': mean,
            'tiled_height': height, 'tiled_width': width}


def metrics(reference, restored):
    if reference.shape != (1, 256, 80, 80) or restored.shape != reference.shape:
        raise ValueError('Expected matching dark3 shape [1,256,80,80]')
    if reference.dtype != np.float32 or restored.dtype != np.float32:
        raise ValueError('Expected float32 dark3')
    if not np.isfinite(reference).all() or not np.isfinite(restored).all():
        raise ValueError('Non-finite feature values')
    a, b = reference.astype(np.float64).ravel(), restored.astype(np.float64).ravel()
    delta = b-a
    return {'mse': float(np.mean(delta*delta)), 'mean_abs': float(np.mean(np.abs(delta))),
            'max_abs': float(np.max(np.abs(delta))),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(a), 1e-12)),
            'cosine': float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, help='Specific live_fcm_runs directory; defaults to latest complete sample')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    home = Path.home()/'yolox_project'
    if args.run:
        run = args.run.expanduser().resolve()
    else:
        candidates = sorted((home/'darknet53/live_fcm_runs').glob('*/last_payload.bin'))
        if not candidates:
            raise FileNotFoundError('No saved live payload')
        run = candidates[-1].parent
    bundle = home/'fcm_l37/fcm_bundle'
    sys.path.insert(0,str(bundle))
    import fcm_server
    torch.set_num_threads(args.threads)
    blob = (run/'last_payload.bin').read_bytes()
    header = validate_header(blob)
    reference = np.load(run/'last_dark3.npy',allow_pickle=False)
    info = json.loads((run/'last_frame.json').read_text())
    if info['payload_bytes'] != len(blob):
        raise ValueError('Saved frame metadata and payload size disagree')
    clock = {}
    original_hm, original_subprocess = fcm_server._hm_decode, fcm_server.subprocess

    def timed_hm(bits,h,w):
        tick = perf_counter()
        value = original_hm(bits,h,w)
        clock['hm_decode_ms'] = (perf_counter()-tick)*1000
        return value

    def bounded_run(*pos, **kw):
        kw.setdefault('timeout',120)
        return original_subprocess.run(*pos,**kw)

    print('Sample:',run,flush=True)
    print(f'Frame={info["capture_id"]} QP={info["qp"]} bytes={len(blob)}',flush=True)
    tick = perf_counter()
    decoder = fcm_server.load_decoder(device='cpu')
    load_ms = (perf_counter()-tick)*1000

    class TimedDecoder:
        def __call__(self,x):
            tick = perf_counter()
            output = decoder(x)
            clock['drnet_ms'] = (perf_counter()-tick)*1000
            return output

    fcm_server._hm_decode = timed_hm
    fcm_server.subprocess = types.SimpleNamespace(run=bounded_run)
    try:
        tick = perf_counter()
        result = fcm_server.decompress(TimedDecoder(),blob)
        total_ms = (perf_counter()-tick)*1000
    finally:
        fcm_server._hm_decode, fcm_server.subprocess = original_hm, original_subprocess
    restored = result.detach().cpu().numpy()
    errors = metrics(reference,restored)
    np.save(run/'last_dark3_reconstructed.npy',restored)
    report = {'sample_directory':str(run), 'capture_id':info['capture_id'],'qp':info['qp'],
              'payload_bytes':len(blob), 'original_feature_bytes':reference.nbytes,
              'payload_sha256':sha(run/'last_payload.bin'),
              'original_feature_sha256':sha(run/'last_dark3.npy'),
              'decoder_sha256':sha(bundle/'drnet_l37.pt'),
              'shape':list(restored.shape),'dtype':str(restored.dtype),'header':header,
              'decoder_load_ms':load_ms,'decode_ms':{**clock,'total_ms':total_ms,
                  'other_ms':total_ms-sum(clock.values())},
              'comparison_to_precompression_npu_dark3':errors,
              'versions':{'torch':torch.__version__,'numpy':np.__version__},
              'threads':args.threads,
              'note':'One first-call decode on Orange Pi CPU, no warmup. Not Mac server performance or mAP. Metrics quantify lossy reconstruction vs original NPU dark3; do not compare to bundle reference-reproduction tolerance.'}
    (run/'decode_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print('LIVE PAYLOAD DECODE CHECK COMPLETE; detection accuracy remains unvalidated.',flush=True)


if __name__ == '__main__':
    main()
