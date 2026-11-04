"""Time original FCM compress() stages on CPU without editing bundle code.

First run, warmups and measured repetitions are reported separately.
HM stage includes subprocess startup, temporary files and bitstream readback.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def temperatures():
    values = {}
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            kind = (path.parent / 'type').read_text().strip()
            values[path.parent.name + ':' + kind] = float(path.read_text()) / 1000
        except (OSError, ValueError):
            pass
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path.home() / 'yolox_project/darknet53/zidane_dark3_sample/zidane_dark3_npu.npy')
    parser.add_argument('--bundle', type=Path, default=Path.home() / 'yolox_project/fcm_l37/fcm_bundle')
    parser.add_argument('--output', type=Path, default=Path.home() / 'yolox_project/fcm_l37/encode_benchmark')
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--qp', type=int, choices=[20, 25], default=20)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('Invalid repeat/warmup/threads')
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle))
    import fcm_edge
    torch.set_num_threads(args.threads)
    array = np.load(args.input, allow_pickle=False)
    if array.shape != (1, 256, 80, 80) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError('Expected finite float32 NCHW dark3 [1,256,80,80]')
    x = torch.from_numpy(np.ascontiguousarray(array))
    args.output.mkdir(parents=True, exist_ok=True)
    original_quantize = fcm_edge.quantize_feature
    original_hm = fcm_edge._hm_encode
    clock = {}

    start = perf_counter()
    encoder = fcm_edge.load_encoder(device='cpu')
    model_load_ms = (perf_counter() - start) * 1000

    class TimedEncoder:
        def __call__(self, value):
            start = perf_counter()
            y = encoder(value)
            clock['fenet_ms'] = (perf_counter() - start) * 1000
            if tuple(y.shape) != (128, 40, 40) or not torch.isfinite(y).all():
                raise RuntimeError('Invalid FENet output')
            # Validation is outside FENet timing, in other_ms.
            return y

    def timed_quantize(y):
        start = perf_counter()
        value = original_quantize(y)
        clock['quantize_tile_ms'] = (perf_counter() - start) * 1000
        return value

    def timed_hm(frame, qp):
        start = perf_counter()
        bits = original_hm(frame, qp)
        clock['hm_ms'] = (perf_counter() - start) * 1000
        return bits

    metadata = {
        'platform': platform.platform(), 'machine': platform.machine(),
        'torch': torch.__version__, 'numpy': np.__version__,
        'torch_threads': torch.get_num_threads(), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'qp': args.qp, 'input': str(args.input.resolve()), 'input_sha256': digest(args.input),
        'encoder_sha256': digest(bundle / 'fenet_l37.pt'),
        'edge_code_sha256': digest(bundle / 'fcm_edge.py'),
        'common_code_sha256': digest(bundle / 'fcm_common.py'),
        'hm_binary_sha256': digest(Path(fcm_edge.HM_DIR) / 'bin/TAppEncoderStatic'),
        'hm_config_sha256': digest(Path(fcm_edge.HM_DIR) / 'cfg/encoder_intra_main_rext.cfg'),
        'model_load_ms': model_load_ms, 'temperature_before_C': temperatures(),
        'scope': 'CPU FENet + quantization/tiling + original HM subprocess + header. No camera/backbone/network/decoder/head.',
        'hm_scope': 'Temporary files, YUV write, process startup, HM encoding and bitstream readback.',
        'other_scope': 'Header mean/std/packing, validation and instrumentation overhead.',
        'input_policy': 'Same saved image repeated; not diverse frames or sustained performance.',
    }
    print(f'Model load: {model_load_ms:.1f} ms', flush=True)
    rows = []
    first_blob = None
    fcm_edge.quantize_feature, fcm_edge._hm_encode = timed_quantize, timed_hm
    try:
        for i in range(1 + args.warmup + args.repeat):
            phase = 'first' if i == 0 else ('warmup' if i <= args.warmup else 'measured')
            clock.clear()
            start = perf_counter()
            blob = fcm_edge.compress(TimedEncoder(), x, qp=args.qp)
            total = (perf_counter() - start) * 1000
            row = {'phase': phase, 'index': i, 'payload_bytes': len(blob),
                   **clock, 'other_ms': total - sum(clock.values()), 'total_ms': total}
            rows.append(row)
            if i == 0:
                first_blob = blob
                (args.output / 'first_payload.bin').write_bytes(blob)
            print(f'{phase:8s} {i:02d}: FENet={clock["fenet_ms"]:.1f} '
                  f'quant/tile={clock["quantize_tile_ms"]:.1f} HM={clock["hm_ms"]:.1f} '
                  f'other={row["other_ms"]:.1f} total={total:.1f} ms bytes={len(blob)}', flush=True)
            with (args.output / 'runs.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    finally:
        fcm_edge.quantize_feature, fcm_edge._hm_encode = original_quantize, original_hm

    # Verify that instrumentation preserved the original compression output.
    baseline = fcm_edge.compress(encoder, x, qp=args.qp)
    metadata['instrumented_payload_equals_original'] = baseline == first_blob
    if baseline != first_blob:
        raise RuntimeError('Payload changed between instrumented and original calls; investigate determinism')
    measured = [r for r in rows if r['phase'] == 'measured']
    summary = {}
    for key in ('fenet_ms', 'quantize_tile_ms', 'hm_ms', 'other_ms', 'total_ms'):
        values = [r[key] for r in measured]
        summary[key] = {'median': float(np.median(values)), 'mean': float(np.mean(values)),
                        'p95': float(np.percentile(values, 95)), 'min': float(np.min(values)),
                        'max': float(np.max(values))}
    report = {'metadata': metadata, 'first_run': rows[0], 'measured_count': len(measured),
              'summary_ms': summary, 'payload_bytes': [r['payload_bytes'] for r in measured],
              'temperature_after_C': temperatures()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nMEASURED SUMMARY (ms):', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print('Payload equivalence to original: OK', flush=True)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
