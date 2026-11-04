"""Prune existing ONNX to dark3; build separate RKNN and compare on board.

All new model files go into dark3_only/. Existing full-backbone files untouched.
Uses the previously verified zidane input and PyTorch reference, without resizing.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dark3_only'
SAMPLE = ROOT / 'zidane_dark3_sample'
EXPECTED = (1, 256, 80, 80)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(actual, reference):
    if actual.shape != reference.shape or actual.shape != EXPECTED:
        raise RuntimeError(f'Unexpected shape: {actual.shape}, {reference.shape}')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise RuntimeError('Non-finite features')
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    return {'max_abs': float(np.max(np.abs(a-b))),
            'mean_abs': float(np.mean(np.abs(a-b))),
            'relative_l2': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)),
            'cosine': float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))}


def inputs():
    x = np.load(SAMPLE / 'input_bgr_uint8_nhwc.npy', allow_pickle=False)
    if x.shape != (1, 640, 640, 3) or x.dtype != np.uint8:
        raise RuntimeError('Expected verified uint8 NHWC BGR sample')
    return x


def export():
    import onnx
    from onnx.utils import extract_model
    import onnxruntime as ort
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'backbone.onnx'
    graph = onnx.load(str(source))
    if [v.name for v in graph.graph.input] != ['images']:
        raise RuntimeError('Unexpected ONNX input names')
    if [v.name for v in graph.graph.output] != ['dark3', 'dark4', 'dark5']:
        raise RuntimeError('Unexpected full-backbone output names')
    target = OUT / 'dark3.onnx'
    extract_model(str(source), str(target), ['images'], ['dark3'])
    pruned = onnx.load(str(target))
    onnx.checker.check_model(pruned)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    x = inputs().transpose(0,3,1,2).astype(np.float32)
    actual = session.run(None, {'images': x})[0]
    reference = np.load(SAMPLE / 'zidane_dark3_pytorch.npy', allow_pickle=False)
    errors = compare(actual, reference)
    # Export sanity gate, not a detection-accuracy threshold.
    if errors['relative_l2'] > 1e-3:
        raise RuntimeError(f'ONNX export reference mismatch: {errors}')
    report = {'source_onnx_sha256': sha(source), 'dark3_onnx_sha256': sha(target),
              'input_sha256': sha(SAMPLE / 'input_bgr_uint8_nhwc.npy'),
              'source_nodes': len(graph.graph.node), 'dark3_nodes': len(pruned.graph.node),
              'outputs': [v.name for v in pruned.graph.output], 'onnx_vs_pytorch': errors,
              'note': 'Exact reachable subgraph to original dark3 output; no new/random weights or activation substitutions.'}
    (OUT / 'export_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('DARK3 ONNX EXPORT CHECK OK', flush=True)


def build():
    from rknn.api import RKNN
    if not (OUT / 'export_report.json').exists():
        raise RuntimeError('Run export first')
    report = json.loads((OUT / 'export_report.json').read_text())
    if sha(OUT / 'dark3.onnx') != report['dark3_onnx_sha256']:
        raise RuntimeError('Exported ONNX changed; export again')
    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform='rk3588', mean_values=[[0,0,0]], std_values=[[1,1,1]])
        for name, operation in (
            ('load', lambda: rknn.load_onnx(model=str(OUT / 'dark3.onnx'))),
            ('build', lambda: rknn.build(do_quantization=False)),
            ('export', lambda: rknn.export_rknn(str(OUT / 'dark3_fp.rknn')))):
            if operation() != 0:
                raise RuntimeError(name + ' failed')
            print(name, 'OK', flush=True)
        (OUT / 'build_report.json').write_text(json.dumps({
            'onnx_sha256': sha(OUT / 'dark3.onnx'),
            'rknn_sha256': sha(OUT / 'dark3_fp.rknn')}, indent=2))
    finally:
        rknn.release()


def run_model(path, x, count, repeat):
    from rknnlite.api import RKNNLite
    runtime = RKNNLite()
    try:
        if runtime.load_rknn(str(path)) != 0:
            raise RuntimeError('Model load failed: ' + str(path))
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('NPU init failed')
        for _ in range(3):
            if runtime.inference(inputs=[x]) is None:
                raise RuntimeError('Warmup failed')
        times = []
        for _ in range(repeat):
            tick = perf_counter()
            outputs = runtime.inference(inputs=[x])
            times.append((perf_counter()-tick)*1000)
            if outputs is None or len(outputs) != count:
                raise RuntimeError('Unexpected outputs')
            if outputs[0].shape != EXPECTED or not np.isfinite(outputs[0]).all():
                raise RuntimeError('Invalid dark3')
        return outputs[0].copy(), times
    finally:
        runtime.release()


def test():
    x = inputs()
    original = ROOT / 'backbone_fp.rknn'
    candidate = OUT / 'dark3_fp.rknn'
    sample_manifest = json.loads((SAMPLE / 'manifest.json').read_text())
    if sha(original) != sample_manifest['rknn_sha256']:
        raise RuntimeError('Full-backbone RKNN differs from verified sample model')
    reference = np.load(SAMPLE / 'zidane_dark3_pytorch.npy', allow_pickle=False)
    results = {'full': [], 'dark3_only': []}
    features = {}
    # Sequential ABBA order reduces simple order bias; one NPU core, no concurrency.
    for name in ('full', 'dark3_only', 'dark3_only', 'full'):
        path, count = (original, 3) if name == 'full' else (candidate, 1)
        print('Benchmark:', name, flush=True)
        value, times = run_model(path, x, count, 10)
        results[name].extend(times)
        features[name] = value
    timing = {name: {'median': float(np.median(v)), 'mean': float(np.mean(v)),
                     'p95': float(np.percentile(v,95)), 'runs': len(v), 'raw_ms': v}
              for name,v in results.items()}
    comparisons = {'dark3_only_vs_pytorch': compare(features['dark3_only'], reference),
                   'full_vs_pytorch': compare(features['full'], reference),
                   'dark3_only_vs_full': compare(features['dark3_only'], features['full'])}
    np.save(OUT / 'zidane_dark3_npu.npy', features['dark3_only'])
    report = {'timing_api_ms': timing, 'comparisons': comparisons,
              'median_speedup': timing['full']['median']/timing['dark3_only']['median'],
              'full_rknn_sha256': sha(original), 'dark3_rknn_sha256': sha(candidate),
              'model_bytes': {'full': original.stat().st_size, 'dark3_only': candidate.stat().st_size},
              'conditions': 'Fixed same zidane input, core0, ABBA order, 3 warmups per block, 20 measurements per model. API time includes output transfer. No camera, FCM, network, head or mAP.'}
    (OUT / 'board_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('DARK3 BOARD TEST COMPLETE; review numerical differences before FCM integration.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'build', 'test'])
    args = parser.parse_args()
    {'export': export, 'build': build, 'test': test}[args.stage]()
