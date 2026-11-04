"""Official YOLOX Darknet53 backbone: export, RKNN build, board comparison.

Run from ~/yolox_project/darknet53. Does not install the YOLOX training package.
The official Darknet module includes its SPP block; neck/head are excluded.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ('dark3', 'dark4', 'dark5')


def export():
    import torch
    torch.set_num_threads(4)
    repo = ROOT / 'YOLOX'
    package = types.ModuleType('official_darknet_modules')
    package.__path__ = [str(repo / 'yolox/models')]
    sys.modules[package.__name__] = package
    for name in ('network_blocks', 'darknet'):
        fullname = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(fullname, repo / 'yolox/models' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    backbone = sys.modules[package.__name__ + '.darknet'].Darknet(53).eval()
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    checkpoint_path = ROOT / 'yolox_darknet.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    prefix = 'backbone.backbone.'
    selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    backbone.load_state_dict(selected, strict=True)
    print('Official backbone weights loaded strictly:', len(selected), flush=True)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            result = self.backbone(x)
            return tuple(result[name] for name in NAMES)

    image_path = ROOT.parent / 'rknn_model_zoo/examples/yolox/model/bus.jpg'
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    # Official YOLOX default preprocessing: BGR 0..255, top-left letterbox.
    ratio = min(640 / image.shape[0], 640 / image.shape[1])
    resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    np.save(ROOT / 'input_bgr.npy', canvas[None])
    inputs = torch.from_numpy(canvas.transpose(2, 0, 1)[None].copy()).float()
    model = Wrapper().eval()
    with torch.inference_mode():
        references = model(inputs)
    np.savez(ROOT / 'reference.npz', **{k: v.numpy() for k, v in zip(NAMES, references)})
    for name, value in zip(NAMES, references):
        print(name, tuple(value.shape), flush=True)
    torch.onnx.export(model, inputs, str(ROOT / 'backbone.onnx'),
                      input_names=['images'], output_names=list(NAMES),
                      opset_version=12, do_constant_folding=True)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(ROOT / 'backbone.onnx'), options, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'images': inputs.numpy()})
    checks = {}
    for name, actual, expected in zip(NAMES, outputs, references):
        expected = expected.numpy()
        relative_l2 = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))
        checks[name] = {'relative_l2': relative_l2, 'max_abs': float(np.max(np.abs(actual - expected)))}
        if relative_l2 > 1e-3:
            raise RuntimeError(f'ONNX export mismatch: {name} {checks[name]}')
    manifest = {'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                'input': 'BGR uint8 NHWC -> float32 NCHW, 0..255, top-left pad114',
                'onnx_checks': checks, 'outputs': {k: list(v.shape) for k, v in zip(NAMES, references)}}
    (ROOT / 'export_report.json').write_text(json.dumps(manifest, indent=2))
    print('EXPORT AND ONNX COMPARISON PASSED', flush=True)


def build():
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])
        for name, operation in (
            ('load', lambda: rknn.load_onnx(model=str(ROOT / 'backbone.onnx'))),
            ('build', lambda: rknn.build(do_quantization=False)),
            ('export', lambda: rknn.export_rknn(str(ROOT / 'backbone_fp.rknn'))),
        ):
            result = operation()
            if result != 0:
                raise RuntimeError(f'{name} failed: {result}')
            print(name, 'OK', flush=True)
    finally:
        rknn.release()


def test():
    from rknnlite.api import RKNNLite
    inputs = np.load(ROOT / 'input_bgr.npy')
    reference = np.load(ROOT / 'reference.npz')
    rknn = RKNNLite()
    try:
        if rknn.load_rknn(str(ROOT / 'backbone_fp.rknn')) != 0:
            raise RuntimeError('Model load failed')
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('NPU init failed')
        for _ in range(3):
            if rknn.inference(inputs=[inputs]) is None:
                raise RuntimeError('Warmup inference failed')
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            outputs = rknn.inference(inputs=[inputs])
            timings.append((time.perf_counter() - start) * 1000)
            if outputs is None:
                raise RuntimeError('Inference failed')
        if len(outputs) != len(NAMES):
            raise RuntimeError(f'Unexpected output count: {len(outputs)}')
        report = {'inference_api_ms': {'median': float(np.median(timings)),
                                      'p95': float(np.percentile(timings, 95))}, 'features': {}}
        saved = {}
        for name, actual in zip(NAMES, outputs):
            expected = reference[name]
            if actual.shape != expected.shape:
                raise RuntimeError(f'{name}: unexpected shape {actual.shape}, expected {expected.shape}')
            if not np.isfinite(actual).all():
                raise RuntimeError(f'{name}: non-finite outputs')
            a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
            report['features'][name] = {
                'shape': list(actual.shape), 'dtype': str(actual.dtype),
                'cosine': float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
                'max_abs': float(np.max(np.abs(a - b))),
            }
            saved[name] = actual
        np.savez(ROOT / 'npu_features.npz', **saved)
        (ROOT / 'board_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print('BOARD BACKBONE EXECUTION COMPLETE; feature metrics are NOT detection mAP.')
    finally:
        rknn.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'build', 'test'])
    args = parser.parse_args()
    {'export': export, 'build': build, 'test': test}[args.stage]()
