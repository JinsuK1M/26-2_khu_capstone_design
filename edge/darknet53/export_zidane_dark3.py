"""Save matched original PyTorch and existing board RKNN dark3 samples.

Run on Orange Pi inside ~/yolox_project/darknet53 with the current checkpoint.
Does not rebuild or replace the existing full-backbone RKNN model.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np
import torch
from rknnlite.api import RKNNLite


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    result = root / 'zidane_dark3_sample'
    result.mkdir(exist_ok=True)
    image_path = root / 'zidane.jpg'
    checkpoint_path = root / 'yolox_darknet.pth'
    model_path = root / 'backbone_fp.rknn'
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    ratio = min(640 / h, 640 / w)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    np.save(result / 'input_bgr_uint8_nhwc.npy', canvas[None])

    repo = root / 'YOLOX'
    package = types.ModuleType('official_darknet_sample')
    package.__path__ = [str(repo / 'yolox/models')]
    sys.modules[package.__name__] = package
    for name in ('network_blocks', 'darknet'):
        fullname = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(fullname, repo / 'yolox/models' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    torch.set_num_threads(4)
    backbone = sys.modules[package.__name__ + '.darknet'].Darknet(53).eval()
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    prefix = 'backbone.backbone.'
    weights = {k[len(prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(prefix)}
    backbone.load_state_dict(weights, strict=True)
    inputs = torch.from_numpy(canvas.transpose(2, 0, 1)[None].copy()).float()
    with torch.inference_mode():
        # Exact dark3 split: no dark4, dark5, neck or head in this reference path.
        dark3 = backbone.dark3(backbone.dark2(backbone.stem(inputs))).numpy()
    np.save(result / 'zidane_dark3_pytorch.npy', dark3)

    runtime = RKNNLite()
    try:
        if runtime.load_rknn(str(model_path)) != 0:
            raise RuntimeError('RKNN load failed')
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('RKNN init failed')
        outputs = runtime.inference(inputs=[canvas[None]])
        if outputs is None or len(outputs) != 3:
            raise RuntimeError('Unexpected RKNN output count')
        board = outputs[0]
        if board.shape != (1, 256, 80, 80) or not np.isfinite(board).all():
            raise RuntimeError('Invalid RKNN dark3 output')
        np.save(result / 'zidane_dark3_npu.npy', board)
    finally:
        runtime.release()
    a, b = dark3.astype(np.float64).ravel(), board.astype(np.float64).ravel()
    report = {
        'image_sha256': sha256(image_path),
        'checkpoint_sha256': sha256(checkpoint_path),
        'rknn_sha256': sha256(model_path),
        'yolox_revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__},
        'preprocessing': {'color': 'BGR', 'range': '0..255, no /255 or mean/std normalization',
                          'resize': 'INTER_LINEAR; ratio=min(640/h,640/w); int(w*ratio),int(h*ratio)',
                          'padding': 'bottom/right, constant114; image placed top-left'},
        'shape': list(board.shape), 'dtype': str(board.dtype),
        'cosine': float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
        'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)),
        'max_abs': float(np.max(np.abs(a - b))),
        'note': 'PyTorch reference runs only through dark3. Current RKNN still computes full backbone; first output is dark3. Not mAP.'
    }
    (result / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', result)


if __name__ == '__main__':
    main()
