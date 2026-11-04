"""Inspect/convert the original FENet ONNX, then compare CPU/NPU on RK3588.

Run alongside fcm_edge.py. No original model or live pipeline is overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fenet_npu_candidate'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f'출력 shape 불일치: {actual.shape} != {reference.shape}')
    assert np.isfinite(actual).all() and np.isfinite(reference).all(), 'NaN/Inf 출력'
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    return dict(max_abs=float(np.max(np.abs(a-b))), mean_abs=float(np.mean(np.abs(a-b))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
                cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)))


def samples():
    home = Path.home() / 'yolox_project/darknet53'
    paths = [ROOT/'test_l37.npy',
             home/'zidane_dark3_sample/zidane_dark3_npu.npy',
             home/'live_fcm_runs/20261103-155444-451889/last_dark3.npy']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        x = np.load(path,allow_pickle=False)
        assert x.shape == (1,256,80,80) and x.dtype == np.float32
        assert np.isfinite(x).all()
        yield path,np.ascontiguousarray(x)


def inspect():
    import onnx
    import onnxruntime as ort
    import torch
    import fcm_edge
    torch.set_num_threads(4)
    source = ROOT/'fenet_l37.onnx'
    graph = onnx.load(str(source))
    onnx.checker.check_model(graph)
    def describe(v):
        return dict(name=v.name,shape=[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim])
    report = dict(onnx_sha256=sha(source),torchscript_sha256=sha(ROOT/'fenet_l37.pt'),
                  inputs=[describe(v) for v in graph.graph.input],
                  outputs=[describe(v) for v in graph.graph.output],
                  operations=dict(Counter(n.op_type for n in graph.graph.node)),samples=[])
    assert [v['shape'] for v in report['inputs']] == [[1,256,80,80]]
    assert [v['shape'] for v in report['outputs']] == [[128,40,40]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    encoder = fcm_edge.load_encoder()
    with torch.no_grad():
        for path,x in samples():
            ref = encoder(torch.from_numpy(x)).numpy()
            actual = session.run(None,{session.get_inputs()[0].name:x})[0]
            errors = metrics(actual,ref)
            report['samples'].append(dict(path=str(path),sha256=sha(path),onnx_vs_torchscript=errors))
            # Export consistency check only; not a detection accuracy criterion.
            assert errors['relative_l2'] < 1e-3, f'기존 ONNX와 TorchScript 불일치: {errors}'
    OUT.mkdir(exist_ok=True)
    (OUT/'inspection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print('ONNX / TorchScript 입력·출력 비교 통과',flush=True)


def build():
    from rknn.api import RKNN
    report = json.loads((OUT/'inspection.json').read_text())
    source = ROOT/'fenet_l37.onnx'
    assert sha(source) == report['onnx_sha256'], 'ONNX 변경됨: inspect 재실행 필요'
    rknn = RKNN(verbose=True)
    try:
        # Float feature input: no image normalization or channel swapping.
        assert rknn.config(target_platform='rk3588') == 0, 'config 실패'
        assert rknn.load_onnx(model=str(source)) == 0, 'ONNX load 실패'
        assert rknn.build(do_quantization=False) == 0, 'RKNN build 실패'
        target = OUT/'fenet_fp.rknn'
        assert rknn.export_rknn(str(target)) == 0, 'RKNN export 실패'
        (OUT/'build.json').write_text(json.dumps(dict(onnx_sha256=sha(source),
            rknn_sha256=sha(target),quantization=False),indent=2))
        print('FENET RKNN BUILD COMPLETE:',target,flush=True)
        print('변환 완료이며 보드 출력·속도는 아직 미검증입니다.',flush=True)
    finally:
        rknn.release()


def test():
    import torch
    import fcm_edge
    from fcm_common import quantize_feature
    from rknnlite.api import RKNNLite
    torch.set_num_threads(4)
    build_report = json.loads((OUT/'build.json').read_text())
    model = OUT/'fenet_fp.rknn'
    assert sha(model) == build_report['rknn_sha256']
    encoder = fcm_edge.load_encoder()
    runtime = RKNNLite()
    report = dict(rknn_sha256=sha(model),torchscript_sha256=sha(ROOT/'fenet_l37.pt'),samples=[])
    try:
        assert runtime.load_rknn(str(model)) == 0
        assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
        def npu(x):
            outputs = runtime.inference(inputs=[x],data_format='nchw')
            assert outputs is not None and len(outputs)==1, 'NPU 출력 없음'
            y = outputs[0]
            if y.shape == (1,128,40,40):
                y = y[0]
            assert y.shape == (128,40,40), f'NPU 출력 shape 확인 필요: {y.shape}'
            assert np.isfinite(y).all()
            return np.ascontiguousarray(y,dtype=np.float32)
        with torch.no_grad():
            for path,x in samples():
                tx = torch.from_numpy(x)
                ref = encoder(tx).numpy()
                actual = npu(x)
                entry = dict(path=str(path),input_sha256=sha(path),npu_vs_cpu=metrics(actual,ref))
                q_cpu = quantize_feature(torch.from_numpy(ref))
                q_npu = quantize_feature(torch.from_numpy(actual))
                entry['tiled_quantized_frame'] = dict(
                    changed_fraction=float(np.mean(q_cpu!=q_npu)),
                    max_abs=float(np.max(np.abs(q_cpu.astype(np.float64)-q_npu))))
                report['samples'].append(entry)
                np.save(OUT/(path.parent.name+'_'+path.stem+'_fenet_npu.npy'),actual)
            # Same real-camera input; sequential CPU/NPU/NPU/CPU blocks.
            x = list(samples())[-1][1]
            tx = torch.from_numpy(x)
            timings = dict(cpu=[],npu=[])
            for name in ('cpu','npu','npu','cpu'):
                operation = (lambda: encoder(tx)) if name=='cpu' else (lambda:npu(x))
                for _ in range(3): operation()
                for _ in range(10):
                    start = perf_counter()
                    operation()
                    timings[name].append((perf_counter()-start)*1000)
            report['timing_ms'] = {name:dict(median=float(np.median(t)),
                p95=float(np.percentile(t,95)),raw_ms=t) for name,t in timings.items()}
        report['note'] = 'FENet only; NPU API includes input/output transfer. Non-INT8 build, not FP32 equivalence. No live pipeline/HM/decode/mAP acceptance.'
        (OUT/'board_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    finally:
        runtime.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['inspect','build','test'])
    args = parser.parse_args()
    dict(inspect=inspect,build=build,test=test)[args.stage]()
