"""Same dark3 -> CPU/NPU FENet -> combined4 HM -> original decoder.

Board-only diagnostic; no original source/model or live settings are changed.
"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
from time import perf_counter
import types

import numpy as np
import torch
from rknnlite.api import RKNNLite
import fcm_edge
import fcm_server
from fenet_npu_candidate import samples, metrics, sha


OPTIONS = ['--RDOQ=0', '--RDOQTS=0', '--TransformSkip=0',
           '--TransformSkipFast=0', '--QuadtreeTUMaxDepthIntra=1', '--SAO=0']


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    model = root/'fenet_npu_candidate/fenet_fp.rknn'
    build = json.loads((model.parent/'build.json').read_text())
    assert sha(model) == build['rknn_sha256'], 'RKNN 모델 해시 불일치'
    out = root/'fenet_roundtrip_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    encoder, decoder = fcm_edge.load_encoder(), fcm_server.load_decoder()
    runtime = RKNNLite()
    original_edge_subprocess, original_server_subprocess = fcm_edge.subprocess, fcm_server.subprocess

    def run_encoder(command, **kwargs):
        kwargs.setdefault('timeout',120)
        return original_edge_subprocess.run(list(command)+OPTIONS, **kwargs)

    def run_decoder(command, **kwargs):
        kwargs.setdefault('timeout',120)
        return original_server_subprocess.run(command, **kwargs)

    fcm_edge.subprocess = types.SimpleNamespace(run=run_encoder)
    fcm_server.subprocess = types.SimpleNamespace(run=run_decoder)
    report = dict(qp=20,hm_profile='combined4',hm_options=OPTIONS,threads=4,
                  rknn_sha256=sha(model),encoder_sha256=sha(root/'fenet_l37.pt'),
                  decoder_sha256=sha(root/'drnet_l37.pt'),samples=[],
                  hm_config_sha256=sha(Path(fcm_edge.HM_DIR)/'cfg/encoder_intra_main_rext.cfg'),
                  note='Fixed identical inputs. Compression includes FENet, quantization, HM and I/O. No camera/network/head/mAP. Decode on board CPU for verification only.')
    try:
        assert runtime.load_rknn(str(model)) == 0
        assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
        for index,(source,x) in enumerate(samples()):
            directory = out/f'sample_{index}'
            directory.mkdir()
            tx = torch.from_numpy(x)
            latent = {}
            def cpu(value):
                y = encoder(value)
                assert tuple(y.shape) == (128,40,40) and torch.isfinite(y).all()
                assert float(y.max()) > float(y.min()), 'CPU FENet 상수 출력'
                latent['cpu'] = y.detach().numpy().copy()
                return y
            def npu(value):
                inputs = np.ascontiguousarray(value.detach().numpy(),dtype=np.float32)
                outputs = runtime.inference(inputs=[inputs],data_format='nchw')
                assert outputs is not None and len(outputs) == 1
                y = outputs[0]
                if y.shape == (1,128,40,40):
                    y = y[0]
                assert y.shape == (128,40,40) and np.isfinite(y).all()
                assert float(y.max()) > float(y.min()), 'NPU FENet 상수 출력'
                latent['npu'] = np.ascontiguousarray(y,dtype=np.float32)
                return torch.from_numpy(latent['npu'])
            timings = dict(cpu=[],npu=[])
            payloads = {}
            functions = dict(cpu=cpu,npu=npu)
            print(f'\n입력 {index+1}/3: {source}',flush=True)
            # One complete warmup per path. Alternate order of five measured pairs.
            for name in ('cpu','npu'):
                fcm_edge.compress(functions[name],tx,qp=20)
            for repeat in range(5):
                order = ('cpu','npu') if repeat%2 == 0 else ('npu','cpu')
                for name in order:
                    start = perf_counter()
                    payloads[name] = fcm_edge.compress(functions[name],tx,qp=20)
                    elapsed = (perf_counter()-start)*1000
                    timings[name].append(elapsed)
                    print(f'{repeat+1}/5 {name}: {elapsed:.1f}ms, {len(payloads[name])}bytes',flush=True)
            entry = dict(source=str(source),input_sha256=sha(source),
                         latent_npu_vs_cpu=metrics(latent['npu'],latent['cpu']),paths={})
            reconstructions = {}
            for name in ('cpu','npu'):
                blob = payloads[name]
                (directory/f'{name}_payload.bin').write_bytes(blob)
                start = perf_counter()
                restored = fcm_server.decompress(decoder,blob).detach().numpy()
                decode_ms = (perf_counter()-start)*1000
                assert restored.shape == x.shape and restored.dtype == np.float32
                assert np.isfinite(restored).all()
                reconstructions[name] = restored.copy()
                np.save(directory/f'{name}_reconstructed.npy',restored)
                entry['paths'][name] = dict(compress_median_ms=float(np.median(timings[name])),
                    compress_p95_ms=float(np.percentile(timings[name],95)),
                    raw_compress_ms=timings[name],payload_bytes=len(blob),
                    decode_ms=decode_ms,reconstruction_vs_original=metrics(restored,x))
            entry['reconstructed_npu_vs_cpu'] = metrics(reconstructions['npu'],reconstructions['cpu'])
            report['samples'].append(entry)
            (out/'report.json').write_text(json.dumps(report,indent=2))
        print('\n=== FENET CPU/NPU ROUNDTRIP COMPARISON ===',flush=True)
        for entry in report['samples']:
            print(json.dumps(entry,indent=2),flush=True)
        print('저장:',out,flush=True)
        print('압축·복원 검사 완료. 라이브 속도 및 mAP 검증은 아직입니다.',flush=True)
    finally:
        fcm_edge.subprocess, fcm_server.subprocess = original_edge_subprocess, original_server_subprocess
        runtime.release()


if __name__ == '__main__':
    with torch.no_grad():
        main()
