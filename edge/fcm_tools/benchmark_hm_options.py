"""Run from fcm_bundle; preserve all original codec/model settings and files."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
from datetime import datetime
from time import perf_counter

import numpy as np
import torch
import fcm_edge
import fcm_server
from fcm_common import quantize_feature, dequantize_frame


def metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a - b
    return dict(mse=float(np.mean(diff ** 2)),
                relative_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
                cosine=float(np.vdot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)))


def main():
    torch.set_num_threads(4)
    hm = Path(os.environ.get('FCM_HM_DIR', 'hm')).resolve()
    cfg = hm / 'cfg/encoder_intra_main_rext.cfg'
    encoder_exe = hm / 'bin/TAppEncoderStatic'
    decoder_exe = hm / 'bin/TAppDecoderStatic'
    for path in (cfg, encoder_exe, decoder_exe):
        assert path.is_file(), f'파일이 없습니다: {path}'
    source = Path(os.environ.get('FCM_BENCH_INPUT', 'test_l37.npy')).resolve()
    x = np.load(source, allow_pickle=False)
    assert x.shape == (1, 256, 80, 80) and x.dtype == np.float32
    assert np.isfinite(x).all()
    out = Path('hm_option_runs') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    enc, dec = fcm_edge.load_encoder(), fcm_server.load_decoder()
    with torch.no_grad():
        y = enc(torch.from_numpy(x))
    assert torch.isfinite(y).all() and float(y.max() - y.min()) > 0
    frame = quantize_feature(y)
    h, w = frame.shape
    rdoq = ['--RDOQ=0', '--RDOQTS=0']
    skip = ['--TransformSkip=0', '--TransformSkipFast=0']
    depth = ['--QuadtreeTUMaxDepthIntra=1']
    sao = ['--SAO=0']
    variants = dict(baseline=[], rdoq_off=rdoq, transformskip_off=skip,
                    tu_depth1=depth, sao_off=sao,
                    combined3=rdoq + skip + depth,
                    combined4=rdoq + skip + depth + sao)
    payloads, rows, errors = {}, [], {}

    def encode(name, suffix):
        start = perf_counter()
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            input_path, bit_path = temp / 'in.yuv', temp / 'out.bin'
            frame.astype('<u2').tofile(input_path)
            cmd = [str(encoder_exe), '-c', str(cfg), '--Level=6.2',
                   '--InputBitDepth=10', '--InternalBitDepth=10',
                   '--InputChromaFormat=400', '--ConformanceWindowMode=1',
                   '-i', str(input_path), '-f', '1', '-fr', '1',
                   '-wdt', str(w), '-hgt', str(h), '-q', '20',
                   '-b', str(bit_path)] + variants[name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode:
                (out / f'{name}_{suffix}.log').write_text(result.stdout + result.stderr)
                raise RuntimeError(f'HM returncode={result.returncode}; 로그 확인')
            bits = bit_path.read_bytes()
        elapsed = (perf_counter() - start) * 1000
        (out / f'{name}_{suffix}.log').write_text(result.stdout + result.stderr)
        blob = struct.pack(fcm_edge.HDR, float(y.std()), float(y.mean()), h, w, len(bits)) + bits
        return elapsed, blob

    print(f'입력: {source}\n출력: {out.resolve()}', flush=True)
    print('QP=20, FENet은 한 번만 실행. HM 시간은 임시파일 I/O와 프로세스 실행 포함.', flush=True)
    encode('baseline', 'warmup')
    rng = random.Random(20261103)
    for repeat in range(5):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            if name in errors:
                continue
            try:
                elapsed, blob = encode(name, str(repeat + 1))
                if name in payloads:
                    assert blob == payloads[name], '동일 조건의 비트스트림이 달라짐'
                payloads[name] = blob
                rows.append(dict(case=name, repeat=repeat + 1,
                                 hm_ms=elapsed, payload_bytes=len(blob)))
                print(f'{repeat + 1}/5 {name}: {elapsed:.1f}ms, {len(blob)}bytes', flush=True)
            except Exception as exc:
                errors[name] = str(exc)
                print(f'{name} 실패: {exc}', flush=True)
    assert 'baseline' in payloads and 'baseline' not in errors, '기준 인코딩 실패'

    def decode(blob):
        std, mean, height, width, size = struct.unpack_from(fcm_server.HDR, blob)
        off = struct.calcsize(fcm_server.HDR)
        assert len(blob) == off + size
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            bit_path, yuv_path = temp / 'in.bin', temp / 'out.yuv'
            bit_path.write_bytes(blob[off:])
            result = subprocess.run([str(decoder_exe), '-b', str(bit_path),
                                     '-o', str(yuv_path)], capture_output=True,
                                    text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stdout[-1000:] + result.stderr[-1000:])
            array = np.fromfile(yuv_path, dtype='<u2')
            assert array.size == height * width
        feature = dequantize_frame(array.reshape(height, width).astype(int),
                                   128, 40, 40, std, mean)
        with torch.no_grad():
            recon = dec(feature).numpy()
        assert recon.shape == x.shape and np.isfinite(recon).all()
        return recon

    baseline_recon = decode(payloads['baseline'])
    report = dict(input=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  qp=20, repeats=5, threads=4, options=variants, errors=errors, results={})
    print('\n=== 비교 결과 ===', flush=True)
    for name, blob in payloads.items():
        if name in errors:
            continue
        (out / f'{name}.bin').write_bytes(blob)
        times = [row['hm_ms'] for row in rows if row['case'] == name]
        try:
            recon = baseline_recon if name == 'baseline' else decode(blob)
            entry = dict(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                         payload_bytes=len(blob), vs_original=metrics(recon, x),
                         vs_baseline_reconstruction=metrics(recon, baseline_recon))
            report['results'][name] = entry
            print(f"{name}: median={entry['median_ms']:.1f}ms, p95={entry['p95_ms']:.1f}ms, "
                  f"bytes={len(blob)}, MSE={entry['vs_original']['mse']:.6f}, "
                  f"cos={entry['vs_original']['cosine']:.6f}", flush=True)
        except Exception as exc:
            errors[name] = f'복원 실패: {exc}'
    with (out / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'repeat', 'hm_ms', 'payload_bytes'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\n저장: {out.resolve()}/report.json')
    print('고정 입력 HM 비교입니다. 전체 지연시간 및 탐지 mAP 결과가 아닙니다.')


if __name__ == '__main__':
    main()
