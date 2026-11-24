"""보드 실행 전에 v2/v3 ONNX 출력과 원래 PyTorch 기준 특징을 비교."""
from pathlib import Path
import argparse
import gc
import json

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "eva01_board_k4_layout_v2")
    args = parser.parse_args()
    x = np.load(ROOT / "inputs/input_nchw.npy", allow_pickle=False)
    ref = np.load(ROOT / "inputs/reference_nchw.npy", allow_pickle=False)
    cases = [x, np.random.default_rng(20261124).normal(size=x.shape).astype(np.float32)]
    results = {}
    for name, root in (("v2", args.source), ("v3", ROOT)):
        print("ONNX CPU 비교:", name, flush=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(str(root / "models/eva01_k4_352.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        results[name] = [session.run(None, {"input": case})[0] for case in cases]
        del session
        gc.collect()
    comparisons = []
    for index, (a, b) in enumerate(zip(results["v2"], results["v3"])):
        if a.shape != (1, 1408, 22, 22) or b.shape != a.shape or not np.isfinite(b).all():
            raise RuntimeError("출력 형상/유한값 검사 실패")
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5)
        comparisons.append(dict(case="photo" if index == 0 else "seeded_random_normalized_buffer", max_abs=float(np.max(np.abs(b - a))), relative_l2=float(np.linalg.norm(b - a) / np.linalg.norm(a))))
    rel = float(np.linalg.norm(results["v3"][0] - ref) / np.linalg.norm(ref))
    if rel > 1e-4:
        raise RuntimeError("원래 PyTorch 기준 특징과 출력 불일치")
    report = dict(state="completed", v2_vs_v3=comparisons, original_pytorch_photo_relative_l2=rel,
                  scope="사진/고정 난수 입력의 ONNX CPU 수치 비교. RKNN 실행 배치/속도/탐지 정확도 검증 아님.")
    (ROOT / "runs/onnx_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
