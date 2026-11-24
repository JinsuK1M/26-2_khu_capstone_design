"""이미 측정한 k4 보드 출력·CPU 기준·입력·계약을 서버 전달용으로 묶는다."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="검증한 runs/npu_... 폴더")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    report = json.loads((run / "report.json").read_text())
    contract_path = ROOT / "configs/contract.json"
    contract = json.loads(contract_path.read_text())
    if report.get("execution_completed") is not True or report.get("provisional_numeric_screen_pass") is not True:
        raise RuntimeError("보드 실행 및 특징 잠정 검사를 통과한 결과가 아닙니다.")
    for key in ("checkpoint_sha256", "onnx_sha256", "split_k"):
        if report[key] != contract[key]:
            raise RuntimeError("결과와 계약의 설정 불일치: " + key)
    feature = run / "npu_output_nchw.npy"
    y = np.load(feature, allow_pickle=False)
    if y.shape != (1, 1408, 22, 22) or y.dtype != np.float32 or not np.isfinite(y).all():
        raise RuntimeError("보드 특징의 형상/dtype/유한값 불일치")
    for name, key in (("input_nchw.npy", "input_sha256"), ("reference_nchw.npy", "reference_sha256")):
        if digest(ROOT / "inputs" / name) != contract[key]:
            raise RuntimeError("검증 입력/CPU 기준 해시 불일치: " + name)
    out = ROOT / "transfers" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    files = {
        "contract.json": contract_path,
        "input_nchw.npy": ROOT / "inputs/input_nchw.npy",
        "reference_nchw.npy": ROOT / "inputs/reference_nchw.npy",
        "npu_output_nchw.npy": feature,
    }
    # 전달용 report에서는 로컬 절대 경로를 제외한다. 원본 결과는 그대로 둔다.
    public_report = {k: v for k, v in report.items() if k not in ("image", "rknn_file", "error")}
    (out / "board_report.json").write_text(json.dumps(public_report, ensure_ascii=False, indent=2) + "\n")
    files["board_report.json"] = out / "board_report.json"
    manifest = dict(mode="uncompressed_k4_handoff", shape=list(y.shape), dtype=str(y.dtype), layout="NCHW",
                    raw_feature_bytes=y.nbytes, checkpoint_sha256=contract["checkpoint_sha256"],
                    resume="blocks[4], then remaining blocks/FPN/RPN/ROI; do not repeat patch/position embedding",
                    preprocessing=contract["preprocessing"],
                    scope="무압축 파일 전달. 서버 분할 실행/FCM/실시간 네트워크/mAP 성공을 의미하지 않음.",
                    files={name: dict(sha256=digest(path), file_bytes=path.stat().st_size) for name, path in files.items()})
    metadata = out / "manifest.json"
    metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive = out / "transfer_k4_v2.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        for name, path in files.items():
            packed.add(path, arcname=name)
        packed.add(metadata, arcname="manifest.json")
    print("서버 전달 파일:", archive)
    print("압축 파일 SHA256:", digest(archive))
    print("주의: tar.gz는 전달 파일 묶음이며 학습한 FCM 압축이 아닙니다.")


if __name__ == "__main__":
    main()
