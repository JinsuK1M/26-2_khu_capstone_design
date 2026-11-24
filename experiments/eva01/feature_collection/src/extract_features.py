"""실제 보드 v2 출력의 FCM 학습용 FP16 NPY 수집. 학습·mAP 실행 아님."""
from pathlib import Path
from datetime import datetime
import argparse
import gc
import importlib.metadata
import json
import platform
import shutil
import sys
import tarfile
from time import perf_counter

import numpy as np

from common import ROOT, SHAPE, MODEL_SHA, ONNX_SHA, CHECKPOINT_SHA, sha, write_json, preprocess, output_nchw, to_fp16


def read_index(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r["image_id"] for r in rows}) != len(rows):
        raise RuntimeError("특징 목록에 중복 ID가 있습니다.")
    return rows


def validate_dataset(manifest):
    if manifest.get("split") != "train2017" or manifest.get("state") != "completed":
        raise RuntimeError("완료된 COCO train2017 목록만 사용합니다. val 평가 입력은 사용하지 마세요.")
    frames = manifest["frames"]
    if manifest["count"] != len(frames) or len({f["image_id"] for f in frames}) != len(frames):
        raise RuntimeError("학습 이미지 수/ID 불일치")
    return frames


def pack_run(out, rows):
    target = ROOT / "transfers" / (out.name + f"_{len(rows)}frames.tar.gz")
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        print("기존 전달 묶음:", target, flush=True)
        return
    if shutil.disk_usage(target.parent).free < sum(r["file_bytes"] for r in rows) + 64 * 1024**2:
        raise RuntimeError("전달 묶음 저장 공간이 부족합니다. 원본 특징은 보존되어 있습니다.")
    temp = target.with_name(target.name + ".part")
    print("FP16 특징 전달 묶음 생성 중. tar.gz는 FCM 압축이 아닙니다.", flush=True)
    with tarfile.open(temp, "w:gz", compresslevel=1) as archive:
        for name in ("metadata.json", "contract.json", "source_manifest.json", "report.json", "index.jsonl"):
            archive.add(out / name, arcname=name)
        for row in rows:
            path = out / row["feature_file"]
            if sha(path) != row["feature_sha256"]:
                raise RuntimeError("전달 파일 해시 불일치: " + str(path))
            archive.add(path, arcname=row["feature_file"])
    temp.replace(target)
    print("전달 파일:", target, flush=True)
    print("전달 파일 SHA256:", sha(target), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path.home() / "yolox_project/experiments/eva01_board_k4_layout_v2")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/coco_train_1000_seed20261124")
    parser.add_argument("--limit", type=int, default=20, help="첫20장 점검 후1000으로 늘릴 수 있음")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path)
    resume_group.add_argument("--resume-last", action="store_true")
    parser.add_argument("--pack", action="store_true", help="완료한 특징을 서버 전달용으로 묶기")
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames = validate_dataset(manifest)
    if not 1 <= args.limit <= len(frames):
        parser.error("--limit는 1..준비한 이미지 수 범위여야 합니다.")
    contract_path = model_root / "configs/contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["checkpoint_sha256"] != CHECKPOINT_SHA or contract["onnx_sha256"] != ONNX_SHA or contract["split_k"] != 4:
        raise RuntimeError("검증한 v2 체크포인트/ONNX/분할점과 다릅니다.")
    model_path = model_root / "models/eva01_k4_352_fp.rknn"
    print("기존 v2 모델 해시 검사 중. 재변환은 하지 않습니다.", flush=True)
    if sha(model_path) != MODEL_SHA:
        raise RuntimeError("검증한 v2 RKNN 해시와 다릅니다.")
    expected = dict(source_manifest_sha256=sha(manifest_path), contract_sha256=sha(contract_path), rknn_sha256=MODEL_SHA)
    if args.resume_last:
        out = Path((ROOT / ".last_features_run").read_text().strip()).resolve()
    elif args.resume:
        out = args.resume.expanduser().resolve()
    else:
        out = ROOT / "runs" / ("features_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        (out / "features").mkdir(parents=True, exist_ok=False)
        shutil.copy2(contract_path, out / "contract.json")
        shutil.copy2(manifest_path, out / "source_manifest.json")
        versions = {}
        for name in ("numpy", "opencv-python", "rknn-toolkit-lite2"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "배포 패키지 메타데이터 없음"
        write_json(out / "metadata.json", dict(**expected, checkpoint_sha256=CHECKPOINT_SHA,
                   split_k=4, producer="actual_board_rknn_layout_v2", saved_shape=list(SHAPE), saved_dtype="float16", layout="NCHW",
                   normalization=contract["preprocessing"], python=sys.version.split()[0], platform=platform.platform(), packages=versions,
                   note="실제 보드 출력 후 FP16 저장. 336/352는 현재 검증 경로와 동일. FCM 담당자와 이 조건을 공유한 뒤 학습. 학습·복원·정확도 검증을 실행한 결과가 아님."))
    metadata = json.loads((out / "metadata.json").read_text())
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("이어 추출할 데이터/모델/계약이 기존 수집과 다릅니다.")
    index_path = out / "index.jsonl"
    rows = read_index(index_path)
    frame_by_id = {f["image_id"]: f for f in frames}
    for row in rows:
        frame = frame_by_id.get(row["image_id"])
        path = out / row["feature_file"]
        if frame is None or frame["sha256"] != row["image_sha256"] or sha(path) != row["feature_sha256"]:
            raise RuntimeError("기존 수집 특징/원본의 해시 검사 실패")
        saved = np.load(path, allow_pickle=False)
        if saved.shape != SHAPE or saved.dtype != np.float16 or not np.isfinite(saved).all():
            raise RuntimeError("기존 수집 특징의 shape/dtype/유한값 불일치")
        del saved
    if len(rows) > args.limit or [r["image_id"] for r in rows] != [f["image_id"] for f in frames[:len(rows)]]:
        raise RuntimeError("수집 순서/개수가 요청한 첫 이미지 목록과 다릅니다.")
    remaining = args.limit - len(rows)
    if shutil.disk_usage(out).free < remaining * (int(np.prod(SHAPE)) * 2 + 256) + 128 * 1024**2:
        raise RuntimeError("남은 FP16 특징을 저장할 공간이 부족합니다.")
    (ROOT / ".last_features_run").write_text(str(out) + "\n")
    engine = None
    report = dict(state="running", producer="actual_board_rknn_layout_v2", dataset="COCO train2017", requested_images=args.limit,
                  source_images=len(frames), completed_images=len(rows), saved_shape=list(SHAPE), saved_dtype="float16", npu_core="NPU_CORE_0",
                  checkpoint_sha256=CHECKPOINT_SHA, rknn_sha256=MODEL_SHA, **{k:v for k,v in expected.items() if k != "rknn_sha256"},
                  note="특징 수집 작업. FCM 학습·복원·최종 검출·mAP·정식 지연 벤치마크 아님. 재개하면 여러 세션의 결과가 합쳐짐.")
    print("결과 폴더:", out, flush=True)
    print(f"목표 {args.limit}장 / 기존 완료 {len(rows)}장 / 새 추출 {remaining}장", flush=True)
    try:
        if remaining:
            import cv2
            from rknnlite.api import RKNNLite
            engine = RKNNLite()
            if engine.load_rknn(str(model_path)) != 0 or engine.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise RuntimeError("RKNN 로딩/초기화 실패")
            for name, key in (("input_nchw.npy", "input_sha256"), ("reference_nchw.npy", "reference_sha256")):
                if sha(model_root / "inputs" / name) != contract[key]:
                    raise RuntimeError("기존 v2 검사 입력 해시 불일치")
            x = np.load(model_root / "inputs/input_nchw.npy", allow_pickle=False)
            ref = np.load(model_root / "inputs/reference_nchw.npy", allow_pickle=False)
            for _ in range(3):
                warm = output_nchw(engine.inference(inputs=[np.ascontiguousarray(x.transpose(0,2,3,1))], data_format=["nhwc"]))
            a, b = warm.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
            rel = float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30))
            cos = float(np.dot(a,b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))
            if rel > .02 or cos < .999:
                raise RuntimeError("기존 검사 사진의 CPU 기준 특징과 차이가 큽니다.")
            report["preflight_feature_comparison"] = dict(relative_l2=rel, cosine=cos, scope="기존 검사 사진1장")
            del x, ref, warm, a, b
            gc.collect()
            write_json(out / "report.json", report)
            with index_path.open("a") as stream:
                for frame in frames[len(rows):args.limit]:
                    image_path = dataset / frame["file"]
                    if sha(image_path) != frame["sha256"]:
                        raise RuntimeError("학습 이미지 해시 불일치: " + frame["file"])
                    bgr = cv2.imread(str(image_path))
                    if bgr is None or bgr.shape[:2] != (frame["height"], frame["width"]):
                        raise RuntimeError("학습 이미지 디코딩/크기 오류")
                    inputs = preprocess(bgr)
                    tick = perf_counter()
                    outputs = engine.inference(inputs=[inputs], data_format=["nhwc"])
                    api_ms = (perf_counter() - tick) * 1000
                    feature = output_nchw(outputs)
                    saved, rounding = to_fp16(feature)
                    name = f'features/{frame["image_id"]:012d}.npy'
                    target = out / name
                    temp = target.with_name(target.name + ".part")
                    with temp.open("wb") as handle:
                        np.save(handle, saved, allow_pickle=False)
                    temp.replace(target)
                    row = dict(image_id=frame["image_id"], image_sha256=frame["sha256"], feature_file=name,
                               feature_sha256=sha(target), file_bytes=target.stat().st_size, raw_bytes=saved.nbytes,
                               shape=list(saved.shape), dtype=str(saved.dtype), layout="NCHW", npu_api_ms=api_ms,
                               fp16_rounding_relative_l2=rounding)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    report["completed_images"] = len(rows)
                    write_json(out / "report.json", report)
                    if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == args.limit:
                        print(f'{len(rows)}/{args.limit}: image_id={frame["image_id"]}, API={api_ms:.1f}ms, FP16 저장={saved.nbytes}B', flush=True)
                    del bgr, inputs, outputs, feature, saved
        report["state"] = "completed"
    except KeyboardInterrupt:
        report.update(state="interrupted", error="사용자가 중단. --resume-last로 이어서 추출 가능")
        raise
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        if engine is not None:
            engine.release()
        report.update(completed_images=len(rows), feature_file_bytes=sum(r["file_bytes"] for r in rows),
                      fp16_rounding_relative_l2_max=max((r["fp16_rounding_relative_l2"] for r in rows), default=None))
        write_json(out / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.pack:
        pack_run(out, rows)


if __name__ == "__main__":
    main()
