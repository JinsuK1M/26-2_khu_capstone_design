"""학습 특징 수집용 공통 규격. 기존 v2 모델과 전처리를 그대로 사용한다."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "b6d833638fe9e99e54b1fa8dd5fa13cb611c8216ae50965c8e27b302e531ebb6"
ONNX_SHA = "1daacc9c2c7ced7cae301ca245454ce9ddf2504b1c59365c9cd434c306729fa1"
CHECKPOINT_SHA = "2cbbd46259851bad0052b418464cd866097e0e01bb60ae368b213ab040960650"
SHAPE = (1, 1408, 22, 22)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def preprocess(bgr):
    import cv2
    rgb = cv2.cvtColor(cv2.resize(bgr, (336, 336), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    pixels = rgb.astype(np.float32)
    pixels = (pixels - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
    canvas = np.zeros((1, 352, 352, 3), dtype=np.float32)
    canvas[0, :336, :336] = pixels
    return canvas


def output_nchw(outputs):
    if outputs is None or len(outputs) != 1:
        raise RuntimeError("모델 출력 개수가 다릅니다.")
    result = np.asarray(outputs[0])
    if result.shape == (1, 22, 22, 1408):
        result = result.transpose(0, 3, 1, 2)
    if result.shape != SHAPE or result.dtype != np.float32 or not np.isfinite(result).all():
        raise RuntimeError("특징 형상/dtype/유한값 검사 실패")
    return np.ascontiguousarray(result)


def to_fp16(feature):
    with np.errstate(over="ignore", invalid="ignore"):
        saved = feature.astype(np.float16)
    if not np.isfinite(saved).all():
        raise RuntimeError("FP16 저장 과정에서 overflow 또는 비유한값 발생")
    delta = saved.astype(np.float32) - feature
    numerator = np.sum(delta.astype(np.float64) ** 2)
    denominator = np.sum(feature.astype(np.float64) ** 2)
    error = float(np.sqrt(numerator / max(float(denominator), 1e-30)))
    return saved, error
