"""v2의 8개 Transpose/Gather를 작은 조각으로 나누는 동일 계산 후보.

학습 가중치, 입력, 블록, attention 범위는 바꾸지 않는다.
보드 RKNN 변환·실행 성공이나 성능 향상은 이 스크립트가 보장하지 않는다.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import shutil
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper

EXPECTED_V2 = "1daacc9c2c7ced7cae301ca245454ce9ddf2504b1c59365c9cd434c306729fa1"
EXPECTED_V3 = "de386f6066fb41399e476b776e6577b1945ad1deb27949983dd55d45f3a0a9da"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "eva01_board_k4_layout_v2")
    args = parser.parse_args()
    source = args.source.resolve()
    if source == ROOT:
        raise RuntimeError("기존 v2와 출력 폴더가 같으면 안 됩니다.")
    for name in ("models", "inputs", "configs", "runs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    dest = ROOT / "models/eva01_k4_352.onnx"
    if dest.exists():
        raise RuntimeError("v3 ONNX가 이미 있습니다. 기존 결과를 덮어쓰지 않습니다.")
    path = source / "models/eva01_k4_352.onnx"
    contract = json.loads((source / "configs/contract.json").read_text())
    if digest(path) != EXPECTED_V2 or contract["onnx_sha256"] != EXPECTED_V2:
        raise RuntimeError("v2 ONNX 해시 불일치")
    started = time.perf_counter()
    model = onnx.load(str(path))
    nodes = list(model.graph.node)
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    replacements = {}
    removed = set()
    changes = []
    for block in range(4):
        positions, batches, keys = (256, 64, 16) if block < 3 else (484, 16, 22)
        suffixes = (4, 6) if block < 3 else (6, 8)
        for suffix in suffixes:
            name = f"/model/blocks.{block}/attn/Transpose_{suffix}"
            matched = [n for n in nodes if n.name == name]
            if len(matched) != 1:
                raise RuntimeError("대상 Transpose가 한 개가 아닙니다: " + name)
            transpose = matched[0]
            attrs = {a.name: helper.get_attribute_value(a) for a in transpose.attribute}
            uses = consumers.get(transpose.output[0], [])
            if transpose.op_type != "Transpose" or list(attrs.get("perm", [])) != [1, 0, 2]:
                raise RuntimeError("예상한 Transpose와 다릅니다: " + name)
            if len(uses) != 1 or uses[0].op_type != "Gather":
                raise RuntimeError("Transpose 다음 연산이 단일 Gather가 아닙니다: " + name)
            gather = uses[0]
            gather_attrs = {a.name: helper.get_attribute_value(a) for a in gather.attribute}
            if gather_attrs.get("axis", 0) != 2 or gather.input[0] != transpose.output[0]:
                raise RuntimeError("예상한 Gather 축과 다릅니다: " + name)
            # Gather와 Transpose는 서로 다른 축에 작용하므로 순서를 바꿀 수 있다.
            # query 축을 나누어 각 재배치의 query 수×key 수를 8192 이하로 만든다.
            chunk_size = min(32, 8192 // positions)
            replacement = []
            pieces = []
            for index, lo in enumerate(range(0, positions, chunk_size)):
                hi = min(lo + chunk_size, positions)
                prefix = name + f"_chunk_{index}"
                constant_names = []
                for label, value in (("starts", [lo]), ("ends", [hi]), ("axes", [0]), ("steps", [1])):
                    const_name = prefix + "_" + label
                    model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), const_name))
                    constant_names.append(const_name)
                sliced, gathered, transposed = [prefix + s for s in ("_slice", "_gather", "_transpose")]
                replacement.append(helper.make_node("Slice", [transpose.input[0], *constant_names], [sliced], name=prefix + "/Slice"))
                replacement.append(helper.make_node("Gather", [sliced, gather.input[1]], [gathered], axis=2, name=prefix + "/Gather"))
                replacement.append(helper.make_node("Transpose", [gathered], [transposed], perm=[1, 0, 2], name=prefix + "/Transpose"))
                pieces.append(transposed)
            replacement.append(helper.make_node("Concat", pieces, list(gather.output), axis=1, name=name + "_chunk_concat"))
            # 원래 Gather 위치에서 바꾸어 indices 생성 연산과의 의존 관계를 유지한다.
            replacements[gather.name] = replacement
            removed.add(transpose.name)
            changes.append(dict(block=block, original_transpose=name, input_shape=[positions, batches, keys],
                                query_chunk_size=chunk_size, chunks=len(pieces), max_chunk_query_key_product=chunk_size * positions))
    revised = []
    for node in nodes:
        if node.name in removed:
            continue
        revised.extend(replacements.get(node.name, [node]))
    del model.graph.node[:]
    model.graph.node.extend(revised)
    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    del model, nodes, revised, replacements, consumers
    gc.collect()
    # 출력 수치 비교는 validate_chunked_layout.py와 보드 실행에서 별도로 수행한다.
    updated = copy.deepcopy(contract)
    updated.update(onnx_sha256=digest(dest), parent_onnx_sha256=EXPECTED_V2,
                   implementation="layout_v3: eight relative-position Transpose/Gather pairs chunked along query axis; same weights/input/attention")
    if updated["onnx_sha256"] != EXPECTED_V3:
        raise RuntimeError("로컬에서 출력 비교를 통과한 v3 ONNX 해시와 다릅니다. 빌드를 중단합니다.")
    for name, key in (("input_nchw.npy", "input_sha256"), ("reference_nchw.npy", "reference_sha256")):
        src = source / "inputs" / name
        if digest(src) != contract[key]:
            raise RuntimeError("검증 입력 해시 불일치")
        shutil.copy2(src, ROOT / "inputs" / name)
    (ROOT / "configs/contract.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
    report = dict(state="completed", scope="ONNX 표준 연산의 동일 계산 분할. 보드 NPU 실행/성능/mAP 아님.",
                  parent_onnx_sha256=EXPECTED_V2, onnx_sha256=updated["onnx_sha256"],
                  onnx_bytes=dest.stat().st_size, prepare_seconds=time.perf_counter() - started, changes=changes)
    (ROOT / "runs/prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
