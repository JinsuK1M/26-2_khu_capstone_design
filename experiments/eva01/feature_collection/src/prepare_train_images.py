"""COCO train2017의 고정 1000장을 별도 준비. 기존 val 평가 자료는 변경하지 않는다."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import random
import re
import shutil
import subprocess
import zipfile

from common import ROOT, sha, write_json

BASE = "https://s3.amazonaws.com/images.cocodataset.org"


def read_image_array(stream):
    """큰 annotations 배열은 읽지 않고 공식 JSON의 images 배열만 읽는다."""
    text = ""
    decoder = json.JSONDecoder()
    start = None
    for _ in range(128):
        part = stream.read(1024 * 1024)
        if not part:
            break
        text += part
        if start is None:
            match = re.search(r'"images"\s*:\s*\[', text)
            if match:
                start = match.end() - 1
        if start is not None:
            try:
                images, _ = decoder.raw_decode(text, start)
                if isinstance(images, list):
                    return images
            except json.JSONDecodeError:
                pass
    raise RuntimeError("공식 정답 JSON의 images 배열을 읽지 못했습니다.")


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    timeout = "1800" if path.suffix == ".zip" else "120"
    result = subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "20", "--max-time", timeout,
                             "-o", str(temp), url], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("다운로드 실패: " + url + "\n" + result.stderr.decode(errors="replace")[-600:])
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--annotations", type=Path, help="기존 instances_train2017.json 또는 annotations_trainval2017.zip")
    parser.add_argument("--coco-root", type=Path, default=Path.home() / "yolox_project/evaluation/coco2017")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/coco_train_1000_seed20261124")
    args = parser.parse_args()
    if not 1 <= args.count <= 118287:
        parser.error("--count는 1..118287 범위여야 합니다.")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("split") != "train2017" or previous.get("count") != args.count or previous.get("seed") != args.seed:
            raise RuntimeError("기존 데이터 목록의 split/count/seed가 다릅니다. 별도 --output을 사용하세요.")
        print("기존 train 목록·SHA256 확인 중", flush=True)
        for frame in previous["frames"]:
            if sha(out / frame["file"]) != frame["sha256"]:
                raise RuntimeError("기존 이미지가 변경됐습니다: " + frame["file"])
        print("준비 완료:", manifest_path, flush=True)
        return
    coco_root = args.coco_root.expanduser().resolve()
    candidates = [args.annotations.expanduser()] if args.annotations else [
        coco_root / "annotations/instances_train2017.json", coco_root / "annotations_trainval2017.zip"]
    annotation = next((p for p in candidates if p.is_file()), None)
    if annotation is None:
        if args.annotations:
            raise FileNotFoundError(str(args.annotations))
        annotation = ROOT / "datasets/annotations_trainval2017.zip"
        print("기존 train 정답 목록이 없어 공식 정답 ZIP을 다운로드합니다. 약250MB", flush=True)
        download(BASE + "/annotations/annotations_trainval2017.zip", annotation)
    print("train2017 이미지 목록만 읽는 중:", annotation, flush=True)
    if annotation.suffix == ".zip":
        with zipfile.ZipFile(annotation) as archive:
            with archive.open("annotations/instances_train2017.json") as raw:
                images = read_image_array(io.TextIOWrapper(raw, encoding="utf-8"))
    else:
        if annotation.name != "instances_train2017.json":
            raise RuntimeError("train2017 정답 JSON만 사용할 수 있습니다.")
        with annotation.open() as stream:
            images = read_image_array(stream)
    if len(images) != 118287 or len({im["id"] for im in images}) != 118287:
        raise RuntimeError("공식 train2017의 118287장 목록이 아닙니다.")
    if any("/train2017/" not in im.get("coco_url", "") for im in images):
        raise RuntimeError("train2017 외 이미지가 목록에 포함되어 있습니다.")
    images.sort(key=lambda im: im["id"])
    random.Random(args.seed).shuffle(images)
    selected = images[:args.count]
    del images
    gc.collect()
    if shutil.disk_usage(out).free < 3 * 1024**3:
        raise RuntimeError("이미지·특징·전달 묶음을 위해 여유 공간3GiB 이상을 확보해주세요.")

    def obtain(im):
        name = f'{int(im["id"]):012d}.jpg'
        if im["file_name"] != name:
            raise RuntimeError("이미지 이름과 ID 불일치")
        path = out / "images" / name
        cached = coco_root / "train2017" / name
        if not path.exists():
            if cached.is_file():
                path.parent.mkdir(exist_ok=True)
                shutil.copy2(cached, path)
            else:
                download(BASE + "/train2017/" + name, path)
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None or bgr.shape[:2] != (im["height"], im["width"]):
            raise RuntimeError("이미지 디코딩/크기 검사 실패: " + name)
        return dict(image_id=int(im["id"]), file="images/" + name, width=im["width"], height=im["height"], sha256=sha(path))

    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, frame in enumerate(pool.map(obtain, selected), 1):
            frames.append(frame)
            if index == 1 or index % 25 == 0:
                print(f"학습 이미지 준비 {index}/{args.count}", flush=True)
    record = dict(state="completed", dataset="COCO train2017", split="train2017", purpose="FCM training feature extraction",
                  count=args.count, seed=args.seed, frames=frames,
                  source_annotations_sha256=sha(annotation), source_images_base=BASE + "/train2017/",
                  note="val2017 평가500장을 재사용하지 않음. 원본 JPEG와 평가 폴더는 변경하지 않음. 아직 특징 추출/FCM 학습/mAP를 실행한 것이 아님.")
    write_json(manifest_path, record)
    print("준비 완료:", manifest_path, flush=True)
    print("manifest SHA256:", sha(manifest_path), flush=True)


if __name__ == "__main__":
    main()
