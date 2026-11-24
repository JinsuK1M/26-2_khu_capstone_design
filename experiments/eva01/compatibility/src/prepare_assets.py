"""기존 EVA 검사 폴더의 검증된 외부 파일을 읽기 전용 용도로 연결한다.

가중치/소스를 다운로드하거나 기존 환경에 패키지를 설치하지 않는다.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5623d08bb08a8268c7bc18e1d9b8b4bd651fe4d9"
FILES = {
    "models/eva_coco_det.pth": "2cbbd46259851bad0052b418464cd866097e0e01bb60ae368b213ab040960650",
    "vendor/bus.jpg": "fb4914d123d97c440cd127ef0e98d4bdc68cd88e2683657528928b4a34014e16",
    "vendor/timm_0_6_11_mlp.py": "da6ed06863182ad4d40f41a3b914228edeb412bb8813213b5176664246823486",
    "vendor/EVA/EVA-01/det/detectron2/modeling/backbone/vit.py": "bd7fdb9c53794520c0aad675334c91ef83e665bbaef26d5c99376c1b63d82cf4",
    "vendor/EVA/EVA-01/det/detectron2/modeling/backbone/utils.py": "f73e6ea4ebad52634eda5535834539e379ddea2fa70a12f7b9ab8451d619054c",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", type=Path, required=True,
                        help="models/eva_coco_det.pth와 vendor/EVA 등을 준비한 기존 EVA 검사 폴더")
    args = parser.parse_args()
    existing = args.existing_root.expanduser().resolve()
    if existing == ROOT:
        raise RuntimeError("원본과 연결할 폴더가 같으면 안 됩니다.")
    revision = subprocess.check_output(["git", "-C", str(existing / "vendor/EVA"), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("공식 소스 revision 불일치: " + revision)
    for name, expected in FILES.items():
        if digest(existing / name) != expected:
            raise RuntimeError("외부 파일 해시 불일치: " + name)
    for name in ("models/eva_coco_det.pth", "vendor/bus.jpg", "vendor/timm_0_6_11_mlp.py", "vendor/EVA"):
        dest = ROOT / name
        src = existing / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if dest.resolve() != src.resolve():
                raise RuntimeError("다른 파일이 이미 있습니다. 덮어쓰지 않습니다: " + str(dest))
        else:
            dest.symlink_to(src, target_is_directory=src.is_dir())
    print(json.dumps(dict(state="completed", source_revision=revision, hashes_verified=list(FILES),
                          note="기존 파일에 심볼릭 링크만 생성. 기존 파일/환경 수정 및 다운로드 없음."), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
