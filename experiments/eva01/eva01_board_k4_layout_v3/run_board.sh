#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
PYTHON="$HOME/yolox_project/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "보드 가상환경 Python 경로 확인 필요: $PYTHON"
  exit 1
fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/board_check_${STAMP}.log"
printf '진행 로그: %s\n' "$LOG"
printf '1/3: 기존 v2 ONNX에서 v3 생성. 가중치 재다운로드 없음.\n'
if [ ! -f models/eva01_k4_352.onnx ]; then
  "$PYTHON" -u src/prepare_chunked_layout.py 2>&1 | tee "$LOG"
else
  printf '기존 v3 파일 사용. 빌드 단계에서 계약 파일의 해시를 검사합니다.\n' | tee "$LOG"
fi
printf '\n2/3: RKNN 변환\n' | tee -a "$LOG"
/usr/bin/time -v timeout --signal=TERM --kill-after=30s 1200s "$PYTHON" -u src/build_rknn.py 2>&1 | tee -a "$LOG" | sed -u '/^D /d'
printf '\n3/3: 실제 보드 실행 및 원래 CPU 특징 비교\n' | tee -a "$LOG"
/usr/bin/time -v timeout --signal=TERM --kill-after=30s 300s "$PYTHON" -u src/test_npu.py 2>&1 | tee -a "$LOG"
printf '\nCPU 실행 배치 및 크기 제한 경고:\n' | tee -a "$LOG"
grep -E 'CPU|fallback|exceeded|limitations' "$LOG" | tail -n 70 || true
printf '\n완료. 실행 성공과 특징 검사 결과를 report에서 확인하세요.\n'
