#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
PYTHON="$HOME/yolox_project/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then echo "보드 가상환경 Python 경로 확인 필요: $PYTHON"; exit 1; fi
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/board_check_${STAMP}.log"
printf '진행 로그: %s\n' "$LOG"
printf '1/2: 실제 k4 ONNX → RKNN 빌드\n'
if /usr/bin/time -v timeout --signal=TERM --kill-after=30s 1200s "$PYTHON" -u src/build_rknn.py 2>&1 | tee "$LOG" | sed -u '/^D /d'; then
  printf '\n2/2: 실제 NPU 실행·CPU 기준 특징 비교\n' | tee -a "$LOG"
else
  STATUS=$?
  printf '\n빌드 종료 코드=%s. NPU 실행은 진행하지 않습니다.\n' "$STATUS" | tee -a "$LOG"
  exit "$STATUS"
fi
if /usr/bin/time -v timeout --signal=TERM --kill-after=30s 300s "$PYTHON" -u src/test_npu.py 2>&1 | tee -a "$LOG"; then
  printf '\n보드 검사 명령 완료. report의 execution_completed와 provisional_numeric_screen_pass를 확인하세요.\n' | tee -a "$LOG"
else
  STATUS=$?
  printf '\nNPU 검사 종료 코드=%s\n' "$STATUS" | tee -a "$LOG"
  exit "$STATUS"
fi
