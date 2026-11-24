#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p reports models
SIZE=${1:-640}
case "$SIZE" in 640|1280) ;; *) echo "입력 크기는 640 또는 1280이어야 합니다."; exit 2;; esac
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="reports/board_backbone_${SIZE}_${STAMP}.log"
printf "진행 로그: %s\n" "$LOG"
printf "실행 중 출력은 이 파일에 저장됩니다. 종료 후 마지막 결과를 표시합니다.\n"
PYTHON="$HOME/yolox_project/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then echo "기존 보드 Python 경로 확인 필요: $PYTHON"; exit 1; fi
{ date -Is; uname -a; free -h; cat /proc/swaps; } > "$LOG"
/usr/bin/time -v timeout --signal=TERM --kill-after=30s 1800s \
 "$PYTHON" -u src/probe_full_backbone_cpu.py --checkpoint models/eva_coco_det.pth --size "$SIZE" >> "$LOG" 2>&1
STATUS=$?
printf '\nexit_code=%s\n' "$STATUS" >> "$LOG"
free -h >> "$LOG"
tail -n 70 "$LOG"
printf '\n종료 코드: %s\n로그: %s\n' "$STATUS" "$LOG"
exit "$STATUS"
