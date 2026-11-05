# HW FCM 통합과 저장 입력 재생

## 범위

A는 보드 영상 압축(JPEG/HEVC) 후 서버 전체 YOLOX, B는 보드 dark3·FENet·특징 압축 후 서버 DRNet·suffix이다. HW 후보는 10비트 단색 HM을 8비트 NV12 HEVC로 변형한다. 기존 SW 경로는 유지한다. mAP 검증은 미완료이다.

## 외부 의존성

기존 ~/yolox_project 모델·FCM 번들, OpenCV/numpy/PyTorch, RKNN Lite2 2.3.2, GStreamer GI와 rockchipmpp, ffmpeg가 필요하다. HW는 GStreamer mpph265enc로 검증했다. FFmpeg hevc_rkmpp 파일 경로에서는 60장 중 59장만 복원되어 이번 통합에 사용하지 않았다(원인 미확정).

`jpeg_send.py`, `hevc_send.py`, `network_test/board_send.py`는 서버 담당 브랜치 커밋 `d5bd80bf843b0e0dbbb0debe5d31dda99eb44a6b`의 통신 코드 스냅샷이다. 설명의 서버 IP만 일반화했고 프로토콜은 유지했다. 서버 브랜치가 변경되면 호환성을 다시 확인한다.

배치 도구는 코드만 백업·복사하며 모델·번들·서버 어댑터·OS를 설치하지 않는다.

```bash
python3 tools/deploy_edge.py
python3 tools/deploy_edge.py --apply
```

## 보드 HW FCM 재생

서버 담당자가 [서버 어댑터](../integrations/server_hw/README.md)를 시작한 뒤 실행한다. SERVER_IP와 INPUT_SET은 실제 값으로 대체한다.

```bash
source ~/yolox_project/.venv/bin/activate
cd ~/yolox_project/darknet53
python -u replay_fcm_hw_server.py --inputs INPUT_SET --host SERVER_IP --port 9997 --warmup 5 --limit 3
```

3장 completed 확인 뒤 --limit 3을 빼면 전체 입력을 처리한다. PNG 읽기 제외, 전처리부터 결과 반환까지 포함한다. 같은 manifest 입력 순서·SHA256을 검증한다. 초기화와 워밍업은 제외한다. 실시간 센서 기준 E2E가 아니다.

## HW 전용 프로토콜

기존 길이 프레이밍: big-endian uint32 JSON 길이 + JSON + uint32 payload 길이 + payload.
JSON: mode=fcm_hw_v1, ratio=0.5, frame_id, qp=20.
Payload: FHW1 4바이트 + little-endian ddHHI 24바이트(std,mean,h,w,bitstream length) + Annex-B HEVC.
고정 h=320,w=640; 8bit NV12 all-I fixedQP20. std/mean은 FENet 출력에서 계산한다.

서버는 PyAV로 각 독립 프레임을 복원한다. Y 평면의 stride padding을 제거하고 직접 읽는다. RGB/gray 변환은 하지 않는다. Y*(1023/255) → 기존 dequantize_frame(n_bit=10) → 동일 DRNet → suffix. SW 10bit mono payload와 비호환이므로 별도 모드가 필수이다.

## 대역폭 추가 측정

기본 fq_codel 상태, 서버 경로 tailscale0를 먼저 확인한다. sudo 권한이 필요하다. 서버 TCP 업로드만 제한하며 인터페이스 기본 root qdisc는 일시 변경된다. SSH는 필터 대상이 아니다.

```bash
python -u benchmark_hw_fcm_bandwidth.py --inputs INPUT_SET --host SERVER_IP --rates 5 3 1 0.5
```

종료/오류/Ctrl+C 시 복구를 시도하며 restored=true를 확인한다. 강제 종료·전원 차단 시 복구가 보장되지 않는다. 수동 복구:

```bash
sudo tc qdisc replace dev tailscale0 root handle 7ffe: fq_codel limit 10240 flows 1024 quantum 1280 target 5ms interval 100ms memory_limit 33554432 ecn
sudo tc qdisc show dev tailscale0
```

TBF burst16KiB/latency5초를 고정한다. 작은 FCM payload는 burst로 초기 전송이 허용될 수 있다. 이 설정은 실험 모델이며 실제 일정 속도 링크·인터넷의 가용 대역폭을 보장하지 않는다. 비교 시 모든 설정을 공개한다.

## 실행 파일

- capture_replay_inputs.py: 카메라 입력 고정 PNG 세트 생성.
- live_dark3_fcm_server.py: 기존 SW FCM 카메라→서버.
- replay_fcm_server.py / replay_hevc_live_encode.py: 저장 입력 SW FCM / 영상 HEVC.
- run_replay_aba.py: JPEG–SW FCM–JPEG 반복.
- compare_bandwidth.py: JPEG/영상 HEVC/SW FCM 대역폭 비교.
- test_fcm_hw_candidate.py: 최초 단일 특징 HW 시험(비교 시간 범위 상이).
- compare_fcm_hw.py: 동일 NPU 특징으로 SW combined4 vs HW 압축·복원 비교.
- replay_fcm_hw_server.py: HW FCM 보드→서버.
- benchmark_hw_fcm_bandwidth.py: HW FCM만 대역폭 추가 측정.

HW FCM은 저장 입력 60장에서 물리 보드 검증됐다. 실시간 카메라 HW FCM 운영·장시간 안정성·전력·mAP는 아직 완료되지 않았다.
