# 서버 실험 기록 (2026-11-05)

## 1. 보드–서버 네트워크 연결

- Mac 서버를 Tailscale(100.90.83.29:9997)에 연결하고 Edge 계정에 공유.
- Mac 자체 Tailscale 주소로 `board_send.py` 연결 확인: 서버 처리 약 270ms (HM 디코딩 + DRNet 약 100ms, 탐지 약 170ms).
- 보드 → 서버 → 결과 반환까지 연결 동작은 Edge 쪽에서 확인.

## 2. 카메라 회전 보정 후 보드 payload 3프레임 비교

NPU FENet + HM combined4, QP20. 사진·payload·JSON이 같은 프레임. 신뢰도 0.25 이상, 같은 클래스·IoU 0.5 이상을 같은 물체로 판단.

| 프레임 | payload | 보드 처리 | 주요 물체 (원본 → payload 복원 후) | 같은 물체 |
| --- | --- | --- | --- | --- |
| 1 | 21,698B | 876ms | teddy bear 0.934→0.929, tv 0.931→0.930, cup 0.908→0.879, laptop 0.832→0.830 | 8/10 |
| 2 | 22,682B | 881ms | tv 0.923→0.919, laptop 0.755→0.793, bottle 0.437→0.463, mouse 0.293→0.268 | 10/10 |
| 3 | 21,616B | 924ms | tv 0.924→0.918, teddy bear 0.919→0.918, laptop 0.802→0.809, cup 0.568→0.472 | 11/11 |

- 원본이 찾은 31개 중 29개 탐지. 놓친 2개는 원본 0.278·0.268의 작은 책.
- 회전 보정 전(누운 사진)에는 원본 모델부터 오탐(물컵→tv, 선반→sink)이 있었고, 보정 후 사라짐.
- 3프레임 비교이며 mAP가 아니다.

## 3. 비교군 A (JPEG) 경로 추가

- 서버: 요청 헤더 `mode`로 `fcm` / `jpeg` 구분. JPEG는 `cv2.imdecode`(BGR) → 640 비율 유지·오른쪽/아래 114 패딩 → YOLOX-Darknet53 전체 → 원본 좌표 반환. FCM과 같은 체크포인트, conf 0.25, NMS 0.45.
- 서버 시간을 단계별로 반환: JPEG `decode/preprocess/infer/post`, FCM `hm_decode/drnet/infer/post`.
- 보드 클라이언트 `server/jpeg_send.py`: manifest 순서대로 PNG → JPEG(기본 q90) → 순차 전송, 워밍업 분리, SHA256 확인, 프레임별 CSV/JSON + 요약.
- 측정 경계: PNG 읽기 제외, `total_ms` = JPEG 인코딩 시작 ~ 결과 수신 (보드 시계), `net_ms` = `rtt_ms` − `server_ms`.
- 서버 단독 시험(6장, 같은 컴퓨터): JPEG q90 약 188KB, 보드 인코딩 약 4ms, 서버 디코딩 + 전처리 약 6ms. 기존 FCM 요청도 그대로 동작.
- 규격: [서버 수신 규격](server_protocol.md)

## 4. HEVC 파일 디코딩 확인 (비교군 확장 준비)

Edge가 GStreamer `mpph265enc`(1280×960, 1fps, 목표 1Mbps, GOP 30)로 만든 `test_gstreamer.hevc`를 서버 환경에서 디코딩. 도구: `server/hevc_check.py`.

| 항목 | 결과 |
| --- | --- |
| 파일 크기 | 6,692,961 bytes |
| 프레임 수 | **60** (모두 1280×960×3) |
| 코덱 | HEVC Main, yuv420p |
| 프레임 종류 | I 2장 (0, 30번), P 58장, B 없음 → 디코딩 순서 = 표시 순서 |
| 평균 크기 | 약 109KB/프레임 (I 약 130KB, P 약 100~140KB, 0번 I만 31KB) |
| 방향 | 정상 (똑바로) |
| 색상 | 정상 (빨간 인형은 빨강, 노란 포스터는 노랑 → RGB/BGR 뒤바뀜 없음) |
| 연속성 | 앞뒤 프레임 평균 차이 중앙값 2.9 (0~255), 장면 순서 자연스러움 |
| 탐지 (YOLOX 전체) | 0번 mouse 0.91·tv 0.85, 20번 person 0.92, 28번 teddy bear 0.92 |

- Mac에서도 OpenCV로 60장 1280×960 디코딩 확인.
- 결과: `results/2026-11-05/hevc_decode_check.json`, 60장 모음 `results/2026-11-05/hevc_decode_frames.jpg`.

### 서버 HEVC 경로 설계 메모 (구현 전)

- 연결 하나에 디코더 하나를 유지 (P 프레임이 앞 프레임을 참조하므로 상태 유지 필요).
- B 프레임이 없어 요청 하나(Access Unit 하나) → 디코딩 프레임 하나 → 결과 하나로 frame_id를 짝지을 수 있다.
- 첫 요청은 VPS·SPS·PPS + I 프레임을 포함해야 한다.
- 디코더: PyAV (`pip install av`). Edge의 프레임 단위 전송 형식 확정 후 구현.

## 다음 작업

- [ ] Edge에서 JPEG 60장 실행 → `summary.json`으로 A 결과 정리
- [ ] 같은 60장으로 B(FCM) 같은 측정 경계 실행 → A와 비교
- [ ] HEVC 전송 단위 확정 후 서버 HEVC 모드 구현
