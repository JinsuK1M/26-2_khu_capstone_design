# Server: FCM 복원 → YOLOX-Darknet53 탐지

Edge(Orange Pi)가 보낸 dark3 FCM payload를 받아 복원하고, 나머지 YOLOX-Darknet53을 실행해 탐지 결과를 돌려주는 서버 코드와 정확도 검증 도구이다. 실행 환경은 Mac(Apple Silicon)에서 확인했다.

## 구성

```text
Edge payload (FCM bytes, <ddHHI 24B 헤더 + HM 비트스트림)
 → [TCP 9997] 길이 헤더 + JSON 헤더(frame_id, qp, ratio) + payload
 → HM 디코딩 → 역타일링·역정규화 → DRNet → dark3 (1,256,80,80)
 → dark4 → dark5 → YOLOFPN neck → YOLOX head → NMS
 → [TCP] JSON 결과 {frame_id, dets, server_ms}
```

서버는 dark3 하나만 받고 dark4·dark5를 직접 계산한다. Edge 쪽 규격은 [Edge와 Server 연결 규격](../docs/edge_server_contract.md)을 따르고, 서버 수신 형식은 [서버 수신 규격](../docs/server_protocol.md)에 정리했다.

## 파일

| 파일 | 실행 위치 | 역할 |
| --- | --- | --- |
| `yolox_split.py` | 서버 | YOLOX 모델 로드, 전처리, Edge/Server 분할 함수 (다른 파일이 import) |
| `fcm_plugin_yolox.py` | 서버·평가 | FCM 팀 번들 연결, HM 설정 `default` / `combined4` 선택 |
| `yolox_server.py` | 서버 | TCP 서버. FCM·JPEG·HEVC 모드 (HEVC는 연결별 디코더 유지), 단계별 서버 시간, frame_id 반환 |
| `hevc_send.py` | **보드** | 비교군 A': HEVC AU 단위 전송 (파일 시험 + 실시간용 `HevcLink`), AU 분리 함수, CSV/JSON 로그 |
| `jpeg_send.py` | **보드** | 비교군 A: manifest PNG → JPEG 전송, 순차·워밍업 분리, 프레임별 CSV/JSON 로그 |
| `board_send.py` | **보드** | 서버로 payload 전송 (표준 라이브러리만 사용). 파일 시험 + 실시간 코드용 `ServerLink` |
| `yolox_client.py` | Mac | 가짜 Edge. 사진 → dark3 → FCM 압축 → 서버 전송 |
| `yolox_eval.py` | Mac | 원본 / 분할 / uint8 / FCM 경로의 mAP 비교 (yolov5 `val.py` 사용) |
| `yolox_check.py` | Mac | 분할 전후 출력 일치 확인 |
| `yolox_npu_check.py` | Mac | 보드 NPU dark3 → 서버 탐지 비교 |
| `hevc_check.py` | Mac | 보드 HEVC 파일 디코딩 확인 (프레임 수·종류·크기, 프레임 저장, 선택 탐지) |
| `yolox_payload_check.py` | Mac | 보드 payload(.bin) → 서버 복원·탐지, 원본 모델과 비교 그림 저장 |

## 빠른 실행

환경 준비는 [Mac 서버 환경](../docs/server_setup_mac.md)을 따른다.

```bash
# 서버
python yolox_server.py

# 보드에서 저장 payload 로 연결 시험
python board_send.py --host <서버 주소> last_payload.bin --repeat 20

# 보드 payload 를 파일로 직접 확인
python yolox_payload_check.py --payload last_payload.bin --image last_frame.png --meta last_frame.json --conf 0.05

# 정확도 (yolov5 폴더에서)
python yolox_eval.py --data data/coco.yaml --limit 500 --only ACD --qp 20,25 --hm combined4
```

## 검증 상태 (2026-11-04)

| 항목 | 결과 |
| --- | --- |
| 분할 전후 출력 | 차이 0 |
| FCM 경로 mAP50-95 (val2017 앞 500장, CPU FENet + combined4) | 원본 0.532 → QP20 0.526 (19.9KB), QP25 0.521 (11.7KB) |
| combined4 vs 학습 HM 설정 (COCO128) | mAP 차이 0.003 이내 |
| 보드 NPU dark3 → 서버 (zidane) | 원본과 같은 3개 탐지 |
| 보드 최종 payload (NPU FENet + combined4) → 서버 | 복원 성공, 원본 6개 중 5개 같은 위치 (신뢰도 0.05 이상) |
| 서버 처리 (Mac) | 복원 약 170ms + 탐지 약 200ms |

자세한 수치와 측정 범위는 [서버 실험 기록](../docs/experiments_2026-11-04.md)과 `results/2026-11-04/`에 있다.

**아직 하지 않은 것:** 보드–서버 실제 네트워크 연결과 E2E 지연, 보드 NPU FENet 경로의 mAP, 물체가 분명한 카메라 프레임에서의 비교.
