# 서버 수신 규격

`server/yolox_server.py`가 현재 구현한 형식이다. [Edge와 Server 연결 규격](edge_server_contract.md)의 "전송 계층에서 추가할 사항"에 대한 서버 쪽 제안이며, Edge와 합의 후 확정한다.

## 연결

- TCP, 기본 포트 `9997`, 연결 하나에서 프레임을 계속 주고받는다.
- 한 프레임 = 요청 메시지 2개(헤더, payload) + 응답 메시지 1개.
- 모든 메시지는 `[4바이트 길이, big-endian uint32][내용]`이다. 서버는 길이만큼 다 받을 때까지 읽는다(`recv` 한 번에 다 온다고 가정하지 않음).
- 메시지 최대 길이 64MB. 넘으면 연결을 끊는다.

## 요청

```text
[len][JSON 헤더 UTF-8]
[len][FCM payload]
```

JSON 헤더:

| 키 | 예 | 설명 |
| --- | --- | --- |
| `mode` | `"fcm"` / `"jpeg"` / `"hevc"` | payload 종류. 없으면 `fcm: true` 여부로 판단 (예전 호환) |
| `reset` | `true` | (hevc) 이 요청부터 HEVC 디코더를 새로 만든다 |
| `fcm` | `true` | (예전 방식) FCM payload 모드 |
| `frame_id` | `70` | 결과에 그대로 돌려줌. Edge의 processed 또는 capture 번호 사용 |
| `qp` | `20` | 기록용 (디코더는 QP와 무관) |
| `hm` | `"combined4"` | 기록용 (디코더는 HM 설정과 무관) |
| `ratio` | `0.5` | 원본 사진 → 640 축소 비율. 1280×960 카메라는 0.5. 박스를 원본 좌표로 돌려주기 위함. 없으면 1.0 (640 입력 좌표) |

`ratio`는 FCM 모드에서만 쓴다. JPEG 모드는 서버가 받은 이미지 크기로 비율을 직접 계산한다.

### mode = "fcm" (비교군 B)

FCM payload는 팀 번들 `compress()` 반환값 그대로이다: `<ddHHI` (std, mean, h, w, n) 24바이트 + HM 비트스트림 n바이트.

서버 검사: 길이 ≥ 24, `전체 길이 == 24 + n`, 타일 `320×640`, std·mean 유한값. 실패하면 오류 응답을 보내고 연결은 유지한다.

서버 처리: HM 디코딩 → DRNet → dark3 → dark4·dark5 → neck·head → 후처리

### mode = "jpeg" (비교군 A)

payload는 JPEG 파일 바이트 그대로이다 (`cv2.imencode(".jpg", BGR 이미지)` 결과).

서버 처리: `cv2.imdecode` (BGR) → 640×640 비율 유지 INTER_LINEAR 리사이즈 + 오른쪽·아래 114 패딩, BGR 0~255 → YOLOX-Darknet53 전체 → 후처리

### mode = "hevc" (비교군 A')

payload는 HEVC Annex-B **Access Unit 1개**(프레임 1장, 시작 코드 포함)이다. GStreamer라면 `h265parse config-interval=-1 ! video/x-h265,stream-format=byte-stream,alignment=au ! appsink`의 샘플 하나.

- 서버는 **연결 하나에 HEVC 디코더 하나**를 유지한다 (PyAV, 프레임 스레딩 끔 → AU 1개 입력 시 프레임 1개 바로 출력).
- 같은 연결로 AU를 순서대로 보낸다. 첫 AU는 VPS/SPS/PPS + I(IDR) 프레임이어야 한다.
- 연결이 끊기면 디코더도 버린다. 같은 연결에서 다시 시작하려면 헤더에 `"reset": true` (그 요청은 I 프레임이어야 함).
- AU 1개에서 프레임이 정확히 1개 나오지 않으면 오류 응답 (연결 유지).
- 서버 처리: AU → 디코딩(BGR) → 640 전처리 → YOLOX-Darknet53 전체 → 후처리.
- 응답 추가 키: `frame_type`(I/P), `decoded_index`(이 연결에서 디코딩한 순번, 0부터).

세 모드 모두 같은 체크포인트(`yolox_darknet.pth`), confidence 0.25, NMS 0.45, 박스는 원본 사진 좌표.

## 응답

```json
{
  "frame_id": 12,
  "mode": "jpeg",
  "payload_bytes": 198664,
  "dets": [{"box": [x1, y1, x2, y2], "conf": 0.93, "cls": "tv"}],
  "server_ms": 981.2,
  "t": {"decode_ms": 4.8, "preprocess_ms": 1.6, "infer_ms": 972.6, "post_ms": 2.2},
  "image_hw": [960, 1280],
  "ratio": 0.5
}
```

| `t` 키 | jpeg | fcm |
| --- | --- | --- |
| `decode_ms` | JPEG 디코딩 (hevc: AU 디코딩 + BGR 변환) | HM 디코딩 + DRNet (= `hm_decode_ms` + `drnet_ms` + 역정규화) |
| `hm_decode_ms` | – | HM 디코더 실행 (임시 파일·프로세스 포함) |
| `drnet_ms` | – | 역정규화 + DRNet |
| `preprocess_ms` | 640 리사이즈·패딩 | – (보드에서 수행) |
| `infer_ms` | YOLOX 전체 | dark4 ~ head |
| `post_ms` | 디코딩·NMS·좌표 변환 | 같음 |

- `server_ms`: payload를 다 받은 직후 ~ 결과 JSON을 만들기 직전. 받는 시간·보내는 시간은 포함하지 않는다.
- `box`: 원본 사진 좌표, `conf`: obj × class 점수.
- 예전 클라이언트 호환용으로 `unpack_ms`(decode + preprocess), `tail_ms`(infer)도 함께 보낸다.
- 오류: `{"frame_id": 70, "error": "길이 불일치: ...", "dets": [], "server_ms": 0.0}`

## 시간 측정

Edge와 서버의 시계는 맞추지 않는다. `board_send.py`가 보드 시계로 왕복 시간(`rtt_ms`)을 재고, 서버가 보낸 `server_ms`를 빼서 네트워크 시간(`net_ms`)을 구한다.

```text
total_ms (보드) = 보드 처리 시작 ~ 결과 수신 끝     (PNG 읽기 제외)
  A (jpeg): JPEG 인코딩 시작 ~ 결과 수신
  B (fcm) : 전처리(또는 dark3 NPU) 시작 ~ 결과 수신
rtt_ms   = 요청 전송 시작 ~ 결과 수신 끝 = 전송 + server_ms + 응답
net_ms   = rtt_ms − server_ms
```

보드 클라이언트: A'(hevc)는 `server/hevc_send.py` (파일 시험 + 실시간용 `HevcLink`, 표준 라이브러리만 사용). A는 `server/jpeg_send.py` (manifest 순서대로 PNG → JPEG, 순차 전송, 워밍업 분리, CSV/JSON 로그). B는 `server/board_send.py`의 `ServerLink`를 쓰면 같은 형식으로 보낼 수 있다.

## 아직 정하지 않은 것

- 재접속 정책 (현재: 서버는 끊기면 다음 연결을 기다림, 보드 쪽 재시도 없음)
- 모델 해시·버전 확인을 헤더에 넣을지
- 프레임이 밀릴 때 서버에서 버릴지 (현재: 순서대로 하나씩 처리)
