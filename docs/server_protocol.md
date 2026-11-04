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
| `fcm` | `true` | FCM payload 모드 |
| `frame_id` | `70` | 결과에 그대로 돌려줌. Edge의 processed 또는 capture 번호 사용 |
| `qp` | `20` | 기록용 (디코더는 QP와 무관) |
| `hm` | `"combined4"` | 기록용 (디코더는 HM 설정과 무관) |
| `ratio` | `0.5` | 원본 사진 → 640 축소 비율. 1280×960 카메라는 0.5. 박스를 원본 좌표로 돌려주기 위함. 없으면 1.0 (640 입력 좌표) |

FCM payload는 팀 번들 `compress()` 반환값 그대로이다: `<ddHHI` (std, mean, h, w, n) 24바이트 + HM 비트스트림 n바이트.

서버 검사: 길이 ≥ 24, `전체 길이 == 24 + n`, 타일 `320×640`, std·mean 유한값. 실패하면 오류 응답을 보내고 연결은 유지한다.

## 응답

```json
{
  "frame_id": 70,
  "dets": [{"box": [x1, y1, x2, y2], "conf": 0.94, "cls": "person"}],
  "unpack_ms": 173.0,
  "tail_ms": 204.0,
  "server_ms": 378.0
}
```

- `box`: 원본 사진 좌표(`ratio` 적용), `conf`: obj × class 점수, 기본 0.25 이상, NMS 0.45.
- `unpack_ms`: HM 디코딩 + DRNet, `tail_ms`: dark4 ~ 후처리, `server_ms`: 서버 전체 처리.
- 오류: `{"frame_id": 70, "error": "길이 불일치: ...", "dets": [], "server_ms": 0.0}`

## 시간 측정

Edge와 서버의 시계는 맞추지 않는다. `board_send.py`가 보드 시계로 왕복 시간(`rtt_ms`)을 재고, 서버가 보낸 `server_ms`를 빼서 네트워크 시간(`net_ms`)을 구한다.

```text
E2E (보드 기준) = 보드 처리(전처리 ~ payload 완성) + rtt_ms
rtt_ms        = 전송 + server_ms + 결과 수신
net_ms        = rtt_ms − server_ms
```

## 아직 정하지 않은 것

- 재접속 정책 (현재: 서버는 끊기면 다음 연결을 기다림, 보드 쪽 재시도 없음)
- 모델 해시·버전 확인을 헤더에 넣을지
- 프레임이 밀릴 때 서버에서 버릴지 (현재: 순서대로 하나씩 처리)
