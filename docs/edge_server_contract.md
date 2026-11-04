# Edge와 Server 연결 규격

## 분할 지점

| 구간 | 역할 |
| --- | --- |
| Edge | 카메라 전처리 → stem → dark2 → dark3 → FENet → 10비트 정수화·타일링 → HM |
| Server | HM → 역타일링·역정규화 → DRNet → dark3 → dark4·dark5 → neck·head·후처리 |

FCM은 dark3(l37) 하나를 입력으로 학습했다. dark4·dark5용 FCM을 추가로 요구하지 않는다. 서버는 복원된 dark3를 보존하면서 후속 특징을 계산하고 neck으로 전달해야 한다. 이 저장소에는 서버 구현이 아직 없다.

## 이미지와 특징 형식

- 카메라: 1280×960 NV12, stride 1280, 1,843,200바이트.
- 전처리: NV12 → BGR, `INTER_LINEAR` 비율 유지 축소, 이미지 좌상단 배치, 오른쪽·아래쪽 상수 114 패딩.
- 현재 카메라: 640×480으로 축소하고 아래 160행 패딩 → 640×640.
- BGR 값 범위: 0~255, `/255` 또는 mean/std 정규화 없음.
- dark3: float32 NCHW `(1,256,80,80)`, 유한값, 6,553,600바이트.
- FENet 출력: float32 인터페이스 `(128,40,40)`. NPU 내부 연산은 비INT8이며 FP32와 동일하다는 의미가 아니다.
- codec: 128개 채널을 320×640 단색 10비트 프레임으로 타일링하고 HM에 입력한다.

모델과 샘플의 식별값은 [모델 파일 문서](artifacts.md)를 따른다. 단순한 채널 수 일치만으로 다른 Backbone이나 전처리와 호환된다고 판단하지 않는다.

## FCM payload

팀 번들의 `compress(enc, dark3, qp=20)` 반환값은 다음과 같다.

```text
struct format: <ddHHI
little-endian

std       float64  8바이트
mean      float64  8바이트
height    uint16   2바이트   현재 320
width     uint16   2바이트   현재 640
n         uint32   4바이트   뒤따르는 HM 비트스트림 길이
bitstream bytes    n바이트

전체 길이 = 24 + n
```

현재 선택한 Edge HM 설정은 아래와 같다.

```text
QP=20
--RDOQ=0 --RDOQTS=0
--TransformSkip=0 --TransformSkipFast=0
--QuadtreeTUMaxDepthIntra=1 --SAO=0
```

DRNet은 기존 학습 모델을 사용한다. HM 옵션과 FENet 실행 정밀도가 변경됐으므로 압축·복원 성공만으로 최종 mAP 유지가 입증되지는 않는다.

## 전송 계층에서 추가할 사항

아래는 합의가 필요한 항목이며 현재 구현된 네트워크 프로토콜이 아니다.

- 세션·프레임 ID와 payload 대응
- 메시지 경계와 길이, 수신 허용 최대 크기
- QP·codec 프로필·모델 버전/해시
- 시간 측정 지점과 clock, 결과의 프레임 ID
- 연결 종료·재접속·오류 처리

TCP를 사용하더라도 한 번의 `recv`가 전체 payload를 반환한다고 가정하지 않는다. 헤더의 길이·타일 크기·유한한 통계값을 검증한 후 완전한 메시지를 복원한다.

## 정확도·지연시간 측정 시 구분

1. PyTorch dark3와 보드 dark3의 차이
2. CPU FENet과 NPU FENet의 출력 차이
3. 손실 압축 후 복원 특징의 차이
4. 탐지 head·후처리까지 연결한 mAP 차이

코사인 유사도나 상대 L2는 mAP가 아니다. 기존 보드 처리 지연은 프레임 선택 후 전처리 시작부터 payload 완성까지이다. 네트워크·서버 시간과 합칠 때 측정 경계 및 서로 다른 clock을 맞춰야 한다.
