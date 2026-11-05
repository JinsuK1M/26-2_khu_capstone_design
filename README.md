# 26-2_khu_capstone_design

FCM(Feature Coding for Machines)을 이용한 엣지–서버 오프로딩 지연시간 최적화 연구.

Orange Pi 5 Pro에서 카메라 입력, YOLOX-Darknet53 dark3 특징 추출, FCM 압축을 연결한 Edge 코드와 보드 실험 기록이다. SW FCM 서버 연결과 저장 입력 HW FCM 통합까지 검증했다.

## 최신 진행 (2026-11-06)

HW 특징 압축 후보와 서버 복원·탐지 연결을 동일 저장 입력 60장에서 검증했다. 서버 방향 업로드 상한 5/3/1/0.5Mbps에서 전체 지연 중앙값은 566/568/584/601ms였다. mAP와 실시간 카메라 HW 경로는 미검증이며 기존 SW 기본값은 보존한다.

- [HW 통합·실행 방법](docs/hw_fcm_integration.md)
- [지연·전송량 및 한계](docs/experiments_2026-11-06.md)
- [서버 담당자용 별도 어댑터](integrations/server_hw/README.md)

아래 구성과 11월3일 값은 기존 SW 경로 기록이다.

## 현재 구성

```text
OV13855 카메라 1280×960 NV12
 → 640×480 BGR 축소 + 아래쪽 114 패딩 → 640×640
 → YOLOX-Darknet53 stem / dark2 / dark3 (NPU)
 → FENet (NPU)
 → 10비트 정수화·타일링
 → HM combined4, QP20
 → payload

SW 연결 완료: 소켓 전송 → 서버 HM·DRNet 복원 → dark4·dark5 → neck·head → 탐지
```

### 선택한 기본값

| 항목 | 설정 |
| --- | --- |
| CPU | 코어 고정 없이 4스레드, 기존 performance 정책 |
| Backbone | dark3까지만 NPU core0 실행 |
| FENet | 비INT8 RKNN, NPU core0에서 순차 실행 |
| HM | combined4, QP20 |
| 프레임 처리 | 최신 프레임 선택 후 순차 처리, 중간 파이프라인 미사용 |
| FCM 입력 | float32 NCHW `(1, 256, 80, 80)` |

## 검증 상태

2026-11-03 물리 보드 실험에서 위 NPU+combined4 순차 경로는 처리 중앙값 약 **0.83초**, 처리율 약 **1.15~1.17fps**였고 저장 샘플 복원도 통과했다. 초기 CPU FENet+기존 HM 구성은 약 1.69초·0.58fps였다.

**측정 범위는 선택한 프레임의 전처리 시작부터 압축 완료까지**이다. 센서 촬영부터의 지연이나 네트워크·서버를 포함한 E2E, 탐지 mAP, 소비 전력의 결과가 아니다. 단계별 실행 사이의 입력 영상도 달라질 수 있다.

`live_dark3_fcm.py`는 검증한 경로를 기본값으로 반영한 파일이다. 최종 기본 파일로 교체한 뒤의 확인 실행과 10분 안정성 결과는 아직 기록되지 않았다. 실험 경로 검증과 최종 파일의 장시간 운영 검증을 구분한다.

## 저장소 구성

```text
edge/darknet53/     최종 실행 파일, Backbone 변환, 라이브 A/B 실험
edge/fcm_tools/     FENet 변환, HM 비교, 압축·복원 검증
tools/             보드 배치 도구
docs/              실행 방법, 서버 연결 규격, 실험 해석
results/           터미널 출력에서 옮겨 적은 대표 측정 요약
tests/             배치 도구의 로컬 검증
```

## 시작하기

1. [보드 환경 및 실행](docs/board_setup.md)에 따라 기존 `~/yolox_project` 환경과 외부 모델·FCM 번들을 준비한다.
2. 아래 배치 명령으로 변경 목록을 확인한다. `--apply`를 붙일 때 기존 파일을 자동 백업하고 복사한다.

```bash
python3 tools/deploy_edge.py
python3 tools/deploy_edge.py --apply
```

3. 보드에서 실행한다.

```bash
source ~/yolox_project/.venv/bin/activate
cd ~/yolox_project/darknet53
export FCM_HM_DIR="$HOME/yolox_project/fcm_l37/fcm_bundle/hm"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u live_dark3_fcm.py --seconds 60
```

`--seconds 0`은 Ctrl+C까지 실행한다. 자동 부팅·백그라운드 서비스는 설정하지 않는다. 출력 디렉터리는 실행할 때마다 새로 생성된다.

## 팀 연결 문서

- [서버 전달 규격](docs/edge_server_contract.md): 전처리, dark3, FCM payload, 프레임 ID 규칙
- [모델 및 외부 파일](docs/artifacts.md): 별도 준비 파일, 해시, upstream
- [실험 결과와 결정](docs/experiments_2026-11-03.md): 네 가지 개선안과 선택 근거

모델·데이터셋·카메라 원본·HM 실행 바이너리·개인 접속 정보는 이 저장소에 포함하지 않는다. FCM 팀원의 번들과 각 upstream 코드는 별도로 준비한다.
