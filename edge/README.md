# Edge 코드 안내

기존 SW 카메라 경로의 실행 진입점은 `darknet53/live_dark3_fcm.py` 하나이다. 기본값은 FENet NPU, HM combined4, QP20, 4스레드, 순차 실행이다.

## 실행·실험 스크립트

| 파일 | 용도 |
| --- | --- |
| live_dark3_fcm.py | 최종 기본 설정, 기본 30초 또는 --seconds 지정 |
| live_dark3_fcm_hm_compare.py | HM 비교용, 기본 CPU FENet·기존 HM |
| live_dark3_fcm_fenet_compare.py | CPU/NPU 비교용, 기본 CPU FENet·기존 HM; 비교 runner가 옵션 지정 |
| live_dark3_fcm_pipeline.py | 제외한 파이프라인 실험. 최종 기본 경로가 아님 |
| run_live_hm_compare.py | HM A/B/A 및 저장 샘플 복원 |
| run_live_fenet_compare.py | FENet CPU/NPU/CPU 및 저장 샘플 복원 |
| run_live_pipeline_compare.py | 순차/파이프라인/순차 및 저장 샘플 복원 |
| test_backbone.py | 원본 전체 Backbone 내보내기·합성·수치 검증 |
| dark3_only.py | 기존 ONNX의 dark3 subgraph 추출·RKNN 변환·비교 |
| export_zidane_dark3.py | 팀 간 동일 입력·특징 확인용 자료 생성 |
| live_backbone.py | 초기 전체 Backbone 카메라 실험, 최종 FCM 경로와 구분 |
| camera_preview.py | 초기 live_backbone 뷰어 보조 모듈 |

`fcm_tools`의 검증·변환 도구는 [배치 도구](../tools/deploy_edge.py)가 필요한 실행 폴더로 복사한다. 모델·팀원 FCM 번들이 있어야 하며, 저장소의 소스 검사 통과가 보드 테스트를 대신하지 않는다.

## 2026-11-06 추가

저장 입력 HW FCM은 `darknet53/replay_fcm_hw_server.py`이며 기존 SW 기본 파일을 대체하지 않는다. [HW 통합 문서](../docs/hw_fcm_integration.md)에 의존 파일·서버 어댑터·실행 명령·대역폭 제한 및 복구를 기록했다. `network_test/board_send.py`도 배치 도구가 복사한다. 네트워크 주소는 `--host`로 명시한다.
