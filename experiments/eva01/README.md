# EVA-01 보드 분할 실행 검증

현재 선택한 보드 경로는 `eva01_board_k4_layout_v2`다. 공식 COCO 탐지 가중치의 앞 4개 블록을 RKNN으로 실행했다. 서버 담당자 메시지로 사진1장의 무압축 보드 특징→나머지 모델 연결 성공을 확인했다. EVA용 FCM 학습·통합·mAP는 아직 확인하지 않았다. 기존 YOLOX 실험과 별도 폴더로 관리한다.

## 구현과 검증 범위

| 구분 | 확인한 내용 | 해석 범위 |
|---|---|---|
| 보드 전체 백본 CPU | 실제 가중치 40블록, 합성 640 입력 1회, 74.20초 | 전체 백본은 실행 가능. FPN/RPN/ROI 포함 전체 검출기 실행 결과는 아님 |
| 맥북 전체 검출기 CPU | M2 Pro 16GB, 실제 사진 640 입력, 17.56초, 12개 탐지와 mask | 사용자 맥북 검사. 팀원 서버 재현·분할·FCM·실시간·mAP 검증은 아님 |
| 보드 k=4 | 실제 사진 입력, CORE0, 특징 형상 및 CPU 기준 수치 비교 통과 | CPU와 NPU 혼합 실행. 잠정 특징 검사이며 최종 탐지 정확도 합격을 의미하지 않음 |
| 서버 분할 연결 | 서버 담당자 메시지로 사진1장 성공 확인 | reference33개가 전체 실행과 동일. NPU30개 중29개가 reference와 같은 클래스로 대응. mAP/FCM 검증 아님 |
| EVA용 FCM | 아직 미통합 | 팀원 학습 후 FENet/DRNet 및 실제 코덱 연결 필요 |

이 결과로 “EVA-01 전체 모델은 보드에서 실행 불가능하다”고 주장할 수 없다. 실제로 전체 백본 CPU 실행은 성공했다. 640 전체 백본과 352 앞 4블록은 범위와 입력이 다르므로 속도 향상 배수로 비교하지 않는다.

## k=4 보드 결과

공통 조건: RK3588, RKNN Toolkit2/Lite2 2.3.2, CORE0, INT8 양자화 미사용, 동일 실제 탐지 가중치와 사진, 워밍업 3회/측정 5회. 아래 시간은 RKNN API 호출만이며 전처리·출력 비교·FCM·압축·전송·서버·전체 탐지·mAP는 제외한다.

| 버전 | 변경 이유/방법 | 중앙값(ms) | p95(ms) | 특징 상대 L2 | CPU 배치 연산 | 판단 |
|---|---|---:|---:|---:|---|---|
| baseline | 공식 attention 표현을 그대로 변환 | 3281.89 | 3615.34 | 0.008459 | Transpose 3 + Add 8 + Reshape 4 | 실행 성공, 느림 |
| layout v2 | 같은 계산을 QKV 1×1 Conv와 3차원 위치 정보 계산으로 표현 | 1475.64 | 1497.69 | 0.008431 | Transpose 8 | 통합 검증용 선택 |
| layout v3 | v2의 큰 재배치를 작은 조각으로 분할 | 1482.87 | 1508.45 | 0.008431 | Transpose 8 | 추가 개선 없음, 비교 기록 보존 |

CPU 배치 수는 컴파일러 target 표에서 Input/Output 행을 제외한 값이다. 개별 연산 실행 시간을 측정하지 않아 전체 지연의 기여도는 미확정이다. `3Core fallback to single core`는 NPU 코어 수 경고이며 CPU fallback과 구분한다. v2/v3는 별도 시점에 한 번씩 측정했으므로 약 0.5% 차이의 통계적 의미는 주장하지 않는다. v3는 맥북 ONNX CPU의 사진·고정 난수 입력에서 v2와 차이가 0이었지만 보드 속도 개선으로 이어지지 않았다.

## 분할 규격

| 항목 | 규격 |
|---|---|
| 공식 모델 | EVA-01 COCO detector, patch16, ViT 40블록, 채널1408, head16 |
| 원본 체크포인트 파라미터 | 전체 검출기 1,073,745,214 / 백본 1,011,754,368 |
| 보드 담당 | PatchEmbed + 절대 위치 정보 + blocks[0..3]의 완전한 residual 출력 |
| 앞 4블록 파라미터 | 102,399,552 |
| 검증 입력 준비 | 사진 직접 336×336 resize, BGR→RGB, mean/std 정규화, 오른쪽·아래 16픽셀 정규화값0 패딩 |
| RKNN 입력 버퍼 | 정규화된 float32 `(1,3,352,352)`; 호출 시 NHWC로 배치 변환, RKNN mean0/std1 |
| 앞 4블록 attention | window16, window16, window16, global |
| 서버 전달 특징 | float32 NCHW `(1,1408,22,22)`; 원시 특징 2,725,888바이트 |
| 서버 재시작 | blocks[4]부터. PatchEmbed와 절대 위치 정보를 다시 적용하지 않음 |
| CLS 토큰 | 이 탐지판의 런타임 특징에는 별도 CLS 토큰 없음 |

336/352 전처리는 변환 가능성 확인용 잠정 조건이다. 팀원 FCM 학습 및 실제 카메라 전처리와 합의해 고정해야 한다. 공식 1536 검출 정확도 조건과 다르다. 현재 YOLOX용 FENet/DRNet 가중치의 EVA 재사용은 검증되지 않았다.

## 폴더

```text
compatibility/src/                원본 백본 CPU 검사, 맥북 검출기 검사, baseline/v2 ONNX 추출
compatibility/run_board_backbone.sh  보드 전체 백본 CPU 검사
compatibility/configs/artifacts.json 외부 소스·모델 URL와 해시

eva01_board_k4/                   baseline RKNN 빌드/실행 코드와 계약
eva01_board_k4_layout_v2/         선택한 보드 버전과 전달 파일 묶기 도구
eva01_board_k4_layout_v3/         개선되지 않은 분할 재배치 비교 후보
results/                         보고서용 실제 측정 JSON; 학습 가중치와 구분
```

## 기존 검증 환경에서 재현

모델·이미지·ONNX·RKNN·NPY·원본 로그·vendor 체크아웃은 Git에 포함하지 않는다. 아래는 기존 EVA 환경을 재사용하는 절차다. 새 장비의 Detectron2 설치 전체를 자동화한 스크립트는 아니다. Python 패키지 환경을 그대로 유지한다.

1. 기존 EVA 원본 폴더를 연결한다. `--existing-root`에는 `models/eva_coco_det.pth`, `vendor/EVA`, `vendor/bus.jpg`, `vendor/timm_0_6_11_mlp.py`가 있는 폴더를 지정한다. 소스 revision과 파일 해시를 검사한 뒤 심볼릭 링크만 만든다.

```bash
python experiments/eva01/compatibility/src/prepare_assets.py --existing-root /기존/EVA_검사폴더
```

2. 기존 EVA 맥북 환경에서 앞부분을 내보낸다. 원래 `img_size=1280`은 체크포인트 위치 파라미터 형상 유지를 위해 유지하며 실제 입력은 352다.

```bash
python -u experiments/eva01/compatibility/src/export_k4_layout_v2.py
```

생성 위치는 `experiments/eva01/eva01_board_k4_layout_v2/`이다. ONNX와 입력/CPU 기준 특징을 보드의 동일 실험 폴더로 별도 전달한다. 이미 v2 ONNX/입력이 보드에 있다면 재추출할 필요 없다. Git에 있는 `configs/contract.json`은 당시 생성물의 기록이며 재생성 해시가 다르면 임의로 바꾸지 말고 코드·환경 차이를 먼저 확인한다.

3. 보드 가상환경에서 실행한다.

```bash
source ~/yolox_project/.venv/bin/activate
cd ~/yolox_project/experiments/eva01_board_k4_layout_v2
bash run_board.sh
```

`run_board.sh`는 기존 보드 Python 경로를 사용한다. 결과는 매번 새 `runs` 폴더에 저장한다. v3는 형제 v2 폴더의 ONNX와 입력을 읽어 생성하므로 baseline이나 v2 파일을 덮어쓰지 않는다. v3는 선택한 통합 버전이 아니다.

## 다음 단계: 서버 담당자에게 무압축 특징 전달

보드 담당 범위는 실제 특징·입력·계약·측정 결과 전달까지다. 서버 담당자는 같은 가중치로 CPU 기준 특징과 보드 특징을 각각 `blocks[4]`부터 최종 검출까지 실행해 비교한다. 아직 서버 뒷부분 구현이나 학습된 FCM이 이 저장소에 추가된 것은 아니다.

새 전달 도구를 기존 보드 v2의 `src`에 복사한 뒤 실행한다.

```bash
python -u src/pack_server_sample.py --run runs/npu_20261124_153821_484855
```

생성된 `transfers/<시각>/transfer_k4_v2.tar.gz`와 SHA256을 전달한다. 도구는 계약/실행 결과 일치, 입력 및 CPU 기준 해시, 보드 특징 shape/dtype/유한값을 검사하고 파일별 해시 manifest를 포함한다. 원본 실험 결과를 수정하지 않는다. tar.gz는 파일 묶음용이며 학습한 FCM 압축이 아니다.

서버 검증 후 팀원이 학습한 FENet→코덱→DRNet을 연결하고, 같은 전처리·분할·평가 입력에서 압축 전후 검출 결과와 전송량·시간을 비교한다. CPU/NPU 혼합 자체는 실험 실패가 아니며 실행 구성을 명시한다.

## 환경과 외부 코드

- 보드: Linux5.10.0-1012-rockchip/aarch64, RAM 약7.7GiB+swap1GiB, Python3.10, PyTorch2.2.0, NumPy1.26.4, RKNN2.3.2/runtime2.3.2/driver0.9.6.
- 맥북 확인 장비: M2 Pro/16GB, Python3.12.0, PyTorch2.2.2, torchvision0.17.2, timm0.9.16, Detectron2 0.6(EVA 제공 소스). 팀원 M3 Pro 서버 결과로 쓰지 않는다.
- Detectron2 CPU 빌드는 기존 검증 시 `CFLAGS`/`CXXFLAGS=-Wno-invalid-specialization`, `FORCE_CUDA=0`, `MAX_JOBS=2`를 사용했다. 새 환경에서는 별도 검증이 필요하다.
- `check_mac_detector.py`의 미설치 Soft-NMS import guard는 호출 시 명시적으로 실패한다. 실제 검사는 torchvision CPU 기본 NMS/ROIAlign 경로다. Soft-NMS가 구현됐다는 의미가 아니다.
- upstream: [EVA](https://github.com/baaivision/EVA), [공식 탐지 가중치](https://huggingface.co/BAAI/EVA/blob/main/eva_coco_det.pth), [timm v0.6.11 MLP](https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py). 제3자 코드는 별도 준비하며 해당 라이선스를 따른다.


## 추가 진행: 무압축 연결 확인과 실제 보드 학습 특징 수집

서버 담당자 메시지에서 같은 체크포인트로 두 특징의 최종 검출 성공이 확인됐다. reference는 전체 실행과33개 모두 점수·박스 차이0이었다. 점수0.3 이상에서 reference33개/NPU30개 중29개가 같은 클래스로 대응했고, 대응한 검출은 최대 점수 차0.028·최대 박스 차1.46px·최소IoU0.98이었다. reference에만4개, NPU에만1개가 있었다. 서버 뒷부분 시간 약4.1~4.4초는 사진1장의 안내값이며 정식 반복/E2E 측정이나 mAP가 아니다. [메시지 근거 기록](results/server_split_message_20261124.json)을 별도 보존했다.

FCM 담당자가 요청한 학습 특징은 [feature_collection/README.txt](feature_collection/README.txt)의 코드로 실제 보드에서 직접 수집할 수 있다. COCO train2017의 고정1000장을 준비하고 기존 v2 RKNN으로 float16 NCHW (1,1408,22,22)를 저장한다. 먼저20장으로 저장 규격을 확인한 뒤 같은 수집 폴더에서1000장까지 재개할 수 있다. 기존 val2017 평가500장은 사용하지 않는다. 로컬 전처리·데이터 보호·저장·가상 런타임 재개 검사를 거친 뒤, 실제 보드에서20장 점검 및1000장 수집까지 완료했다. NPY합계1,363,072,000B, 저장된 FP16의 추가 반올림 상대L2 최대값0.0이다. [실제 보드 수집 기록](results/feature_collection_1000_20261124.json)에 전달 묶음의 보고된 SHA256을 남겼다. 팀원 수신·학습은 아직 확인 전이다. 기존336/352 전처리 조건을 FCM 담당자와 공유한 뒤 학습에 사용해야 한다.
