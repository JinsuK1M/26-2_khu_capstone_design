EVA-01 k4 실제 보드 특징 수집

이 코드는 학습을 수행하지 않는다. 실제 보드에서 검증한 layout v2의 앞4블록 출력을
float16 NCHW (1,1408,22,22) NPY로 저장하여 FCM 담당자에게 전달하는 도구다.
새 학습 데이터는 COCO train2017에서 고정 seed20261124로1000장을 선택한다.
기존 val2017 평가500장, 실험1/2, v2 모델·측정 결과는 수정하지 않는다.

전처리: OpenCV INTER_LINEAR 직접336x336 -> RGB -> 기존 mean/std 정규화 ->
오른쪽/아래16픽셀을 정규화값0으로 채워352x352. 원본 종횡비 유지 resize와 다르다.
이 조건은 현재 v2 검증과 같은 조건이므로 FCM 담당자에게 그대로 공유해야 한다.
현재 특징의 호환성이 새 데이터의 최종 탐지 정확도나 mAP를 보장하지 않는다.

보드 작업 폴더: ~/yolox_project/experiments/eva01_feature_collection
사용하는 기존 모델: ~/yolox_project/experiments/eva01_board_k4_layout_v2

1. 학습 이미지1000장 준비:
  python -u src/prepare_train_images.py --count 1000
기존 ~/yolox_project/evaluation/coco2017/annotations_trainval2017.zip을 우선 재사용한다.
없으면 공식 정답ZIP 약250MB를 다운로드한다. 전체 train 이미지18GB를 받지는 않는다.
선택된1000장만 공식 COCO 버킷의 HTTPS 주소에서 받는다. TLS 검증은 끄지 않는다.
준비가 중단되면 같은 명령을 다시 실행해 이미 받은 이미지를 재사용한다.

2. 실제 보드에서20장 추출 및 전달 묶음 만들기:
  python -u src/extract_features.py --limit 20 --pack
워밍업3회 후 실제 보드 출력이며 파일명은 COCO image ID다. 사진당 약1.30MiB.
CPU 기준 특징과 가깝다는 잠정 검사는 기존 bus 사진1장으로만 수행한다.
데이터셋20장의 CPU 기준 특징 또는 최종 탐지 정확도를 비교한 결과는 아니다.

3. 정상 완료 후 같은 목록의1000장까지 이어 추출:
  python -u src/extract_features.py --resume-last --limit 1000 --pack
이미 저장한20장은 파일 해시/형상/dtype 검사 후 건너뛴다. 모델과 데이터 목록이
바뀌면 이어 추출을 거부한다. 중간 Ctrl+C 시 같은 명령으로 재개할 수 있다.
파일 읽기/저장/압축을 제외한 추론만 현재 약25분 예상이며 실제 완료 시간은 더 길다.
FP16 원시 특징1000장은 약1.27GiB. 이미지·NPY헤더·전달 묶음 저장 공간이 추가된다.

실제 결과: runs/features_<시각>/features/<12자리 image ID>.npy
추출 목록: 동일 폴더 index.jsonl (이미지/특징 해시, shape,dtype, FP16 저장 오차)
설정 및 원본 목록: metadata.json, contract.json, source_manifest.json
실행 상태: report.json (completed는 요청한 --limit 개수의 수집 완료를 뜻함)
서버 전달 파일: transfers/features_<시각>_<개수>frames.tar.gz
tar.gz는 전달용 묶음이며 학습한 FCM이나 프로젝트 코덱의 압축률이 아니다.

FCM 담당자에게는 producer=actual_board_rknn_layout_v2임을 함께 전달한다.
CPU 기준 특징이 아니라 NPU/CPU 혼합 RKNN 경로에서 실제 생성한 특징이다.
처음20장으로 팀원 데이터 로더의 (1,C,H,W)와 float16 처리 여부부터 확인하면 좋다.
학습 때 float32가 필요하면 로더에서 변환하고, 저장 FP16을 무조건 연산 FP16으로
해석하지 않는다. 추가 메타데이터 없이 특징을 재정규화하거나 축을 바꾸지 않는다.

검증 수준: 로컬에서 기존 검증 사진의 전처리 및 저장/데이터 보호 로직 검사.
이후 실제 보드에서20장 점검 후1000장까지 재개해 수집·전달 묶음 생성 완료.
상세 기록: ../results/feature_collection_1000_20261124.json. 팀원 수신·학습은 아직 확인 전.
새로 설치하는 패키지 없음. 기존 보드 환경의 numpy, OpenCV, RKNNLite 및 curl 사용.

공식 출처:
https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm
https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
https://s3.amazonaws.com/images.cocodataset.org/train2017/
