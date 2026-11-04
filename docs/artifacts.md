# 모델과 외부 파일

## 별도 준비 항목

| 파일·코드 | 준비 위치·역할 |
| --- | --- |
| YOLOX 공식 소스 | https://github.com/Megvii-BaseDetection/YOLOX |
| yolox_darknet.pth | 공식 pretrained YOLOX-Darknet53 체크포인트 |
| RKNN SDK 2.3.2 | https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2 |
| FCM 팀 번들 | fcm_edge.py, fcm_server.py, fcm_common.py, TorchScript·ONNX 및 HM 설정 |
| HM-18.0 소스·실행 파일 | 팀 전달 소스에서 보드 ARM64용으로 빌드 |
| dark3_fp.rknn | dark3_only.py로 변환한 보드 모델 |
| fenet_fp.rknn 및 build.json | fenet_npu_candidate.py로 변환한 후보 모델·해시 기록 |
| 검증 입력 | 팀 공유 test_l37.npy, zidane 샘플 및 저장 카메라 dark3 |

체크포인트 공식 주소:
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth

기록한 YOLOX revision은 `6ddff4824372906469a7fae2dc3206c7aa4bbaee`이다. 원본 가중치를 strict load하며, 다른 모델이나 랜덤 가중치를 pretrained 결과로 대체하지 않는다.

## 기준 SHA256

| 항목 | SHA256 |
| --- | --- |
| yolox_darknet.pth | b5905e9faf500a2608c93991f91a41a6150bcd2dd30986865a73becd94542fa1 |
| dark3_only/dark3_fp.rknn | c8070f5c838874b96cba8045b90ef6e03304659407a81360b694ad43c9981ce7 |
| fenet_l37.pt | b5d560d01c8ea7135dcb07b0add644742f3bac3003da1d5af6ab14c720b83393 |
| drnet_l37.pt | e027a604beabfe9c62603eb9ec383a5bc5fbeb3ba0fcdca2a889ab1a061ee3d9 |
| fenet_l37.onnx | 756688550f31e7fb77de7f025c1a2210122025239a8f2915d06a86e857191aab |
| fenet_fp.rknn | ebf4e0da222236758f347ad9008147723a7efd71cca4c9b2505f40cb6e8b37b8 |
| zidane.jpg | 356dad2107bb0254e4e4a81bc1d9c7140043e88569d546e5b404b19bffa77d0a |

위 값은 검증한 모델 파일의 식별값이다. 모델을 다시 생성하면 해시가 달라질 수 있으므로 새 생성물에는 빌드 기록과 수치 비교 결과를 함께 남긴다.

## 변환 스크립트

보드의 실행 경로에 스크립트와 외부 파일을 준비한 뒤 실행한다.

```bash
cd ~/yolox_project/darknet53
python test_backbone.py export
python test_backbone.py build
python test_backbone.py test
```

이 단계는 검증용 전체 Backbone+내부 SPP이다. 최종 Edge 분할과 구분한다. export는 인접한 `rknn_model_zoo/examples/yolox/model/bus.jpg`를 입력 예제로 사용한다.

```bash
python export_zidane_dark3.py
python dark3_only.py export
python dark3_only.py build
python dark3_only.py test

cd ~/yolox_project/fcm_l37/fcm_bundle
python fenet_npu_candidate.py inspect
python fenet_npu_candidate.py build
python fenet_npu_candidate.py test
```

dark3_only의 export/test는 기존 전체 Backbone ONNX·RKNN과 zidane 샘플을 필요로 한다. FENet inspect/test는 번들 입력, zidane dark3, 고정 카메라 특징의 세 입력을 사용한다. 카메라 표본 기본 경로는 `live_fcm_runs/20261103-155444-451889/last_dark3.npy`이며, 다른 보드에서는 공유 표본을 배치하거나 경로를 조정해야 한다.

## 외부 코드

YOLOX·RKNN·HM·FCM 팀 번들 및 fctm 유래 함수는 각 제공처의 배포물로 관리한다. 이번 등록에는 해당 소스·가중치·바이너리를 포함하지 않았다. 같은 배포물과 해시를 사용해 팀의 실행 환경을 맞춘다.
