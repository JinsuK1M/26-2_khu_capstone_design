# Mac 서버 환경

Apple Silicon MacBook, Python 3.12 가상환경에서 확인했다.

## 1. 폴더

```text
~/yolov5/          ultralytics yolov5 v7.0 (yolox_eval.py 가 val.py 를 사용), venv 포함
~/YOLOX/           Megvii YOLOX 공식 소스 (설치하지 않고 경로로만 사용)
~/fcm_bundle/      FCM 팀 번들 (fcm_edge.py, fcm_server.py, fcm_common.py, fenet_l37.pt, drnet_l37.pt, hm/cfg)
~/fcm_bundle/hm/bin/TAppEncoderStatic, TAppDecoderStatic   Mac 에서 빌드한 HM
~/yolov5/yolox_darknet.pth   공식 YOLOX-Darknet53 체크포인트
```

서버 코드(`server/*.py`)는 `~/yolov5`에 복사해서 실행한다. 경로를 바꾸려면 환경변수를 쓴다.

| 변수 | 기본값 |
| --- | --- |
| `YOLOX_DIR` | `~/YOLOX` |
| `FCM_BUNDLE_DIR` | `~/fcm_bundle` |
| `FCM_HM_DIR` | `<FCM_BUNDLE_DIR>/hm` |

모델 해시는 [모델 및 외부 파일](artifacts.md)의 기준값과 같은 것을 확인했다 (`yolox_darknet.pth`, `fenet_l37.pt`, `drnet_l37.pt`, `fenet_l37.onnx`).

## 2. HM-18.0 빌드 (Apple Silicon)

보드 ARM64 빌드와 같은 원인으로 기본 설정에서 실패한다.

```bash
cd ~/HM-18.0
sed -i.bak 's/warnings-as-errors//g' CMakeLists.txt                                        # 경고를 오류로 처리하지 않음
sed -i.bak2 's/add_compile_options( "-msse4.1" )/# arm64: SSE off/' CMakeLists.txt        # x86 전용 옵션 제거
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

출력 경로 이름에 `x86_64`가 들어가지만 실제 바이너리는 arm64이다. 번들 이름으로 복사한다.

```bash
B=~/HM-18.0/bin/umake/clang-*/x86_64/release
cp $B/TAppEncoder ~/fcm_bundle/hm/bin/TAppEncoderStatic
cp $B/TAppDecoder ~/fcm_bundle/hm/bin/TAppDecoderStatic
```

번들 `test_bundle.py` 재현: payload 17.76KB, 기준 복원값과 최대 차이 3.24e-05.

## 3. 패키지

yolov5 `requirements.txt` + `opencv-python`. YOLOX는 `pycocotools` 없이 쓰도록 클래스 이름을 소스에서 직접 읽는다.

## 4. 네트워크

- 서버를 처음 켤 때 macOS가 python 수신 연결 허용을 물으면 허용한다.
- 같은 와이파이: `ipconfig getifaddr en0` 주소를 Edge에 알려준다.
- 떨어져 있을 때: Tailscale에 Mac을 추가하고 `100.x.x.x` 주소를 쓴다.
- Mac이 잠자기에 들어가면 연결이 끊기므로 측정 중에는 전원 연결·덮개 열기.

## 5. 알려진 차이

Mac과 보드의 OpenCV 버전(보드 4.10.0)이 달라 같은 사진의 리사이즈 결과가 픽셀 최대 1 차이 났다 (zidane). 실제 시스템은 보드가 전처리하고 서버는 특징만 받으므로 동작에는 영향이 없지만, Mac에서 사진으로 재현할 때는 보드가 저장한 입력(`input_bgr_uint8_nhwc.npy`)을 쓰는 것이 정확하다.
