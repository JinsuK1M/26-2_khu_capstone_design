# Orange Pi 보드 환경과 실행

## 검증한 환경

| 항목 | 기록 |
| --- | --- |
| 보드 | Orange Pi 5 Pro, RAM 8GB |
| OS | Ubuntu 22.04.5, aarch64 |
| 커널 | 5.10.0-1012-rockchip |
| RKNPU 드라이버 | 0.9.6 |
| RKNN Toolkit2 / Lite2 / librknnrt | 2.3.2 |
| Python / PyTorch | 3.10 / 2.2.0 |
| NumPy / OpenCV | 1.26.4 / 4.10.0 |
| 카메라 | OV13855, CAM1 |

아래는 이미 검증한 보드 환경에 코드를 배치하는 방법이다. 이 저장소만으로 OS·카메라·NPU 라이브러리와 모델이 자동 설치되지는 않는다.

## 필요한 실행 디렉터리

```text
~/yolox_project/
  .venv/
  darknet53/
    live_dark3_fcm.py
    dark3_only/dark3_fp.rknn
  fcm_l37/
    verify_live_fcm.py
    fcm_bundle/
      fcm_edge.py
      fcm_server.py
      fcm_common.py
      fenet_l37.pt
      drnet_l37.pt
      fenet_l37.onnx
      fenet_npu_candidate/
        fenet_fp.rknn
        build.json
      hm/
        cfg/encoder_intra_main_rext.cfg
        bin/TAppEncoderStatic
        bin/TAppDecoderStatic
```

FCM 원본 코드·모델·설정은 팀원 번들에서 준비한다. 배치 도구는 이 저장소의 Python 스크립트만 복사하며 모델·번들은 설치하거나 덮어쓰지 않는다. 현재 코드가 `~/yolox_project`를 기준으로 하므로 실제 보드 실행에는 위 경로를 사용한다.

## 카메라 사전 확인

검증한 CAM1 오버레이는 다음 파일이다.

```text
/lib/firmware/5.10.0-1012-rockchip/device-tree/rockchip/overlay/orangepi-5-pro-cam1.dtbo
```

기존 보드에서는 `/etc/default/u-boot`의 overlay 설정 후 `u-boot-update`로 적용했다. 커널·OS가 다르면 설치된 오버레이와 센서 지원을 먼저 확인한다. 자동 생성된 extlinux.conf를 직접 덮어쓰지 않는다.

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video11 --get-fmt-video
```

기존 코드가 확인하는 값은 1280×960 NV12, stride 1280, Size Image 1843200이다. 다른 카메라 프로그램과 동시에 장치를 점유하지 않는다.

## 배치와 확인

```bash
# 저장소 최상위에서 실행. 첫 명령은 계획만 표시한다.
python3 tools/deploy_edge.py
python3 tools/deploy_edge.py --apply

source ~/yolox_project/.venv/bin/activate
cd ~/yolox_project/darknet53
export FCM_HM_DIR="$HOME/yolox_project/fcm_l37/fcm_bundle/hm"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -u live_dark3_fcm.py --seconds 60
```

시작 로그에서 `HM profile: combined4`, `FENet backend: npu`, `Execution: sequential`을 확인한다. CPU affinity를 변경하는 taskset 명령은 사용하지 않는다.

배치 백업은 `~/yolox_project/.edge-backups/시간/`에 저장되며 manifest에는 원본·복사 후 해시가 기록된다. 복원하려면 해당 백업의 상대 경로에 있는 파일을 같은 실행 경로로 돌려놓는다.

## 결과 검사

종료한 실행 폴더의 `report.json`, `frames.csv`, 마지막 payload·dark3·입력 영상과 일부 샘플을 확인한다. 첫 프레임은 보관하되 `after_first_frame` 통계에서는 제외한다.

```bash
cd ~/yolox_project/fcm_l37
# RUN_DIRECTORY를 해당 실행 폴더의 실제 경로로 바꾼다.
OMP_NUM_THREADS=4 python -u verify_live_fcm.py --run RUN_DIRECTORY
```

60초 확인 후 `--seconds 600`으로 10분 안정성을 측정한다. 현재까지 저장된 60초 실험 결과가 장시간 안정성이나 mAP 검증을 대신하지 않는다.
