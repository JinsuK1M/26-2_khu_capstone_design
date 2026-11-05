# HW FCM 서버 어댑터 (검증 후보)

기존 서버 파일은 덮어쓰지 않는다. 서버 담당 브랜치 `server/mac-yolox-fcm`의 `d5bd80bf843b0e0dbbb0debe5d31dda99eb44a6b`에 있는 `yolox_server.py`, `yolox_split.py`와 같은 폴더에 이 파일을 복사한다. 해당 브랜치의 모델·FCM 번들·PyAV 환경을 사용한다.

```bash
export FCM_BUNDLE_DIR="/actual/path/fcm_bundle"
python yolox_hw_server.py --weights /actual/path/yolox_darknet.pth --port 9997 --conf 0.25
```

`fcm_hw_v1`만 새 경로를 사용한다. 기존 JPEG·영상 HEVC·SW FCM·tensor는 원래 서버로 위임한다. 기존 9997 서버를 중지한 뒤 시작한다. 헤더·복원 계산은 [통합 문서](../../docs/hw_fcm_integration.md)를 따른다. 모델과 번들은 저장소에 포함하지 않는다.
