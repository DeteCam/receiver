# firmware: 수신기 코드

라즈베리파이에서 도는 코드와 계산, 검사입니다. 설치, 폰 화면, 주소(API)는 [혼자 동작](../docs/혼자%20동작.md)에 있습니다.

| 폴더 | 내용 |
|---|---|
| `pi/` | 수신기에 넣는 것: `node.py`(전체), `rx3.py`(모터), `csi.py`, `detect.py`, `web.html`(폰 화면), `marker.json`, `install.sh`, `link_setup.sh` |
| `core/` | 계산: 방향(`analyze3`, `fusion`, `estimator`, `model`), 추정 구간(`locate`), 판별(`identify`), 의심 점수(`score`), 가짜 방(`world`). `../sim` 의 실험도 이 코드를 씁니다 |
| `host/` | 개발할 때 노트북에서: `scan3.py`, `hand_check.py` |
| `tests/` | 장치 없이 도는 검사 |

```
python3 pi/node.py --dry --port 8080     # 장치 없이 화면 보기: http://localhost:8080
python3 tests/test_core.py && python3 tests/test_score.py && python3 tests/test_exam.py && python3 tests/test_node_dry.py && python3 tests/test_control.py && python3 tests/test_marker.py
pi/install.sh pi@raspberrypi.local       # 파이에 넣기 (랜선을 꽂은 채로)
```

- 파이에서는 `apt upgrade` 를 하지 않습니다 (CSI 펌웨어 nexmon 이 깨집니다).
- `pi/` 와 `core/` 는 파이썬 3.7 + numpy 1.16 에서 돌아야 합니다.
- 실물(파이, 모터, 동글)에서는 아직 돌려 보지 못했습니다.
