# firmware: 수신기 코드

패널의 계산과, 허브(라즈베리파이)가 쓸 탐지·판별·의심 점수 코드, 그리고 장치 없이 도는 검사다. ESP32-C5 에는 빠른 시험용 펌웨어(`c5/csi_probe`)만 있다. 패널 펌웨어와 허브 프로그램(패널에서 받기, 자리 내기, 폰 화면)은 아직 없다 ([스펙 006](../specs/006-panel-receiver/tasks.md)).

| 폴더 | 내용 |
|---|---|
| `core/` | `panel.py` (세기 → 위상 → 방향 → 자리), `identify.py` (판별: 움직임에 전송량이 따라오나), `score.py` (의심 점수). `../sim` 의 실험도 이 코드를 쓴다 |
| `pi/` | `detect.py` (둘레를 엿들어 카메라 후보를 추린다), `link_setup.sh` (허브의 무선: 자기 Wi-Fi 만들기, 탐지 동글을 모니터 모드로) |
| `host/` | `watch_report.py` (점검 기록을 의심 점수로 다시 계산해 실험 S1~S5 의 숫자를 낸다) |
| `c5/csi_probe/` | 빠른 시험용 ESP32-C5 펌웨어 (보내는 쪽 / 받는 쪽, USB 명령). 미리 빌드한 `bin/` 과 `flash.sh` |
| `bench/` | 빠른 시험 도우미 `probe.py` (기록, 안내, 판정), 도안 `도안.pdf` ([빠른 시험](../docs/빠른%20시험.md)) |
| `tests/` | 장치 없이 도는 검사 |

```
python3 tests/test_panel.py && python3 tests/test_score.py && python3 tests/test_detect.py && python3 tests/test_band.py && python3 tests/test_probe.py
python3 pi/detect.py --check            # 파이에서: 탐지 동글이 모니터 모드로 듣는지
```

- `pi/` 와 `core/` 는 파이썬 3.7 + numpy 1.16 (라즈베리파이 3B+) 에서도 돌아야 한다.
- 지금 파이에는 CSI 펌웨어 nexmon 이 깔려 있어 `apt upgrade` 를 하지 않는다 (nexmon 이 깨진다).
- 실물(파이, 동글, ESP32-C5)에서는 아직 돌려 보지 못했다.

돌판 수신기 때의 코드(허브 프로그램 `node.py`, 모터 `rx3.py`, nexmon CSI `csi.py`, 폰 웹 화면 `web.html`, 설치 `install.sh`, 방향·위치 계산 `analyze3`·`estimator`·`fusion`·`model`·`world`·`locate`, 노트북 도구, 그 검사)는 [receiver-legacy](https://github.com/DeteCam/receiver-legacy) 에 있다. 허브 프로그램을 새로 쓸 때 참고한다.
