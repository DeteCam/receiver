# sim/pre: 부품을 사기 전의 검증

돈을 쓰기 전에 알 수 있는 것을 모은 곳이다. 해석은 [사전 검증](../../docs/사전%20검증.md), [숙소 방 계산](../../docs/숙소%20방%20계산.md), [사각지대](../../docs/사각지대.md)에 있다.

| 파일 | 하는 일 |
|---|---|
| `amp_stability.py` | 라즈베리파이(nexmon)로 받아 둔 CSI 기록에서, 이어진 패킷의 진폭이 얼마나 다른지와 네 번 재기를 흉내 냈을 때의 위상 오차를 낸다 |
| `collect.py` | `../rt/study/` 의 계산 결과(`log_pre.txt`, `room_*.json`)를 `result.json` 한 파일로 모은다 |
| `result.json` | 흔들림을 넣은 계산, 더 싼 구성, 세기만 재는 수신기, 숙소 방의 숫자 |
| `room_est.npz` | 숙소 방에서 카메라마다 고른 점과 오차 (그림에 쓴다) |
| `figs.py` | `docs/images/사전검증_*.png`, `숙소_방_계산.png` |
| `blind.json` | 사각지대의 숫자: 자리마다 맞힌 비율, 거는 곳 견주기, 패널 셋 (`../rt/blind_study.py`) |
| `blind_fig.py` | `docs/images/사각지대_자리지도.png` |
| `place.json` | 패널 자리 찾기(3차원)와 사진 16장의 숫자: 후보 122곳, 고른 배치, 새 카메라로 검증, 사진 1~32장 곡선, 대안 수신기, 나쁜 경우, 5G, 거는 각도 (`../rt/place_study.py`, `place_followup.py`) |
| `dual_band.py`, `dual_band.json` | 두 대역을 한 판으로 쓸 때의 위상 선 길이 고르기와 위상 단계 재기 (다른 방법 검토 5.5). scipy 가 있어야 해서 `../../../.venv-rt/bin/python` 으로 돌린다 |
| `place_fig.py` | `docs/images/배치_*.png`. `collect` 로 `../rt/study/place_*.json` 을 모으고 `figs` 로 그린다 |

```
python3 amp_stability.py 기록.pcap --mac 카메라주소      # 기록 파일은 내 기기의 주소가 들어 있어 올리지 않는다
cd ../rt && ./run_pre.sh && ./run_room.sh                  # 계산 (GPU, 약 25분)
cd ../pre && python3 collect.py && cp ../rt/study/room_est_ura4x4p37.5h.npz room_est.npz && python3 figs.py
```
