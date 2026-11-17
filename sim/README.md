# sim: 시뮬레이션

패널 수신기가 교실과 숙소 방에서 카메라의 자리를 얼마나 맞히는지, 장치 없이 전파 경로 계산(Sionna RT)으로 돌려 본다. 판정 코드(판별, 의심 점수)는 `../firmware/core` 의 것을 그대로 쓴다 (`paths.py` 가 찾는다). 숫자의 뜻과 결론은 [설계 검토](../docs/설계%20검토.md), [사전 검증](../docs/사전%20검증.md), [사각지대](../docs/사각지대.md)에 있다. 파일 하나하나는 [폴더와 파일](../docs/폴더와%20파일.md).

## 폴더

| 폴더 | 묻는 것 | 스크립트 → 결과 |
|---|---|---|
| `rt/` | 40명이 앉은 교실(과 72석 강의실)에서 패널로 카메라 자리를 얼마나 맞히나, 숙소 방에서 숨긴 카메라를 얼마나 찾나 | `rt_paths.py` → `rt_study.py`, `room_paths.py` → `room_study.py`, `blind_study.py` → [패널 연구](rt/패널%20연구.md) |
| `pre/` | 부품을 사기 전에 알 수 있는 것, 사각지대 | `amp_stability.py`, `collect.py` → `result.json`, `figs.py`, `blind.json`, `blind_fig.py` |
| `identify/` | 판별: 몇 초씩 몇 번, 얼마나 잘못 잡나 | `identify_sim.py` → `identify_result.json` |

결과 json 은 그것을 만든 스크립트 옆이나 `pre/` 에 있고, 그림은 `../docs/images/` 에 쓴다.

```
cd rt && ./run_pre.sh && ./run_room.sh          # 패널 계산 (GPU, 약 25분). 돌리는 법은 rt/패널 연구.md
python3 identify/identify_sim.py 600            # 판별 흉내 (몇 분)
```

## 읽을 때 조심할 것

- 전부 계산 모형이다. 실제는 더 나쁠 수 있다. 사람 모형은 단순하고, 교실의 책상·사물함·창은 실제와 다를 수 있다.
- 숫자는 카메라 수에 따라 몇 점씩 흔들린다. 2026-11-18 에 교실의 주요 숫자를 카메라 1,500대로 다시 셌다 (처음에는 210대). 강의실은 아직 처음 계산이다.
- 돌판 수신기 때의 계산(`direction/`, `location/`, `em/`, `rt/rt_eval.py`)은 [receiver-legacy](https://github.com/DeteCam/receiver-legacy) 에 있다. 돌판이 왜 두 곳으로는 모자랐는지의 바탕이다.
