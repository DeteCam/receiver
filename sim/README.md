# sim: 시뮬레이션

판을 돌려 방향을 찾는 방법이 얼마나 맞는지, 방과 곳에 따라 어떻게 달라지는지 장치 없이 돌려 봅니다. 계산 코드(방향·위치 추정, 판별)는 `../firmware/core` 의 것을 그대로 씁니다 (`paths.py` 가 찾습니다). 숫자의 뜻과 결론은 [설계 검토](../docs/설계%20검토.md)와 [네 단계 검토](../docs/네%20단계%20검토.md)에 있습니다.

```
pip install numpy matplotlib
python3 places.py 100          # 곳별 결과를 100번씩 (몇 분)
```

## 실험

| 파일 | 묻는 것 | 결과 파일 |
|---|---|---|
| `validate.py` | 회절 모형이 손으로 잰 곡선과 맞나 | `validate.json` |
| `montecarlo.py` | 한 자리에서 방향을 얼마나 맞히나 (재는 방법 6가지) | `result.json` |
| `room.py`, `room_extra.py` | 방의 벽 반사와 사람 몸을 넣으면 | `room_result.json` |
| `locate3d.py`, `locate3d_var*.py`, `twopos.py` | 여러 자리를 합쳐 위치를 얼마나 맞히나 | `locate3d_*.json` |
| `places.py` | 변기 칸, 욕실, 모텔 방, 강의실에서는 | `places_result.json` |
| `identify_sim.py` | 판별: 몇 초씩 몇 번 해야 하나, 얼마나 잘못 잡나 | `identify_result.json` |
| `figures.py`, `fig_locate.py`, `fig_places.py` | 결과 그림 | `../docs/그림/` |

## em/: 전파 계산 (NEC2)

수신기 전체(파이 기판, 모터, 동전 균형추, 드라이버, 요크, 배터리, 판)를 금속 선 그물로 만들어 안테나 프로그램 NEC2 로 풉니다. 위의 실험과는 따로 만든 계산이라 서로 견줄 수 있습니다.

```
pip install PyNEC                 # NEC2 (모멘트법)
cd em
python3 test_compare.py           # 넣어 둔 전파 표로 계산이 도는지 (1분)
python3 compare_real.py 내려받은파일.json --az 0 --el 0      # 실물로 잰 곡선을 모형들과 견준다
```

| 파일 | 내용 |
|---|---|
| `device.py`, `necgeo.py` | 수신기의 전파 모형 (치수는 `../../cad/p2.py` 에서) |
| `run_tables.py`, `tables.py` | 방향마다, 판 각마다의 받는 세기를 표로 (`tables/` 에 줄여서 넣어 둠. 전체를 다시 풀면 몇 시간) |
| `study.py`, `tmatch.py` | 그 표로 재는 과정을 흉내 내고, 곡선 전체를 맞추는 계산으로 방향을 구한다 |
| `own_parts.py` → `own_parts.json` | 자기 부품이 얼마나 방해하나: 전체를 넣은 기준 곡선으로는 3.6°, 기판만 넣은 것으로는 48° |
| `mismatch.py` → `mismatch.json` | 안테나 모형이 실제와 다르면 |
| `geo_eval.py` → `geo_eval.json` | 판 크기와 팔 길이 13가지 |
| `plan_eval.py` → `plan_eval.json` | 세 번과 여섯 번: 반사가 심한 방에서 크게 틀리는 비율 30% → 16% |
| `metrics.py` → `metrics.json` | '가장 깊은 골'만 읽으면 어떻게 되나 |
| `compare_real.py` | **첫 실물 실험에 쓰는 도구**: 잰 곡선이 어느 모형과 닮았는지, 그 모형으로 방향이 맞는지 |
| `fdtd/` | 같은 문제를 다른 풀이법(gprMax, FDTD)으로 푼 입력 파일. 두 도구의 답이 같은지 본 것 |
| `fig_em.py` | 그림 (`../../docs/그림/17_전파_검토.png`) |

## 읽을 때 조심할 것

- 전부 계산 모형입니다. 실제는 더 나쁠 수 있습니다.
- 위의 '실험'은 골 모양을 두 가지로 가정합니다: W = 회절 모형 그대로(나쁜 쪽), V = 손으로 잰 곡선과 비슷한 모양(좋은 쪽). 편파, 파이 기판과 모터가 만드는 반사는 들어 있지 않습니다. 그것은 `em/` 이 봅니다.
- `em/` 은 파이의 실제 안테나가 어떻게 생겼는지 모릅니다. 모형 넷으로 해 보고 결론이 같은 것만 믿습니다. 어느 모형이 실제와 닮았는지는 `compare_real.py` 로 실물 곡선과 견줘야 압니다.
