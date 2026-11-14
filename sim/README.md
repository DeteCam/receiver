# sim: 시뮬레이션

판을 돌려 방향을 찾는 방법이 얼마나 맞는지, 방과 곳에 따라 어떻게 달라지는지 장치 없이 돌려 봅니다. 계산 코드(방향·위치 추정, 판별)는 `../firmware/core` 의 것을 그대로 씁니다 (`paths.py` 가 찾습니다). 숫자의 뜻과 결론은 [설계 검토](../docs/설계%20검토.md)와 [네 단계 검토](../docs/네%20단계%20검토.md)에 있습니다. 파일 하나하나는 [폴더와 파일](../docs/폴더와%20파일.md).

```
pip install numpy matplotlib
python3 location/places.py 100          # 곳별 결과를 100번씩 (몇 분)
```

## 폴더

| 폴더 | 묻는 것 | 스크립트 → 결과 |
|---|---|---|
| `direction/` | 한 자리에서 방향을 얼마나 맞히나 | `validate.py` (회절 모형 ↔ 손 측정), `montecarlo.py` (재는 방법 6가지 → `result.json`), `room.py`·`room_extra.py` (벽 반사와 사람 몸), `consistency_eval.py`, 그림 `figures.py` |
| `location/` | 여러 자리를 합쳐 위치를 얼마나 맞히나 | `locate3d.py`, `locate3d_var*.py`, `twopos.py`, `places.py` (변기 칸, 욕실, 모텔 방, 강의실), 그림 `fig_locate.py`·`fig_places.py` |
| `identify/` | 판별: 몇 초씩 몇 번, 얼마나 잘못 잡나 | `identify_sim.py` → `identify_result.json` |
| `em/` | 전파 프로그램으로 푼 수신기 (아래) | |

결과 json 은 그것을 만든 스크립트 옆에 있고, 그림은 `../docs/images/` 에 씁니다.

## em/: 전파 계산 (NEC2, gprMax)

수신기 전체(파이 기판, 모터, 동전 균형추, 드라이버, 요크, 배터리, 판)를 금속 선 그물로 만들어 안테나 프로그램 NEC2 로 풉니다. 방 전체는 시간 영역 격자법 프로그램 gprMax 로 따로 풀었습니다. 위의 실험과는 따로 만든 계산이라 서로 견줄 수 있습니다.

```
pip install PyNEC gprMax           # NEC2 (모멘트법), gprMax (시간 영역 격자법)
python3 em/tools/test_compare.py   # 넣어 둔 전파 표로 계산이 도는지 (1분)
python3 em/tools/compare_real.py 내려받은파일.json --az 0 --el 0      # 실물로 잰 곡선을 모형들과 견준다
OMP_NUM_THREADS=1 python3 em/fdtd/room2d.py      # 방 전체를 gprMax 로 (방 하나에 수십 분)
```

| 폴더 | 내용 |
|---|---|
| `em/model/` | `device.py`·`necgeo.py`: 수신기의 전파 모형 (치수는 `../../cad/model/p2.py` 에서). `run_tables.py`·`tables.py`: 방향마다, 판 각마다의 받는 세기를 표로 (전체를 다시 풀면 몇 시간) |
| `em/studies/` | 표로 재는 과정을 흉내 내는 실험과 결과 json: 자기 부품의 방해 (`own_parts`: 전체를 넣은 기준 곡선으로는 3.4°, 기판만 넣은 것으로는 36°), 안테나 모형이 다르면 (`mismatch`), 판 크기 (`geo_eval`), 세 번·여섯 번 (`plan_eval`: 반사가 심한 방에서 크게 틀리는 비율 30% → 16%), '가장 깊은 골'만 읽으면 (`metrics`), 판 모양 13가지 (`shape_eval`) |
| `em/tools/` | `compare_real.py`: **첫 실물 실험에 쓰는 도구**. 잰 곡선이 어느 모형과 닮았는지, 그 모형으로 방향이 맞는지 |
| `em/figures/` | 그림 17, 31 |
| `em/tables/` | NEC2 로 푼 표를 줄여 넣어 둔 것 |
| `em/fdtd/` | gprMax: 판 그림자 확인(NEC2 와 같은 답인지), 판 틀(플라스틱)의 영향 (`frame_study.py` → 그림 32), **방 전체를 풀어 수신기의 실제 코드로 방향을 찾기** (`room2d.py` → 그림 33) |

## 읽을 때 조심할 것

- 전부 계산 모형입니다. 실제는 더 나쁠 수 있습니다.
- `direction/`, `location/` 의 실험은 골 모양을 두 가지로 가정합니다: W = 회절 모형 그대로(나쁜 쪽), V = 손으로 잰 곡선과 비슷한 모양(좋은 쪽). 편파, 파이 기판과 모터가 만드는 반사는 들어 있지 않습니다. 그것은 `em/` 이 봅니다.
- `em/` 은 파이의 실제 안테나가 어떻게 생겼는지 모릅니다. 모형 넷으로 해 보고 결론이 같은 것만 믿습니다. 어느 모형이 실제와 닮았는지는 `compare_real.py` 로 실물 곡선과 견줘야 압니다.
- `em/fdtd/room2d.py` 는 2차원이라 판이 위아래로 끝없이 긴 셈입니다. 골이 실제보다 깊게 나오므로, 방의 반사가 골의 자리를 얼마나 흔드는지를 보는 계산입니다.
