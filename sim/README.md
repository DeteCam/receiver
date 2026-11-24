# sim — 시뮬레이션 코드

v6 설계(태그 `v6-nano2`)의 숫자를 독립 도구로 다시 확인한 코드와, 손바닥 크기 회전 수신기를 계산한 코드입니다. 스위치 고리 계산은 이 폴더에 이어서 추가합니다.
**모든 결과는 시뮬레이션이며 실측이 아닙니다.** 새 계산은 [docs/AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md) 5장 규칙을 따릅니다. 특히 수신기 · 카메라 자리와 조건을 결과를 보기 전에 `sim/plan_YYYYMMDD.md`에 적습니다.

## 폴더

| 폴더 | 내용 | 도구 |
|---|---|---|
| `repo_sim/` | 저장소 벤치마크를 그대로 다시 돌리고, 난수 · 대역폭 · 지도 오차 · 안테나 응답을 하나씩 바꿔 본 시험 | 저장소 코드 + NumPy |
| `em/` | 안테나 전자기 해석. 84 mm 판(v6)과, 회전 수신기의 안테나 2개(기준 + 회전) | NEC2 (`nec2c`, 모멘트법) |
| `rt/` | 교실(40명) · 숙소 3차원 장면의 광선 추적과 위치 · 방향 추정 | Sionna RT 2.2 (Mitsuba 3), scikit-learn |

## 준비

```bash
pip install sionna-rt numpy scipy torch scikit-learn   # Sionna RT 는 LLVM(CPU) 로도 돕니다
sudo apt install nec2c                                  # em/ 을 다시 돌릴 때만
git clone --branch v6-nano2 https://github.com/DeteCam/receiver receiver-v6   # v6 코드 (1번 · 3번에만 필요)
export DETECAM_V6=$PWD/receiver-v6                      # 기본값은 이 저장소 옆의 receiver-v6
```

## 1. v6 보고서 숫자와 검증 숫자가 다른 이유 (docs `기술검증_보고서.md` 1부)

```bash
cd rt
python3 rt_gen_ladder.py classroom 5g 0 shellroi 1 0   # 벽만, 1차 반사, 카메라는 저장소 격자점
python3 rt_gen_ladder.py classroom 5g 0 shell 1 0      # 벽만, 1차 반사, 카메라는 좌석
python3 rt_gen_ladder.py classroom 5g 0 shell 4 1      # + 4차 반사 · 회절
python3 rt_gen_ladder.py classroom 5g 0 furn 4 1       # + 가구
python3 rt_gen_ladder.py classroom 5g 0 full 4 1       # + 사람
python3 eval_ladder.py classroom 5g 160e6              # 저장소 채널 폭으로 평가 (난수 0~2 가 모두 있어야 합니다)
python3 eval_ladder.py classroom 5g                    # 20 MHz
cd ../repo_sim
SEED=3 OUTNAME=seed_3.json python3 bench_seed.py       # 저장소 벤치마크를 난수만 바꿔서
BW24=20e6 BW5=20e6 JIT=0.18 MAPERR=1 EM=1 OUTNAME=x.json python3 bench_stress.py
```

## 2. 손바닥 크기 회전 수신기 (docs `기술검증_보고서.md` 2부)

```bash
cd em
python3 puck.py            # 접지판 위 안테나(모노폴) 2개: 편파가 뒤틀리는 것을 확인
python3 puck2.py           # 접지판 없는 세로 다이폴 2개
python3 puck3.py           # 기준 안테나 높이, 금속 몸체 거리를 바꿔 가며 (puck3_results.npy)

cd ../rt
python3 rt_gen_extra.py classroom 2g4 0                # 교실 네 모서리(높이 2.35 m)의 경로 데이터. 난수 0~2
python3 rt_gen_xm.py 2g4 0                             # 숙소 네 모서리. 난수 0~3
python3 rt_gen_xl.py classroom 2g4 0                   # 교실 앞쪽 낮은 자리(1.5 m, 교탁)
python3 rt_gen_xc.py 2g4 0                             # 숙소에서 벽에서 떨어진 자리
SPOTS='[[0.3,0.3,2.2],[0.8,0.3,2.2],[0.3,0.8,2.2],[0.8,0.8,1.7]]' OUT=xd python3 rt_gen_xc.py 2g4 0   # 반 미터씩 옮긴 자리

python3 pe.py classroom 2g4 'data/x_classroom_2g4_?.npy' c24 ringH45,ringH90 bart,nomp,map1,map2      # 추정 방법 비교
python3 pe2.py classroom 2g4 'data/x_classroom_2g4_?.npy' base                                        # 권장 추정기, 이상적 안테나
EM=box,65,100 python3 pe2.py classroom 2g4 'data/x_classroom_2g4_?.npy' em100                         # 전자기 해석으로 구한 안테나 응답
ARR=ringH90 NPOS=24 EM=box,65,100 python3 pe2.py motel 2g4 'data/xm_2g4_?.npy' r90                    # 팔을 9 cm 로 펼친 경우
HEADERR=6 CALIB=8 python3 pe2.py classroom 2g4 'data/x_classroom_2g4_?.npy' cal                       # 놓인 각도가 틀렸을 때 + 현장 보정
python3 ai_rank.py gbm                                 # 학습으로 후보 고르기 · 신뢰도 (한 방에서 배우고 다른 방에서 시험)
python3 mk_tables.py ; python3 sens_rows.py ; python3 report_numbers.py    # 결과 표 (pe2.py 가 만든 res_pe2_*.npy 를 읽습니다)
python3 motel_break.py m_em100                         # 숙소: 자리별, 가까운 카메라 제외
```

`pe2.py` 의 환경 변수(파일 맨 위 설명): `ARR` 팔 길이, `NPOS` 한 바퀴 자리 수, `BW` 채널 폭, `EM` 안테나 응답, `DIMERR` · `POSERR` · `HEADERR` 지도와 놓인 자세의 오차,
`ANGJIT` · `CALDEG` · `RNOISE` 장치 오차, `TVAR` 재는 동안 흔들리는 반사(`NSWEEP` 바퀴 평균), `CALIB` · `REFLCAL` 현장 보정, `MAPFURN` 큰 가구 면을 지도에 추가.
벽 재질을 바꾼 숙소는 `WALL=concrete` 또는 `WALL=metal` 을 붙여 `rt_gen_xc.py` 로 만듭니다.

## 3. v6 설계 검증 (84 mm 판, 틀, 막대 비교)

```bash
cd em && python3 em_run2.py                                  # 84 mm 판 해석 (em_results.npy)
cd ../rt
python3 rt_gen.py classroom 2g4 0                            # 장면 · 카메라 40개 · 경로 데이터 생성 (data/)
python3 rt_eval.py classroom 2g4                             # 20 MHz 기준 평가
python3 rt_eval.py classroom 5g "" 160e6                     # 저장소가 가정한 160 MHz 로 평가
PEOPLE=0 VSUF=_empty python3 rt_gen.py classroom 2g4 0 && python3 rt_eval.py classroom 2g4 _empty
python3 rt_gen_extra.py classroom 2g4 0 && python3 rt_eval_x.py classroom 2g4       # 84 mm 판을 네 모서리에
python3 rt_gen_frame4.py classroom 2g4 0 && python3 rt_eval_frame4.py classroom 2g4  # 20 cm 틀을 네 모서리에
python3 rt_gen_mbar.py 2g4 0 && python3 rt_eval_mbar.py motel 2g4                    # 숙소, 47 cm 막대 4개
python3 sanity_los.py classroom 2g4 ; python3 check_approx.py classroom 2g4 ; python3 check_sampling.py
```

## 결과 기록

- `rt/results/ev_ladder_*.log`: 1번. `rt/results/ev_pe_*.log`, `ev_pe2_*.log`, `ai_rank_*.log`, `tables.md`: 2번. 나머지 `ev_*.log`, `checks.txt`: 3번.
- `repo_sim/results/`: 저장소 벤치마크 JSON.
- 경로 데이터(`rt/data/*.npy`, 수백 MB), 결과 배열(`res_*.npy`), 안테나 해석 결과(`em/*.npy`)는 용량 때문에 저장소에 넣지 않습니다. 위 명령으로 같은 난수에서 다시 만들어집니다. 메모리는 6 GB 이상, 광선 추적은 한 번에 하나씩 돌리세요.

## 기록 읽는 법

- `ev_pe2_*.log` 의 `one node, one spot` 은 수신기 한 대가 한 자리에서 한 번 잰 방향 오차(네 자리 평균), `fused x2 adj` 는 0 · 1번 자리(교실은 앞쪽 두 모서리)를 합친 것,
  `always k spot(s)` 는 한 대를 k 자리로 옮겨 가며 합쳤을 때 첫 자리에서 본 방향 오차입니다.
- 위치 오차 줄의 숫자는 중앙값 / 1 m 이내 / 0.5 m 이내, 교실은 뒤에 정확한 자리 / 주변 3 × 3 이 붙습니다.
- `ev_*.log`(3번)의 `nano2_ideal` 은 저장소 가정(이상적인 점 안테나 12개), `nano2_em` 은 NEC2 로 구한 실제 판 응답, `repo MA-PINN+AI` 는 저장소 추정기와 학습된 가중치 그대로, `capon fusion` 은 단순 빔 합성입니다.
