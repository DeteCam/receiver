# 시뮬레이션 기록 (`sim/`)

**이 폴더의 숫자는 모두 시뮬레이션(계산값)이며 실측이 아니다.** 설계를 고르는 근거로만 쓰고 "성능"이나 "목표 달성"으로 쓰지 않는다. 규칙은 [docs/AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md) 5장이다. 계획서를 결과 전에 고정하고, 시드 3개 이상, 방마다 카메라 40곳 이상, 평균 · 중앙값 · 90% · 95% 지점 · 가장 나쁜 방을 함께 적는다. 결과를 본 뒤 더한 것은 "사후 추가"로 표시한다.

마지막 수정: 2026-11-27

가장 최근 결과는 [안테나-케이블-배선-결과.md](안테나-케이블-배선-결과.md)(케이블 배선, 11-27)와 [위험-점검-결과.md](위험-점검-결과.md)(남은 위험 점검, 11-27)다. 방 계산의 바탕은 [실제-방-시뮬레이션-결과.md](실제-방-시뮬레이션-결과.md)에 있다(실제와 비슷한 새 방 16개). 지금 고른 설계는 [docs/현재_방향.md](https://github.com/DeteCam/docs/blob/main/현재_방향.md)에 있다.

## 1. 폴더

| 경로 | 내용 |
|---|---|
| `plan_*.md` | 계획서. 결과를 보기 전에 장면 · 자리 · 조건 · 판정 기준을 적은 것 |
| [`실제-방-시뮬레이션-결과.md`](실제-방-시뮬레이션-결과.md) | 새 방 16개 본 계산의 전체 표, 사다리, 미리 적은 예상과 결과, 믿어도 되는 것 |
| [`안테나-케이블-배선-결과.md`](안테나-케이블-배선-결과.md) | 필름 안테나 케이블 16가닥을 어떻게 보낼지(등뼈 배선), 기준 안테나 자리, 상자 거리 · 크기. 2.4 · 5 GHz, 묶음 A · B · C |
| [`위험-점검-결과.md`](위험-점검-결과.md) | 스위치–패킷 짝짓기, 케이블 길이, 스위치의 꺼진 포트, 끄덕이는 축 자리 |
| `mysim/` | 광선 추적(Sionna RT), 안테나 계산(NEC2), 신호 합성, 추정기, 분석 스크립트 |
| `v1_ring/` | 11-25 첫 고리 계산(V1) 코드와 결과. `mysim/`이 방 만들기 · CSI 합성 코드를 불러 쓴다 |
| `em/ring_nec2.py` | V1 계산이 쓰는 NEC2 패턴 생성기(nec2c 호출, `em/data/`에 표를 만든다). 11-26부터의 계산은 같은 모형을 PyNEC로 옮긴 `mysim/ring_pynec.py`를 쓴다 |
| `results/*.txt` | 분석 스크립트가 낸 요약 표 |
| `results/시뮬레이션-장면.md`, `시뮬레이션-장면-좌표.md`, `scene_topviews.png` | 기존 방 17개의 크기 · 재질 · 물건 · 사람 · 수신 자리, 시드 0의 카메라 좌표, 위에서 본 그림 |
| `build_nec.sh`, `setup_paths.sh` | 안테나 표 만들기, 스크립트 안 경로 바꾸기 |

저장소에 넣지 않은 것(다시 만들 수 있음): 광선 추적 데이터(`data_wall/` 약 460 MB, `data_real/`), 결과 원본(`results/*.pkl`), 안테나 표(`mysim/*.npy`), 실행 기록(`logs/`).

예전 시뮬레이션 코드(`sim/em`의 v6 · 회전 수신기 모형, `sim/repo_sim`, `sim/rt`: v6 독립 검증과 회전 수신기 계산)는 지금 설계와 맞지 않아 지웠다. 커밋 [`e66aed2`](https://github.com/DeteCam/receiver/tree/e66aed2/sim)에서 볼 수 있다.

## 2. 준비

- 도구: Python 3.13, Sionna RT 2.2.0(Mitsuba 3.9.1, Dr.Jit 1.5.0), PyNEC 2.4.1, NumPy 2.5, SciPy 1.18, scikit-learn 1.9. Sionna RT는 CPU(LLVM)로도 돈다.

```
pip install sionna-rt==2.2.0 numpy scipy scikit-learn PyNEC==2.4.1
```

- 레거시 돌판 비교를 다시 돌리려면 [receiver-legacy](https://github.com/DeteCam/receiver-legacy)를 받아 커밋 `62cd341d08f3ea41af58a0a028144372dee9231d`(2026-11-18)로 맞춘다. `sim/rt` · `firmware/core` · `firmware/pi` · `sim/em`을 쓴다.
- 스크립트에는 자리표시 경로(`/opt/detecam/sim`, `/opt/detecam/receiver-legacy`)가 들어 있다. 이 폴더에서 한 번 바꾼다. 바뀐 파일은 커밋하지 않는다(`git checkout -- mysim v1_ring`으로 되돌린다).

```
cd sim
bash setup_paths.sh /path/to/receiver-legacy
```

- 아래 명령은 모두 `sim/` 폴더에서 돌린다.
- 광선 추적은 반드시 `PYTHONHASHSEED=0`으로 돌린다. 그렇지 않으면 카메라 자리가 파일마다 달라진다. 평가는 `OMP_NUM_THREADS=1`.
- 두 프로세스가 같은 pkl에 쓰지 않게 한다(파일을 나눈다).
- 시험실 평가는 프로세스 하나가 약 2 GB를 쓴다. 메모리 8 GB에서는 광선 추적과 함께 시험실 평가를 하나만 돌린다.

## 3. 다시 돌리는 순서

### 3.1 안테나 표 (NEC2)

```
bash build_nec.sh          # nec_patterns → nec_all → nec_all2 → nec_all3 → nec_all4
```

### 3.2 기존 방 17개 (계획서 `plan_20261126_wall` · `final` · `plate`)

```
# 광선 추적 장면(방 17개, 카메라 40곳, 2.4 · 5 GHz)
PYTHONHASHSEED=0 python3 mysim/gen_wall.py --variants base --seeds 0,1,2                 # 사람 있음(시험실 기본)
PYTHONHASHSEED=0 python3 mysim/gen_wall.py --variants nop --seeds 0,1,2                  # 사람 없음(시험실 밖 기본)
PYTHONHASHSEED=0 python3 mysim/gen_wall.py --variants nop --nodeset free --seeds 0,1,2   # 돌판 자유 자리 비교용

# 최종 검증(plan_20261126_final.md)
OMP_NUM_THREADS=1 python3 mysim/exp_final.py outside 0,1,2
OMP_NUM_THREADS=1 python3 mysim/exp_final.py exam 0,1,2
OMP_NUM_THREADS=1 python3 mysim/exp_final.py anchor-outside 0,1,2 H1,H2,H4,H5
OMP_NUM_THREADS=1 python3 mysim/exp_final.py anchor-exam 0,1,2 H1,H2
OMP_NUM_THREADS=1 python3 mysim/exp_final.py anchor-exam 0,1,2 H4,H5
python3 mysim/analyze_final_check.py outside      # exam, anchor, pairs도 같은 방식

# 레거시 돌판 비교와 1기 여러 자리(plan_20261126_plate.md)
OMP_NUM_THREADS=1 python3 mysim/exp_bearing.py 0,1,2 2g4
for s in 0 1 2; do PLATE_OUT=results/exp_bearing_plate_s$s.pkl OMP_NUM_THREADS=1 python3 mysim/plate_eval.py $s; done
for s in 0 1 2; do PLATE_OUT=results/exp_plate_pos_s$s.pkl OMP_NUM_THREADS=1 python3 mysim/plate_pos.py $s; done
OMP_NUM_THREADS=1 python3 mysim/exp_ringmulti.py 0,1,2
python3 mysim/analyze_bearing.py 0,1,2; python3 mysim/analyze_plate_pos.py 0,1,2

# 면 근접 결합(plan_20261126_wall.md)
OMP_NUM_THREADS=1 python3 mysim/exp_wall.py base 0,1,2 d120_16_fv,d120_16_wcf,d120_16_wmf,d120_16_wcc,d120_16_wmc,d120_16_wpf,d120_16_wmm all
python3 mysim/analyze_final.py nf
```

### 3.3 실제와 비슷한 새 방 16개 (계획서 `plan_20261126_real` · `real_c`)

```
PYTHONHASHSEED=0 python3 mysim/real_meta.py check          # 방 정의 점검
PYTHONHASHSEED=0 python3 mysim/real_meta.py
PYTHONHASHSEED=0 python3 mysim/gen_real.py --scenes A --variants nop,base,op1 --bands 2g4,5g --seeds 0,1,2
python3 mysim/exp_real.py nec
REAL_LL=1 OMP_NUM_THREADS=1 python3 mysim/exp_real.py A design 2g4 0,1,2 P0F A_design_ll
REAL_LL=1 OMP_NUM_THREADS=1 python3 mysim/exp_real.py A real2 2g4 0,1,2 P0F A_real2_ll
python3 mysim/exp_dev_ll.py 0,1,2 2g4 P0F                     # 기존 방 지도(문턱 · 모델용)
python3 mysim/analyze_real.py A_design_ll P0F                 # 한 번 잰 방향, 4자리 위치, 시험실
python3 mysim/ll_study.py rel 2g4 P0F real_A_design_ll.pkl    # 신뢰도(규칙 대 배운 모델)
python3 mysim/ll_policy.py 2g4 P0F real_A_design_ll.pkl       # 옮기기, 가장 믿을 만한 자리
python3 mysim/ll_fbear.py 2g4 P0F real_A_design_ll.pkl        # 융합 방향
python3 mysim/ll_adapt.py 2g4 P0F real_A_design_ll.pkl        # 더 재기
python3 mysim/exp_real.py A design 2g4 0,1,2 P1,P2,P3,P4,P5,P6,P7,SW1,SW3 A_design_c   # 구조 · 처리 후보
REAL_ONE=1 python3 mysim/exp_real.py exam_s,lecture_c,seminar_g design 2g4 0,1,2 P0 X_design_one   # 시험실 한 자리
```

묶음 B는 `A` 자리에 방 이름(`goshiwon_a,studio_b,restroom_e,sauna_locker` / `lecture_c,seminar_g`)을 쓰고, 묶음 C는 `plan_20261126_real_c.md` 6절의 명령을 쓴다. 스크립트마다 정확한 인자(시드, 설정 이름)는 파일 맨 위 설명과 `CFG`에 있다.

### 3.4 상자 거리 · 안테나 케이블 배선 (계획서 `plan_20261126_molex`)

```
PYTHONHASHSEED=0 python3 mysim/molex_heads.py mysim/nec_molex.npy      # 상자 거리 G30 · G15, 곧장 내린 케이블 G*C
PYTHONHASHSEED=0 python3 mysim/molex_layouts.py mysim/nec_molex2.npy   # 바퀴살 + 기둥(L1 · L2 · L5), 뒤집은 머리(L3 · L6 · I30), 진단 R45
python3 mysim/exp_molex.py nec                                         # nec_real.npy에 추가
python3 mysim/exp_molex.py scales                                      # 머리마다 기준 안테나 수평 이득 변화(잡음 's' 방식에 씀)
REAL_LL=1 python3 mysim/exp_molex.py A design 2g4 0,1,2 P0F,Q30,Q15,Q30Cs,Q15Cs,Q60Cs,QL1s molex_a
python3 mysim/molex_compare.py real_molex_a.pkl:P0F real_molex_a.pkl:Q30 real_molex_a.pkl:QL1s   # 짝 비교(M1, 융합 방향, 위치)
python3 mysim/analyze_real.py molex_a QL1s                             # 시험실 3곳 안
```

실제로 돌린 묶음과 태그는 [`plan_20261126_molex.md`](plan_20261126_molex.md)에 있다. 묶음 C 확인은 방 이름(`hotel_c,guesthouse_d,restroom_f,pool_locker,fitting_b`)을 쓰고 기준은 `real_C_design_ll.pkl`의 `P0F`다.

### 3.5 케이블 배선 다시 찾기 · 남은 위험 (계획서 `plan_20261126_rod` · `plan_20261126_risk3`)

```
# 안테나 표(NEC2). 이름은 mysim/rod_heads.py의 LAYOUTS: k2i = 지금 배선, k1 · k2 · k2n · f1 · r1 … = 견준 배선
PYTHONHASHSEED=0 python3 mysim/rod_heads.py mysim/nec_rod.npy d120_16k2i,d120_16k2i_ty+20,d120_16k2i_ty-20 2g4
PYTHONHASHSEED=0 python3 mysim/rod_heads.py mysim/nec_rod.npy d120_16k2i,d120_16k2i_ty+20,d120_16k2i_ty-20 5g
python3 mysim/exp_rod.py nec                 # nec_real.npy에 추가
python3 mysim/exp_rod.py scales 2g4          # 머리마다 기준 안테나 수평 이득 변화
# 방 계산(설정 이름은 Q + 배선 이름 + s). 묶음 A는 'A', 묶음 B · C는 방 이름을 쉼표로
REAL_LL=1 PYTHONHASHSEED=0 python3 mysim/exp_rod.py A design 2g4 0,1,2 QK2is rod_a14
REAL_LL=1 PYTHONHASHSEED=0 python3 mysim/exp_rod.py A design 5g 0,1,2 QK2is rod_a15
python3 mysim/molex_compare.py real_molex_a.pkl:P0F real_rod_a14.pkl:QK2is     # 짝 비교(한 자리 방향, 융합 방향, 위치)
python3 mysim/pos_pairs.py real_molex_a.pkl:P0F real_rod_a14.pkl:QK2is         # 방마다 잃은 · 얻은 카메라와 부호 검정
python3 mysim/full_stats.py real_rod_a14.pkl:QK2is                             # 평균 · 중앙값 · 90% · 95% · 가장 나쁜 방
python3 mysim/analyze_real.py rod_a14 QK2is                                    # 시험실 3곳 안
# 묶음 C의 5 GHz 광선 추적(11-27에 새로 만듦)
PYTHONHASHSEED=0 python3 mysim/gen_real.py --variants nop --scenes hotel_c,guesthouse_d,restroom_f,pool_locker,fitting_b --bands 5g --seeds 0,1,2
# 남은 위험
python3 mysim/risk_switch_sync.py            # 스위치–패킷 짝짓기(몬테카를로)
python3 mysim/risk_switch_sync.py adaptive   # 패킷 수에 맞춰 넘어가는 경우
python3 mysim/cable_budget.py k2i 88.9       # 케이블 길이 셈(k1 · k2i · k2n · k2)
```

실제로 돌린 배선 · 묶음 · 태그는 두 계획서와 결과 문서의 "다시 돌리는 법"에 있다.

## 4. 장면

- **기존 방 17개** (`v1_ring/gen_scenes.py`, `mysim/fresh_scenes.py`, `mysim/gen_wall.py`)
  - 시험실 밖 12개: lodging_a · b · c, restroom_a · b · c, locker_a · b, bathhouse_a · b, office_a, worst_metal(금속 방)
  - 시험실 5개: classroom_a · b · c, lecture_a · b
  - 내용 변형: `base`(물건 + 사람, 시험실은 학생 착석), `nop`(사람 없음, 시험실 밖의 기본 가정)
  - 수신 자리(벽에서 약 0.1 m): 모서리 c0~c3, 벽 가운데 m0~m3, 높이 2.0 m · 1.5 m. 방 안 스탠드 자리 f0~f4는 돌판 비교에만 쓴다(큰 스탠드는 요구사항 R8로 금지).
- **새 방 16개** (`mysim/real_scenes.py`): 묶음 A 5개(개선용) + 묶음 B 6개(따로 떼어 둠) + 묶음 C 5개(두 번째 확인). 네모가 아닌 방, 지도에 없는 금속, 손으로 정한 수신 자리(벽 고리 · 가구 위 · 탁자 높이 · 욕실 · 칸 안), 숨긴 카메라(상자 35%, 얇은 유리 10%). 기존 17개 방은 문턱 · 모델을 정하는 데만 썼다.
- 방마다 카메라 40곳(실제로 숨기는 자리), 2.4 GHz(2.437 GHz) · 5 GHz(5.22 GHz), 반사 4회 · 회절 · 투과(새 방은 난반사 포함, 광선 10만 개).
- 시드마다 바뀌는 것: 카메라 자리, 사람 자리, 시험실 책상 위치의 작은 흔들림. 방 크기 · 벽 · 고정 가구 · 거울 · 칠판은 같다.
- 안테나: NEC2로 푼 지름 12 cm 16소자 고리 응답(막대 다이폴 대리, 필름 안테나보다 좋은 쪽일 수 있음). 끄덕인 자세는 `ty±20` 표.
- 시험실 평가는 좌석 정보를 입력하지 않는다(R10). 책상은 폰 AR 평면 인식 + 사람 인식으로 자동으로 찾는다고 두고 공통 위치 오차 0.10 m, 약한 사전 가중으로 넣었다(`mysim/exp_wall.py`의 `seat10m`). 좌석마다 기기를 두는 구성(R11)은 계산하지 않는다.

## 5. 계획서와 결과 (오래된 것부터, 모두 계산값)

| 계획서 | 물은 것 | 핵심 결과 |
|---|---|---|
| `plan_20261125.md` | 스위치 고리 V1(8소자 등), 기준선 비교 | 16소자 지름 12 cm 쪽으로 감 |
| `plan_20261126_wall.md` | 벽 부착 머리 모양, 자리 수, 높이, 끄덕이기, 면 근접 결합, 책상 자동 인식 | 벽 부착 손해 없음. 반원 · 평면 머리 이득 없음. 사람 있는 조건에서 벽 4곳 × 두 높이 81~91%. 근접 결합은 유전체 벽에서 무시할 만하고 금속면에서 남음 |
| `plan_20261126_final.md` | 최종 부품(2곳, 사람 없음, 실제 앵커 손실) | 시험실 2곳 + 끄덕이기: 의심 책상 3곳 안 92%. 시험실 밖 맞은편 2곳 + 끄덕이기(H4): 0.5 m 안 65%, 1 m 안 89%. 앵커 H1 부호 복원 86% → H4 97% |
| `plan_20261126_plate.md` | 고리 대 레거시 돌판(기준 곡선 보정), 돌판 자유 배치 | 한 자리 방위각 고리 8.4° 대 돌판 18.3°(2.4 GHz, 모든 종류 · 시드에서 고리). 위치 0.5 m 안: 고리 2곳 59%, 4곳 80%, 8곳 91% / 돌판 방 안 2곳 52%, 3곳 64%, 5곳 76%, 9곳 83% |
| `plan_20261126_real.md`, `plan_20261126_real_c.md` | 실제와 비슷한 새 방 16개에서 지금 구조가 기준선을 넘는가, 못 넘으면 무엇으로 줄이는가 | 한 번 잰 방향 13.8~16.3°(기준선 8.92° 미달, 원인은 가림). 구조 후보 9개 채택 없음. 체인 B 입력 6 dB 감쇠기 + 새 판정 규칙. 4자리 "융합 방향" 4.4~5.4°(묶음 C에서 사전 등록 뒤 확인). 위치는 숙박만 5~7곳에서 81~88%, 탈의실 68~79%, 금속 칸막이 화장실 45~49%. 시험실 앞벽 2자리 88~93%, 한 자리 38~61% |
| `plan_20261126_molex.md` (3.2~3.4절 사후 추가, **11-27 결과로 대체됨**) | Molex 필름 케이블(100 mm, 옆으로 나옴) 때문에 상자가 가까워지고 케이블 16가닥이 소자 옆을 지날 때 얼마나 나빠지나 | 상자 윗면은 소자 가운데에서 5.9 cm면 +4%(영향 작음), 4.4 cm면 +11%. 케이블을 소자 안쪽에서 곧장 내리면 한 자리 방향 +40~66%, 위치 −18~34%p(큰 결함). 가로 바퀴살 + 가운데 기둥 + 기준 안테나 45 mm 위(L1)가 가장 덜 나쁨: +11~12%(묶음 A · C), 융합 방향은 그대로(5.1° / 15.4°, 5.3° / 15.8°), 화장실 위치 −10~16%p. 결과: [`안테나-케이블-배선-결과.md`](안테나-케이블-배선-결과.md) |
| `plan_20261126_rod.md` (6절부터 사후 추가) | 케이블 손해를 없애는 배선이 있는가(막대 안테나, 상자 거리, 기준 안테나 자리, 고리 면 안 배선), 5 GHz에서도 되는가 | 기준 안테나를 고리 면 가운데에 두고 케이블을 고리 면 안에서만 뒤쪽으로 비스듬히 모아 고리 안쪽 뒤에서 내리는 등뼈 배선(K2i)이면 케이블 없는 모델 대비 한 자리 방향 2.4 GHz +0.5~1.8%, 5 GHz −3.7~+1.3%(묶음 A · B · C). 손해의 원인은 기준 안테나를 고리 면 밖으로 올린 것(케이블 없이도 화장실 −10%p)과 세로 케이블. 아래에서 급전하는 막대 안테나는 케이블 길이에 따라 +3~94%. 갈비를 등뼈와 직각으로 보내면 5 GHz +13.6%. 실제 크기 상자를 넣으면 2.4 GHz +6.6%(중간). 결과: [`안테나-케이블-배선-결과.md`](안테나-케이블-배선-결과.md) |
| `plan_20261126_risk3.md` | 모델이 넣지 않은 남은 위험(스위치–패킷 짝짓기, 케이블 길이, 스위치의 꺼진 포트, 끄덕이는 축) | 짝짓기는 17상태 + 5패킷마다 넘어가기 + 가드 10 ms에서 잘못 짝지음 4.7% 이하. 꺼진 포트가 반사형이어도 실물 보정표면 +0.8~1.9%, 모형 표만 믿으면 +6.5~6.7%. 끄덕이는 축 때문에 고리 중심이 4 cm 움직이므로 추정기에 넣음. 결과: [`위험-점검-결과.md`](위험-점검-결과.md) |

새 방 16개 요약(2.4 GHz 설계 조건):

| 결과 | 묶음 A | 묶음 B | 묶음 C |
|---|---|---|---|
| 한 번 잰 방향 평균 / 중앙 / 90% / 95% | 13.8° / 4.2° / 39.9° / 72.5° | 16.3° / 4.9° / 48.7° / 81.8° | 14.0° / 4.4° / 43.9° / 65.8° |
| 가장 나쁜 방(평균) | pension_a 16.4° | restroom_e 25.5° | restroom_f 17.2° |
| 융합 방향(4자리) 평균 / 95% | 4.4° / 12.5° (사후 추가) | 5.3° / 16.8° (사후 추가) | 5.4° / 15.8° (사전 등록, 통과) |
| 4자리 0.5 m 안(숙박 / 화장실 / 탈의실) | 69.6 / 65.8 / 66.7% | 79.6 / 49.2 / 61.7% | 62.5 / 85.8 / 68.3% |
| "더 재기" 0.5 m 안, 평균 자리 수 | 87.5 / 79.2 / 75.0%, 5.6곳 | 83.3 / 49.2 / 71.7%, 5.1곳 | 81.2 / 89.2 / 76.7%, 5.6곳 |
| 시험실 의심 책상 3곳 안(앞벽 2자리 / 한 자리) | 93.3% / 60.8% | 88.3% / 37.5~58.3% | — |

- 사전 등록한 현실 조건은 체인 판정 규칙이 거꾸로 동작해 방향 23.6°로 무너졌다. 체인 B 입력 6 dB 감쇠기 + "센 쪽 = 기준 체인" 규칙(`real2`, 사후 추가)으로 13.1°가 됐다.
- 미리 적은 예상 10개 가운데 4개는 맞았고, 3개는 틀렸고, 3개는 일부만 맞았다. 모두 결과 문서 6절에 그대로 적었다.
- 1기로 차례로 재는 것은 계산에서 동시 측정과 같다(장면이 그대로이고 카메라가 계속 보낸다고 둠). 시험실처럼 카메라가 송출을 멈출 수 있는 곳은 한 자리만 남는 경우를 따로 계산했다.
- 기존 방 벽 4곳(m0~m3) 차례로 + 끄덕이기, 2.4 GHz, 사람 없음, H4: 0.5 m 안 80%, 1 m 안 95%, 평균 0.36 m, 중앙 0.27 m, 90% 0.71 m, 95% 0.97 m. 종류별 0.5 m 안은 목욕탕 78%, 탈의실 85%, 숙소 85%, 사무실 80%, 화장실 72%, 금속 방 81%(`plan_20261126_plate.md` 7절, 사후 추가).

## 6. 스크립트 (`mysim/`)

| 파일 | 하는 일 |
|---|---|
| `ring_pynec.py`, `wall_heads.py`, `wall_nf.py`, `shapes.py`, `cand_heads.py` | NEC2 안테나 표(고리 · 끄덕인 자세 · 면 근접 결합 · 모양 비교 · 구조 후보) |
| `gen_wall.py`, `fresh_scenes.py`, `gen_ext.py`, `scene_meta.py` | 기존 방 광선 추적. `--variants base,nop`, `--nodeset wall/corner/wallhi/free` |
| `real_scenes.py`, `real_meta.py`, `gen_real.py` | 새 방 16개 정의 · 점검 · 광선 추적 |
| `dump_scenes.py` | 기존 방 17개를 다시 만들어 장면 문서 2개와 위에서 본 그림을 만든다(광선 추적 없음) |
| `my_eval.py`, `pattern3d.py`, `risk_synth.py`, `est_variants.py` | 신호 합성 + 지도 반영 우도 추정기(직접파 + 벽 · 천장 · 바닥 반사), 추정기 변형 |
| `exp_wall.py`, `analyze_wall.py`, `analyze_alt_seat.py` | 벽 부착 평가, 책상 자동 인식 후보(`seat10` · `seat25` · `seat10m`), 면 근접 결합 설정(`d120_16_w*`) |
| `exp_final.py`, `analyze_final_check.py`, `analyze_final.py` | 최종 검증과 앵커 손실표(H1 · H2 · H4 · H5) |
| `exp_bearing.py`, `analyze_bearing.py`, `exp_ringmulti.py` | 고리 1기 방위각(끄덕이기 3자세), 1기 여러 자리 위치 |
| `plate_eval.py`, `plate_pos.py`, `analyze_plate_pos.py` | 레거시 돌판을 같은 장면에 붙인 비교 |
| `exp_real.py`, `exp_dev_ll.py`, `analyze_real.py`, `cmp_real.py` | 새 방 평가(`REAL_LL=1` 지도 저장, `REAL_ONE=1` 시험실 한 자리), 기존 방 지도, 표 |
| `ll_study.py`, `ll_policy.py`, `ll_adapt.py`, `ll_fuse2.py`, `ll_fbear.py`, `ll_b4_nodev.py` | 측정 뒤 절차: 신뢰도(퍼짐 규칙 대 경사 부스팅), 옮기기, 더 재기, 융합 변형, 융합 방향 |
| `exp_chain_h4.py`, `exp_real_abl.py`, `exp_real2_ladder.py` | 체인 가르기, 조건을 하나씩 바꾼 사다리 |
| `molex_heads.py`, `molex_layouts.py`, `exp_molex.py`, `molex_compare.py` | 상자 거리 · 안테나 케이블 배선(곧장 내림, 바퀴살 + 기둥, 기준 안테나 위치, 뒤집은 머리)과 짝 비교 |
| `rod_heads.py`, `exp_rod.py`, `pos_pairs.py`, `full_stats.py` | 케이블 배선 다시 찾기(막대 안테나, 고리 면 안 등뼈 배선, 묶음 자리, 실제 크기 상자, 꺼진 포트 부하)와 방마다 짝 비교 · 전체 통계 |
| `risk_switch_sync.py`, `cable_budget.py` | 스위치–패킷 짝짓기 몬테카를로, 케이블 길이 셈 |
| 그 밖의 `exp_*`, `ai_*`, `analyze_*` | 11-24~25의 이전 실험(기준선, 크기, 재순위 학습 등). 지금 결정에는 쓰지 않지만 다시 볼 수 있게 남김 |

추정기는 광선 추적 결과를 직접 읽지 않는다. 합성된 CSI와, 사용자가 실제로 알 수 있는 방 정보(벽 · 천장 · 바닥 위치, 큰 가구)만 받는다.

## 7. 알아 둘 함정

- 자세 오차 난수는 평가하는 노드 수에 따라 달라진다. 끄덕이기(48노드) 결과는 끄덕이기 없는 실행과 숫자가 다를 수 있다. 같은 노드 집합끼리만 견준다.
- 돌판 NEC2 표는 2.4 GHz뿐이다. 5 GHz 돌판 비교는 없다. 돌판은 새 방 16개에서 다시 돌리지 않았다(계획서: 구조를 바꿀 때만).
- 고리 SNR은 안테나 출력 20 dB에서 부품 손실(스위치 · 윌킨슨 · 감쇠기)을 뺀 값이다. 돌판 비교에서는 돌판을 패킷 SNR 30 dB로 유리하게 뒀다.
- Sionna RT는 두께 6 mm 유리판에서 직접파를 아예 내지 않는다. 카메라에서 5 mm 안쪽에 면이 있어도 같다. 유리 뒤 카메라 결과는 실제보다 불리하다.
- 앞 계획서의 대책이 뒤에 바꾼 부품 배치와 맞는지 다시 확인한다. 사전 등록 조건의 체인 판정 규칙이 앵커 배치 H4와 맞지 않아 거꾸로 동작했다.
- 시험실 숫자는 책상 자동 인식 오차 0.10 m를 가정한 값이다. 실물 폰 AR 인식이 이보다 나쁘면 그 오차로 다시 계산한다(`seat25`가 0.25 m 판).
- 면 근접 결합에서 아직 안 한 것: ±20° 자세, 5.8 GHz, 선반 · 교탁 윗면 위, 받침판 모터 · 보조배터리, 최종 조건(`nop`, H4).
- 아직 없는 값: 기존 방 벽 4곳의 5 GHz 값. 새 방 묶음 C의 5 GHz 광선 추적은 11-27에 만들었고(시험실 밖 5방), 케이블 배선 확인에만 썼다.
- 머리 표 `d120_16` 등에는 안테나 케이블이 없다. 케이블 배선에 따라 한 자리 방향이 수십 % 바뀐다. 지금 배선이 들어간 표는 `d120_16k2i`(`rod_heads.py`)다. 머리 구조를 비교할 때는 케이블 배선을 같게 둔다.
- NEC2에서 굵은 선(케이블 묶음, 반지름 1.5~2 mm)을 반지름의 2.5배보다 짧게 나누면 계산이 틀어진다. 5 GHz(조각 3 mm)에서 이 때문에 한 배선의 손해가 +13.6%가 아니라 +41.8%로 나왔었다. `rod_heads.py`의 `thin_fix`가 이를 막는다.
- 5 GHz에서 실제 크기 상자(135 × 158 × 56 mm)는 조각이 5천 개를 넘어 한 자세에 20분 넘게 걸린다(2코어). 이번에는 계산하지 않았다.
- 합격선(시험실 밖 종류마다 0.5 m 안 80%, 시험실 의심 책상 3곳 안 80%, 방향 평균 8.92° 이하 · 95% 지점 27.53° 미만)은 결과를 본 뒤 바꾸지 않는다.

## 8. 저장소에 없는 계산

- **빠른 계산(11-26 16시, 생각 확인용).** 상자 방 영상법으로 만든 간단한 모형에서 구조 후보 20개를 견줬다. Sionna RT가 아니라서 근거로 쓰지 않는다(AGENTS.md 5.3절). 이 계산에서 "유망"이던 큰 지그재그 고리 + 세로축 회전은 위 본 계산에서 지금 구조보다 13~21% 나빠 채택하지 않았다. 코드는 저장소에 넣지 않았다.
