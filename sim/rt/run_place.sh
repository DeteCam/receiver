#!/bin/bash
# 패널 자리 찾기(3차원)와 사진 16장 이상 모으기. 결과: study/log_place.txt, study/place_g16.json. 이미 한 것은 건너뛴다.
# 곳이 122곳이라 재기를 둘로 나눈다: 손이 닿는 스무 곳(t16)과 grid 102곳(g16b). 카메라가 같으므로(씨앗 20000) 이으면 한 번에 잰 것과 같다.
# 다음 일(검증, 사진 32장 곡선, 대안 수신기, 나쁜 경우, 5G, 거는 각도)은 place_followup.py all.
cd "$(dirname "$0")"; PY="$(cd ../../.. && pwd)/.venv-rt/bin/python"; L=study/log_place.txt; J=${JOBS:-11}
echo "# 시작 $(date)" >> $L
# 1) 손이 닿는 스무 곳(real2)의 순간 1~16 (자세를 섞은 사람 배치, 씨앗 101~116)
for s in $(seq 1 16); do [ -f paths/kr40_t_s$s.npz ] || $PY rt_paths.py kr40_t_s$s --spots real2 --seed $((100+s)) --jitter 0.07 --pose 1 >> $L 2>&1; done
# 2) 스무 곳 재기: 종류(책상·가슴·안경)마다 카메라 400대 = 자리마다 10대씩, 16순간
T=$(for s in $(seq 1 16); do printf "kr40_t_s%d " $s; done)
[ -f study/place_t16.npz ] || $PY place_study.py measure --tag t16 --sets $T --cams 400 --seed0 20000 --jobs $J >> $L 2>&1
# 3) 3차원 후보 102곳(grid)의 순간 1~16 (같은 사람 배치)
for s in $(seq 1 16); do [ -f paths/kr40_g_s$s.npz ] || $PY rt_paths.py kr40_g_s$s --spots grid --seed $((100+s)) --jitter 0.07 --pose 1 >> $L 2>&1; done
# 4) 잇기 -> kr40_tg_s1..s16 (122곳: real2 20 + grid 102). grid 102곳만 잰다
$PY place_study.py merge --n 16 >> $L 2>&1
TG=$(for s in $(seq 1 16); do printf "kr40_tg_s%d " $s; done)
[ -f study/place_g16b.npz ] || $PY place_study.py measure --tag g16b --sets $TG --cams 400 --seed0 20000 --jobs $J --spots $(seq 20 121) >> $L 2>&1
# 5) 스무 곳 + 102곳 -> place_g16 (122곳)
[ -f study/place_g16.npz ] || $PY place_join.py >> $L 2>&1
# 6) 고르기: 두 곳 모두, 세 곳 모두
$PY place_study.py search --tag g16 --jobs $J >> $L 2>&1
echo PLACEDONE >> $L
