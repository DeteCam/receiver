#!/bin/bash
# 숙소 방: 길 묶음 셋 (숨긴 자리를 다르게 뽑아) -> 패널로 재기 -> 표
cd "$(dirname "$0")"; PY="$(cd ../../.. && pwd)/.venv-rt/bin/python"; L=study/log_room.txt; mkdir -p study paths
for s in 1 2 3; do
  [ -f paths/room_s$s.npz ] || $PY room_paths.py room_s$s --seed $((300+s)) > paths/log_room_s$s.txt 2>&1
  echo "room_s$s $(date +%T) $(tail -1 paths/log_room_s$s.txt)"
done
echo "# 시작 $(date)" >> $L
[ -f study/room_ura4x4p37.5h.npz ] || $PY room_study.py measure >> $L 2>&1
for f in sum center; do $PY room_study.py table --fuse $f --brief >> $L 2>&1; done       # 겹치는 법 견주기
for d in 3 10; do $PY room_study.py table --drop $d --brief >> $L 2>&1; done              # 약한 쪽을 버리는 문턱
$PY room_study.py table >> $L 2>&1                                                        # 기본 (power, 6 dB)
[ -f study/room_ura4x4p37.5h_bad.npz ] || $PY room_study.py measure --tag bad --opt '{"snr":20,"imp":{"gjit":2.0,"pherr":15,"txp":0.2,"txdb":3,"rep":3}}' >> $L 2>&1
$PY room_study.py table --tag bad --brief >> $L 2>&1
echo ROOMDONE >> $L; echo ROOMDONE
