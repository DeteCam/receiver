#!/bin/bash
# 전파 길을 구하고(Sionna RT), 경우마다 재는 과정을 돌린다 (호스트에서). 결과: rt_eval_result.json, 기록: rt_eval.log, paths/log.txt
cd "$(dirname "$0")"; PY=../../../.venv-rt/bin/python; mkdir -p paths
run() { name=$1; shift; [ -f paths/$name.npz ] || $PY rt_paths.py $name "$@" >> paths/log.txt 2>&1; }
for s in 1 2 3; do run kr40_h24_s$s --h 2.4 --seed $s; done
for h in 12 16 20 22 26; do run kr40_h${h}_s1 --h ${h:0:1}.${h:1:1} --seed 1; done
run kr00_h24 --h 2.4 --people 0
run kr20_h24_s1 --h 2.4 --people 20 --seed 1
run kr40_ceil_s1 --h 2.5 --seed 1 --spots ceil
run kr40_aisle_s1 --h 1.2 --seed 1 --spots aisle
run kr40_h24_s1_d5 --h 2.4 --seed 1 --depth 5
run kr40_h24_s1_nodiff --h 2.4 --seed 1 --no-diff
run kr40_h24_s1_board --h 2.4 --seed 1 --ceil ceiling_board
run kr40_h24_s1_hi --h 2.4 --seed 1 --samples 4000000
run lec72_h24_s1 --room lec72 --h 2.4 --seed 1
run lec72_h24_s2 --room lec72 --h 2.4 --seed 2
run lec72_h27_s1 --room lec72 --h 2.7 --seed 1
run lec72_ceil_s1 --room lec72 --h 2.8 --seed 1 --spots ceil
$PY rt_eval.py --n ${1:-200} --jobs 11 >> rt_eval.log 2>&1
echo ALLDONE >> rt_eval.log
