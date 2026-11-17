#!/bin/bash
# 구매 전 검증 계산: (1) 세기 재기가 실물에서 흔들릴 때 어디까지 버티나 (2) 더 싼 구성은 얼마나 맞히나 (3) 패널 하나로는
# 결과는 study/log_pre.txt. 이미 잰 것은 건너뛴다.
cd "$(dirname "$0")"; PY="$(cd ../../.. && pwd)/.venv-rt/bin/python"; L=study/log_pre.txt
SETS="kr40_t_s1 kr40_t_s2 kr40_t_s3 kr40_t_s4 kr40_t_s5 kr40_t_s6 kr40_t_s7 kr40_t_s8"
BASE='"mount":"hinge","tilt":15,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05'
run() { tag=$1; recv=$2; extra=$3; spots=$4; shift 4
  [ -f "study/sc_${recv}_$tag.npz" ] || $PY rt_study.py measure --recv $recv --n 560 --same --sets $SETS --tag $tag --opt "{$BASE,\"spots\":$spots$extra}" >> $L 2>&1
  echo "## $tag $recv $extra" >> $L; $PY rt_study.py table --recv $recv --tag $tag "$@" >> $L 2>&1; echo "$tag $(date +%T)"; }
P=ura4x4p37.5h; TWO='[7,17]'; T2="--combo 7,17 --tdiv 1 4"
echo "# 시작 $(date)" >> $L
# (1) 흔들림: 패킷마다의 이득, 위상 단계의 오차, 세기가 다른 패킷, 정수로 받기, 잡음
run i_g1 $P ',"imp":{"gjit":1.0}' $TWO $T2
run i_g2 $P ',"imp":{"gjit":2.0}' $TWO $T2
run i_g3 $P ',"imp":{"gjit":3.0}' $TWO $T2
run i_s2 $P ',"imp":{"sjit":2.0}' $TWO $T2
run i_p10 $P ',"imp":{"pherr":10}' $TWO $T2
run i_p20 $P ',"imp":{"pherr":20}' $TWO $T2
run i_p40 $P ',"imp":{"pherr":40}' $TWO $T2
run i_tx1 $P ',"imp":{"txp":0.1,"txdb":2}' $TWO $T2
run i_tx3 $P ',"imp":{"txp":0.3,"txdb":3}' $TWO $T2
run i_tx3r $P ',"imp":{"txp":0.3,"txdb":3,"rep":3}' $TWO $T2
run i_q6 $P ',"imp":{"q8":6}' $TWO $T2
run i_q4 $P ',"imp":{"q8":4}' $TWO $T2
run i_n15 $P ',"snr":15' $TWO $T2
run i_n10 $P ',"snr":10' $TWO $T2
run i_bad $P ',"snr":20,"imp":{"gjit":2.0,"pherr":15,"txp":0.2,"txdb":3}' $TWO $T2
run i_badr $P ',"snr":20,"imp":{"gjit":2.0,"pherr":15,"txp":0.2,"txdb":3,"rep":3}' $TWO $T2
# (2) 더 싼 구성: 수신 체인 넷이 한꺼번에 받는 작은 배열 (공유기 방식), 안테나 8개 패널
FOUR='[1,7,11,17]'
run c1 ura2x2 '' $FOUR --combo 7,17 1,7,17 --tdiv 1 4 8
run c1 ura1x4 '' $FOUR --combo 7,17 1,7,17 --tdiv 1 4 8
run c1 ura2x4p37.5h '' $FOUR --combo 7,17 1,7,17 --tdiv 1 4 8
run c1 ura2x2h '' $FOUR --combo 7,17 1,7,17 --tdiv 1 4 8
# (3) 패널 하나로 (이미 잰 t1 에서)
echo "## single t1" >> $L; $PY rt_study.py table --recv $P --tag t1 --combo 7 17 1 --tdiv 1 2 4 8 >> $L 2>&1
# (4) 세기만 재는 값싼 수신기 여러 개
echo "## rss" >> $L; $PY rt_rss.py >> $L 2>&1
echo PREDONE >> $L; echo PREDONE
