#!/usr/bin/env bash
# NEC2(PyNEC) 안테나 패턴을 처음부터 만든다. 순서와 합치는 방식은 11-26 작업 공간의 실제 파일 키로 확인함.
set -e
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
python3 mysim/ring_pynec.py mysim/nec_patterns.npy          # 기본 고리 + 상자
python3 mysim/ring_pynec.py mysim/nec_patterns.npy size     # 간격 · 접지판 변형
python3 mysim/ring_pynec.py mysim/nec_patterns.npy gap      # 간격 80/100 mm
python3 mysim/ring_pynec.py mysim/nec_ring.npy ring         # D120 · D160 16소자
python3 mysim/shapes.py mysim/nec_shapes.npy                # 수신기 모양 비교용
python3 mysim/wall_heads.py mysim/nec_wall.npy              # 벽 부착 머리 + 끄덕인 자세(d120_16_ty±20)
python3 mysim/wall_nf.py mysim/nec_wallnf.npy               # 면 근접 결합(자유공간 · 콘크리트 · 석고보드 · 금속)
python3 - <<'PY'
import numpy as np
L = lambda f: np.load(f"mysim/{f}.npy", allow_pickle=True).item()
def merge(out, *parts):
    d = {}
    for p in parts: d.update(L(p))
    np.save(f"mysim/{out}.npy", d, allow_pickle=True); print(out, len(d))
merge("nec_all",  "nec_patterns", "nec_ring")
merge("nec_all2", "nec_all", "nec_shapes")
merge("nec_all3", "nec_all2", "nec_wall")
merge("nec_all4", "nec_all3", "nec_wallnf")   # exp_wall.py · exp_final.py가 쓰는 표
PY
# nec_final.npy(앵커 H1·H2·H4·H5 손실을 곱한 표)는 exp_final.py가 처음 실행될 때 nec_all4.npy에서 스스로 만든다.
