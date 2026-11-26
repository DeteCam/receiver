import sys, numpy as np
sys.path.insert(0, '/opt/detecam/sim/mysim')
import analyze as A
res = A.load(sys.argv[1] if len(sys.argv) > 1 else '/opt/detecam/sim/results/exp_base.pkl')
cfg = sys.argv[2] if len(sys.argv) > 2 else 'R2'
FRONT = {"classroom_a": "adj", "lecture_a": "adj", "classroom_b": "p02", "lecture_b": "p02", "classroom_c": "p23"}
PN = ["adj", "p02", "p13", "p23", "diag", "p12"]
names = {"adj": "(0,1)", "p02": "(0,2)", "p13": "(1,3)", "p23": "(2,3)", "diag": "(0,3)대각", "p12": "(1,2)대각"}
scenes = sorted({k[0] for k in res})
for band in ("2g4", "5g"):
    print(f"\n[{band}] 장면별 2자리 위치 오차 평균 m (같은 가중 결합) — 짝: " + " ".join(names[p] for p in PN) + "   | 앞쪽 짝")
    for sc in scenes:
        row = []
        for pn in PN:
            e, _ = A.collect(res, cfg, band, [sc], "pos_" + pn + "_equal")
            row.append(np.mean(e) if e is not None else np.nan)
        fr = FRONT.get(sc)
        best = PN[int(np.nanargmin(row))]
        print(f"  {sc:12s} " + " ".join(f"{v:5.2f}" for v in row) + f"   | 최선 {names[best]}" + (f" / 앞쪽 {names[fr]} {row[PN.index(fr)]:.2f} vs 대각 {row[PN.index('diag')]:.2f}" if fr else ""))
