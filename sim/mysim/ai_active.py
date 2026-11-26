"""AI decides when to move: measure once; a model trained on other rooms predicts the error of that estimate from the
map alone; only when the predicted error is above a threshold the head makes its one-axis move (two more looks).
D120 16-element head.  Maps of the single look: exp_ll_d120_16.pkl (base data); results with the move: exp_combo.pkl
(jit10, same rooms/cameras/noise for the first look, seeds 0-1).  Leave-one-room-out for the error model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S
import ai_rerank as A
import ai_correct as C
from sklearn.ensemble import HistGradientBoostingRegressor

ll = S.load("/opt/detecam/sim/results/exp_ll_d120_16.pkl")
combo = S.load("/opt/detecam/sim/results/exp_combo.pkl")
cfg = "d120_16"
rows = [r for r in A.collect({k: v for k, v in ll.items() if k[2] in (0, 1)}, cfg)]
# attach the error after the move from exp_combo (same scene, band, seed, camera order)
idx = {}
for r in rows:
    key = (r["room"], r["band"], r["seed"], r["grp"])
    idx.setdefault(key, []).append(r)
ok = []
for key, rs in idx.items():
    sc, band, seed, grp = key
    o = combo.get((sc, band, seed))
    if o is None or cfg not in o[1]:
        continue
    pk = "all4" if grp == "all4" else (S.BOARD[sc] if sc in S.BOARD else "diag")
    s_err = o[1][cfg]["pos_err"][f"s|{pk}_equal"]
    x_err = o[1][cfg]["pos_err"][f"x|{pk}_equal"]
    for i, r in enumerate(rs):
        if abs(s_err[i] - r["dist"][0]) > 1e-3:      # same estimate expected for the first look
            r["mismatch"] = True
        r["e_s"], r["e_x"] = float(s_err[i]), float(x_err[i])
        ok.append(r)
mm = np.mean([r.get("mismatch", False) for r in ok])
print(f"paired rows {len(ok)}, first-look mismatch rate {mm:.3f}")
rooms = sorted(set(r["room"] for r in ok))
pred = np.zeros(len(ok))
for room in rooms:
    tr = [r for r in ok if r["room"] != room]
    te = [i for i, r in enumerate(ok) if r["room"] == room]
    X = np.vstack([C.gfeat(r) for r in tr])
    y = np.log(np.array([r["e_s"] for r in tr]) + 0.1)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=0).fit(X, y)
    pred[te] = m.predict(np.vstack([C.gfeat(ok[i]) for i in te]))
es = np.array([r["e_s"] for r in ok]); ex = np.array([r["e_x"] for r in ok])
for name, sel in (("exam board pair", [r["room"] in S.BOARD and r["grp"] == "pair" for r in ok]),
                  ("other diag", [r["room"] not in S.BOARD and r["grp"] == "pair" for r in ok]),
                  ("other all4", [r["grp"] == "all4" for r in ok])):
    sel = np.array(sel)
    p, a, b = pred[sel], es[sel], ex[sel]
    line = [f"never move {a.mean():.2f}", f"always move {b.mean():.2f}"]
    for frac in (0.25, 0.5, 0.75):
        thr = np.quantile(p, 1 - frac)
        e = np.where(p >= thr, b, a)
        rnd = np.mean([np.where(np.random.default_rng(k).random(len(a)) < frac, b, a).mean() for k in range(50)])
        line.append(f"move {int(frac * 100)}%: AI {e.mean():.2f} / random {rnd:.2f}")
    print(f"{name:16s} n={sel.sum():4d}  " + "  ".join(line))
