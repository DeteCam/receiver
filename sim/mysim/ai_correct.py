"""AI correction of the multipath (reflection) bias on the fused likelihood map, beyond the re-ranker:
  RESID   gradient-boosted regression of the 2-D offset (truth - MAP point) from map features, added to the MAP point
          (shrunk by a factor chosen on the training rooms)
  RR+RES  re-ranked estimate, then the same residual correction
Leave-one-room-out on the 13 base rooms; final model trained on 13 rooms and tested on the 4 unseen rooms when that
data exists.  Usage: python3 ai_correct.py <cfg> [base.pkl] [fresh.pkl]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S
import ai_rerank as A
from sklearn.ensemble import HistGradientBoostingRegressor


def gfeat(row):
    """global map features + the top candidate's own features (row['F'][0])"""
    lt, g = row["lt"], row["grid"][:, :2]
    p = np.exp(lt - lt.max())
    p /= p.sum()
    mu = (p[:, None] * g).sum(0)
    cov = ((g - mu).T * p) @ (g - mu)
    c0 = g[row["sel"][0]]
    sec = row["F"][1] if len(row["F"]) > 1 else row["F"][0]
    return np.concatenate([row["F"][0], [mu[0] - c0[0], mu[1] - c0[1], np.sqrt(max(cov[0, 0], 0)), np.sqrt(max(cov[1, 1], 0)),
                                         cov[0, 1], sec[0], np.linalg.norm(g[row["sel"][1]] - c0) if len(row["sel"]) > 1 else 0.0,
                                         c0[0], c0[1]]])


def fit_resid(rows, base_fn):
    X = np.vstack([gfeat(r) for r in rows])
    Y = np.vstack([r["cam"] - base_fn(r) for r in rows])
    kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    mx, my = HistGradientBoostingRegressor(**kw), HistGradientBoostingRegressor(**kw)
    mx.fit(X, Y[:, 0])
    my.fit(X, Y[:, 1])
    return mx, my


def apply_resid(rows, models, base_fn, shrink):
    X = np.vstack([gfeat(r) for r in rows])
    d = np.stack([models[0].predict(X), models[1].predict(X)], axis=1)
    return np.array([np.linalg.norm(base_fn(r) + shrink * d[i] - r["cam"]) for i, r in enumerate(rows)])


def run(cfg, base_path=None, fresh_path=None):
    base = S.load(base_path or f"/opt/detecam/sim/results/exp_ll_{cfg}.pkl")
    rows = A.collect(base, cfg)
    rooms = sorted(set(r["room"] for r in rows))
    out = {k: np.zeros(len(rows)) for k in ("MAP", "RERANK", "RESID", "RR+RES")}
    map_fn = lambda r: r["grid"][r["sel"][0], :2]
    for room in rooms:
        tr = [r for r in rows if r["room"] != room]
        te_idx = [i for i, r in enumerate(rows) if r["room"] == room]
        te = [rows[i] for i in te_idx]
        m = A.fit(tr)
        h = A.tune(tr, m)
        A.predict_all(te, m)
        rr_fn = lambda r: A.est_rerank(r, m, h["tau"])
        # shrink factors chosen inside the training rooms (2-fold split by room)
        # shrink factor chosen by an inner split of the training rooms (fit on one half, score on the other)
        trooms = sorted(set(r["room"] for r in tr))
        half = set(trooms[::2])
        tA = [r for r in tr if r["room"] in half]
        tB = [r for r in tr if r["room"] not in half]
        A.predict_all(tB, m)
        best_s = min((np.mean(apply_resid(tB, fit_resid(tA, map_fn), map_fn, s_)), s_) for s_ in (0.0, 0.25, 0.5, 0.75, 1.0))[1]
        A.predict_all(tA, m)
        mA = fit_resid(tA, rr_fn)
        A.predict_all(tB, m)
        best_r = min((np.mean(apply_resid(tB, mA, rr_fn, s_)), s_) for s_ in (0.0, 0.25, 0.5, 0.75, 1.0))[1]
        A.predict_all(tr, m)
        res_m = fit_resid(tr, map_fn)
        res_r = fit_resid(tr, rr_fn)
        A.predict_all(te, m)
        out["MAP"][te_idx] = [r["dist"][0] for r in te]
        out["RERANK"][te_idx] = [np.linalg.norm(rr_fn(r) - r["cam"]) for r in te]
        out["RESID"][te_idx] = apply_resid(te, res_m, map_fn, best_s)
        out["RR+RES"][te_idx] = apply_resid(te, res_r, rr_fn, best_r)
    print(f"[{cfg}] leave-one-room-out (mean m / share <= 0.5 m)")
    A.report("LORO", rows, out)
    return rows, out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "d120_16", *(sys.argv[2:3] if len(sys.argv) > 2 else ()))
