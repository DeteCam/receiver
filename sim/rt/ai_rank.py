"""Learned layers on top of the map-aware likelihood (feature export of pe2.py).  One sample = one receiver at one spot.
 A) re-ranking: choose among the three strongest azimuth modes of the likelihood map
 B) confidence: "is this direction estimate within 10 degrees?"  -> which measurements to keep, when to move the receiver
 C) visibility: "does the direct path carry at least 20 % of the power?"
Every model is trained in one room and tested in the other room (never the same room).  Only quantities the receiver can compute itself
are used as inputs.     Usage: ai_rank.py [gbm|logit]"""
import sys, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SETS = {"classroom": ["f_c24", "f_c5"], "motel": ["f_m24", "f_m5"]}      # four corner spots, 2.4 and 5 GHz, ideal antennas
KIND = sys.argv[1] if len(sys.argv) > 1 else "gbm"

def load(tags):
    rows = []
    for t in tags:
        try: rows += [dict(r, tag=t) for r in np.load(f"res_pe2_{t}.npy", allow_pickle=True).item()["rows"] if not r["beacon"]]
        except FileNotFoundError: print("  (missing", t + ")")
    return rows

def mode_feats(rows):
    """Per mode: log-likelihood below the best, margin over the best other mode, energy explained with / without the direct path, their
    difference, share of the direct path in the fit, plain-beamforming power there, angle to the beamforming peak, posterior spread, rank."""
    X, y, g = [], [], []
    for i, r in enumerate(rows):
        F = r["feats"]; valid = [k for k in range(3) if F[k, 0] > -49]
        for k in valid:
            oth = max([F[j, 0] for j in valid if j != k], default=-30.0)
            X.append([F[k, 0], F[k, 0] - oth, F[k, 1], F[k, 2], F[k, 1] - F[k, 2], F[k, 3], F[k, 5], min(F[k, 6], 90.0), min(r["spread"], 90.0), float(k), float(len(valid))])
            y.append(F[k, 7] <= 10.0); g.append(i)
    return np.array(X), np.array(y), np.array(g)

def sample_feats(rows):
    """Per sample: frequency coherence at two lags, amplitude ripple over the ring, energy share of the best single direction, energy explained
    by the map model with / without the direct path, posterior spread, margin to the second mode, direct-path share, angle between the
    map-aware and the plain-beamforming direction."""
    X = []
    for r in rows:
        F = r["feats"]; m2 = F[1, 0] if F[1, 0] > -49 else -30.0
        X.append(list(r["xf"]) + [r["xf"][4] - r["xf"][5], min(r["spread"], 90.0), -m2, F[0, 3], min(F[0, 6], 90.0)])
    return np.array(X)

def st(e): return f"mean {e.mean():5.1f}  median {np.median(e):5.1f}  P90 {np.percentile(e, 90):5.1f}  <=10deg {100 * np.mean(e <= 10):3.0f}%"

def make():
    if KIND == "logit": return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=0)

def run(tr, te, ntr, nte):
    print(f"\n=== trained in the {ntr} ({len(tr)} samples), tested in the {nte} ({len(te)} samples)  [{KIND}]")
    err = np.array([r["err"] for r in te]); sp = np.array([r["spread"] for r in te]); lf = np.array([r["lf"] for r in te])
    # A) re-ranking
    Xa, ya, _ = mode_feats(tr); ma = make().fit(Xa, ya); X, y, g = mode_feats(te); p = ma.predict_proba(X)[:, 1]; rr, orc = [], []
    for i, r in enumerate(te):
        m = np.where(g == i)[0]; ks = X[m, 9].astype(int); F = r["feats"]; rr.append(F[ks[int(np.argmax(p[m]))], 7]); orc.append(F[ks, 7].min())
    rr, orc = np.array(rr), np.array(orc)
    print(f"  A) likelihood peak (no learning)       {st(err)}\n     learned choice among 3 modes        {st(rr)}\n     best of the 3 modes (upper bound)   {st(orc)}")
    # B) confidence
    S, St = sample_feats(tr), sample_feats(te); mb = make().fit(S, np.array([r["err"] for r in tr]) <= 10.0); c = mb.predict_proba(St)[:, 1]
    print(f"  B) 'within 10 deg' AUC: posterior spread {roc_auc_score(err <= 10, -sp):.2f} | learned confidence {roc_auc_score(err <= 10, c):.2f}")
    for cov in (80, 60, 40):
        k1 = sp <= np.percentile(sp, cov); k2 = c >= np.percentile(c, 100 - cov)
        print(f"     keep the most confident {cov}%: spread -> mean {err[k1].mean():5.1f} P90 {np.percentile(err[k1], 90):5.1f}   |   learned -> mean {err[k2].mean():5.1f} P90 {np.percentile(err[k2], 90):5.1f}")
    # C) visibility of the direct path
    mc = make().fit(S, np.array([r["lf"] for r in tr]) >= 0.2); v = mc.predict_proba(St)[:, 1]
    if 0 < np.mean(lf >= 0.2) < 1:
        print(f"  C) direct path visible (true share {100 * np.mean(lf >= 0.2):.0f}%): AUC {roc_auc_score(lf >= 0.2, v):.2f};  estimates the model calls 'visible' "
              f"({100 * np.mean(v >= 0.5):.0f}%): {st(err[v >= 0.5]) if (v >= 0.5).any() else '-'}")
    return dict(err=err, rr=rr, orc=orc, conf=c, spread=sp, vis=v, lf=lf)

if __name__ == "__main__":
    data = {k: load(v) for k, v in SETS.items()}; out = {}
    for a, b in (("motel", "classroom"), ("classroom", "motel")):
        if data[a] and data[b]: out[b] = run(data[a], data[b], a, b)
    np.save(f"res_ai_rank_{KIND}.npy", out, allow_pickle=True)
