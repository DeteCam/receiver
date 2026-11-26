"""Software / AI post-processing of the fused likelihood map (no change to the radio part).
  MAP        grid point with the highest fused log-likelihood (the current procedure)
  PMEAN      posterior mean over the whole grid (temperature chosen on the training rooms)
  LCENT      posterior-weighted centroid of the points within r of the MAP point (r chosen on the training rooms)
  RERANK     gradient-boosted trees score the top-K separated candidates (features of the map only) and the
             estimate is the softmin-weighted mix of the candidates (weights' temperature chosen on training rooms)
  ORACLE     the best of the same top-K candidates (what a perfect re-ranker could reach)
Protocol: leave-one-room-out over the 13 base rooms (all seeds, both bands); the 4 fresh rooms are only used as a
final test of a model trained on all 13."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S
from sklearn.ensemble import HistGradientBoostingRegressor

K = 8
NMS = 0.5


def groups_for(sc):
    if sc in S.BOARD:
        return {"pair": S.BOARD[sc]}
    return {"pair": "diag", "all4": "all4"}


PIDX = {"adj": (0, 1), "diag": (0, 3), "p02": (0, 2), "p12": (1, 2), "p13": (1, 3), "p23": (2, 3), "all4": (0, 1, 2, 3)}


def topk(lt, grid, k=K, r=NMS):
    order = np.argsort(-lt)
    sel = []
    for i in order:
        if all(np.linalg.norm(grid[i, :2] - grid[j, :2]) > r for j in sel):
            sel.append(i)
            if len(sel) == k:
                break
    return np.array(sel)


def feats(lt, lls, grid, nodes, dims_hint, sel, band):
    p = np.exp(lt - lt.max())
    p /= p.sum()
    g2 = grid[:, :2]
    F = []
    mx = lt.max()
    for rank, c in enumerate(sel):
        d = np.linalg.norm(g2 - g2[c], axis=1)
        near = d < 0.6
        ring = (d > 0.3) & (d < 0.9)
        dn = np.linalg.norm(nodes[:, :2] - g2[c][None], axis=1)
        v = g2[c][None] - nodes[:, :2]
        ang = np.arctan2(v[:, 1], v[:, 0])
        sub = np.abs(np.angle(np.exp(1j * (ang[:, None] - ang[None, :]))))
        per = [float(l[c] - l.max()) for l in lls]
        L, W = dims_hint
        F.append([lt[c] - mx, rank, min(per), max(per), float(p[near].sum()),
                  float(lt[c] - (lt[ring].mean() if ring.any() else lt[c])),
                  dn.min(), dn.max(), float(sub.max()),
                  min(g2[c, 0], L - g2[c, 0], g2[c, 1], W - g2[c, 1]), grid[c, 2],
                  float(np.linalg.norm(g2[c] - g2[sel[0]])), len(lls), float(band == "5g"), L * W])
    return np.array(F, dtype=np.float64)


def collect(res, cfg):
    rows = []
    for (sc, band, seed), (meta, o) in res.items():
        if cfg not in o or "ll" not in o[cfg]:
            continue
        ll = o[cfg]["ll"].astype(np.float64)
        grid = o[cfg]["grid"].astype(np.float64)
        nodes = o[cfg]["node_pos_est"].astype(np.float64)
        cams = meta["cams"]
        dims_hint = (grid[:, 0].max() + 0.15, grid[:, 1].max() + 0.15)
        for gname, pk in groups_for(sc).items():
            idx = PIDX[pk]
            for r in range(len(cams)):
                lls = [ll[r, i] for i in idx]
                lt = sum(lls) + (o[cfg]["logprior"].astype(np.float64) if "logprior" in o[cfg] else 0.0)
                sel = topk(lt, grid)
                F = feats(lt, lls, grid, nodes[list(idx)], dims_hint, sel, band)
                dist = np.linalg.norm(grid[sel, :2] - cams[r, :2][None], axis=1)
                rows.append(dict(room=sc, band=band, seed=seed, grp=gname, F=F, dist=dist, sel=sel, lt=lt, grid=grid,
                                 cam=cams[r, :2]))
    return rows


def est_pmean(row, T):
    w = np.exp((row["lt"] - row["lt"].max()) / T)
    w /= w.sum()
    return (w[:, None] * row["grid"][:, :2]).sum(0)


def est_lcent(row, rad):
    g = row["grid"][:, :2]
    c = g[row["sel"][0]]
    m = np.linalg.norm(g - c, axis=1) <= rad
    w = np.exp(row["lt"][m] - row["lt"][m].max())
    return (w[:, None] * g[m]).sum(0) / w.sum()


def err(rows, fn):
    return np.array([np.linalg.norm(fn(r) - r["cam"]) for r in rows])


def fit(rows):
    X = np.vstack([r["F"] for r in rows])
    y = np.concatenate([np.log(r["dist"] + 0.1) for r in rows])
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    return m


def predict_all(rows, model):
    """One batched predict for all rows (per-row calls are slow); stores row['pr']."""
    if not rows:
        return
    P = model.predict(np.vstack([r["F"] for r in rows]))
    i = 0
    for r in rows:
        r["pr"] = P[i:i + len(r["F"])]
        i += len(r["F"])


def est_rerank(row, model, tau):
    pr = row["pr"]
    w = np.exp(-(pr - pr.min()) / tau)
    w /= w.sum()
    return (w[:, None] * row["grid"][row["sel"], :2]).sum(0)


def tune(rows, model=None):
    best = {}
    best["T"] = min((np.mean(err(rows, lambda r: est_pmean(r, T))), T) for T in (0.5, 1, 2, 4, 8))[1]
    best["rad"] = min((np.mean(err(rows, lambda r: est_lcent(r, rad))), rad) for rad in (0.3, 0.6, 0.9, 1.2))[1]
    if model is not None:
        predict_all(rows, model)
        best["tau"] = min((np.mean(err(rows, lambda r: est_rerank(r, model, tau))), tau)
                          for tau in (0.05, 0.15, 0.3, 0.6))[1]
    return best


def report(name, rows, ests):
    out = []
    for g in ("pair", "all4"):
        sub = [i for i, r in enumerate(rows) if r["grp"] == g]
        for ex in (True, False):
            ii = [i for i in sub if (rows[i]["room"] in S.BOARD) == ex]
            if not ii:
                continue
            tag = ("exam board pair" if ex else "other " + ("diag" if g == "pair" else "all4"))
            s = "  ".join(f"{k}={np.mean(v[ii]):.2f}/{100 * np.mean(v[ii] <= 0.5):.0f}%" for k, v in ests.items())
            out.append(f"  {name:6s} {tag:16s} n={len(ii):4d}  {s}")
    print("\n".join(out))


def main(cfg="d85", base_path=None, fresh_path=None):
    base = S.load(base_path or f"/opt/detecam/sim/results/exp_ll_{cfg}.pkl")
    fp = fresh_path or f"/opt/detecam/sim/results/exp_ll_{cfg}_fresh.pkl"
    fresh = S.load(fp) if os.path.exists(fp) else {}
    rows = collect(base, cfg)
    rows_f = collect(fresh, cfg)
    rooms = sorted(set(r["room"] for r in rows))
    ests = {k: np.zeros(len(rows)) for k in ("MAP", "PMEAN", "LCENT", "RERANK", "ORACLE")}
    for room in rooms:
        tr = [r for r in rows if r["room"] != room]
        te = [i for i, r in enumerate(rows) if r["room"] == room]
        m = fit(tr)
        h = tune(tr, m)
        predict_all([rows[i] for i in te], m)
        for i in te:
            r = rows[i]
            ests["MAP"][i] = r["dist"][0]
            ests["PMEAN"][i] = np.linalg.norm(est_pmean(r, h["T"]) - r["cam"])
            ests["LCENT"][i] = np.linalg.norm(est_lcent(r, h["rad"]) - r["cam"])
            ests["RERANK"][i] = np.linalg.norm(est_rerank(r, m, h["tau"]) - r["cam"])
            ests["ORACLE"][i] = r["dist"].min()
    print(f"[{cfg}] leave-one-room-out, 13 base rooms (mean m / share <= 0.5 m)")
    report("LORO", rows, ests)
    if not rows_f:
        return ests, None
    m = fit(rows)
    h = tune(rows, m)
    predict_all(rows_f, m)
    ef = {"MAP": np.array([r["dist"][0] for r in rows_f]),
          "PMEAN": err(rows_f, lambda r: est_pmean(r, h["T"])),
          "LCENT": err(rows_f, lambda r: est_lcent(r, h["rad"])),
          "RERANK": err(rows_f, lambda r: est_rerank(r, m, h["tau"])),
          "ORACLE": np.array([r["dist"].min() for r in rows_f])}
    print(f"[{cfg}] trained on 13 base rooms, tested on 4 fresh rooms; tuned {h}")
    report("FRESH", rows_f, ef)
    return ests, ef


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d85", *(sys.argv[2:4] if len(sys.argv) > 2 else ()))
