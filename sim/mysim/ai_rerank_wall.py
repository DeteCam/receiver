"""Candidate re-ranking (gradient-boosted trees on map-only features, ai_rerank.py) applied to the wall-mounted
receivers.  Data: results/exp_wall_base_d120_16_ll_pall.pkl (seed 0, every node's map, height A nodes 0-7).
Groups as pre-registered: two = board pair (exam rooms) / diagonal corners (others), four = mid-wall m0-m3,
eight = all eight positions.  Protocol: leave-one-room-out over all 17 rooms (each room's camera errors come from a
model that never saw that room); temperatures tuned on the training rooms only."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ai_rerank as A
import analyze_wall as W

PAIR = {"c01": (0, 1), "c03": (0, 3), "c23": (2, 3), "diag": (0, 2)}
GROUPS = lambda sc: {"two": PAIR[W.BOARD.get(sc, "diag")], "four": (4, 5, 6, 7), "eight": tuple(range(8))}


def rows_from(path, cfg="d120_16", seeds=None):
    with open(path, "rb") as f:
        res = pickle.load(f)
    rows = []
    for (sc, band, seed), (meta, o) in sorted(res.items()):
        if seeds is not None and seed not in seeds:
            continue
        oc = o[cfg]
        ll = oc["ll_all"].astype(np.float64)
        grid = oc["grid"].astype(np.float64)
        nodes = np.asarray(meta["nodes"], dtype=np.float64)   # all 16 nodes (the stored estimate keeps 4 only;
                                                               # features use node geometry, 5 cm does not matter)
        cams = meta["cams"]
        dims_hint = (grid[:, 0].max() + 0.15, grid[:, 1].max() + 0.15)
        for g, idx in GROUPS(sc).items():
            for r in range(len(cams)):
                lls = [ll[r, i] for i in idx]
                lt = sum(lls)
                sel = A.topk(lt, grid)
                F = A.feats(lt, lls, grid, nodes[list(idx)], dims_hint, sel, band)
                dist = np.linalg.norm(grid[sel, :2] - cams[r, :2][None], axis=1)
                rows.append(dict(room=sc, band=band, seed=seed, grp=g, F=F, dist=dist, sel=sel, lt=lt, grid=grid,
                                 cam=cams[r, :2]))
    return rows


def main(seeds=None):
    rows = rows_from("/opt/detecam/sim/results/exp_wall_base_d120_16_ll_pall.pkl", seeds=seeds)
    rooms = sorted(set(r["room"] for r in rows))
    E = {k: np.zeros(len(rows)) for k in ("MAP", "PMEAN", "LCENT", "RERANK", "ORACLE")}
    for room in rooms:
        tr = [r for r in rows if r["room"] != room]
        te = [i for i, r in enumerate(rows) if r["room"] == room]
        m = A.fit(tr)
        h = A.tune(tr, m)
        A.predict_all([rows[i] for i in te], m)
        for i in te:
            r = rows[i]
            E["MAP"][i] = r["dist"][0]
            E["PMEAN"][i] = np.linalg.norm(A.est_pmean(r, h["T"]) - r["cam"])
            E["LCENT"][i] = np.linalg.norm(A.est_lcent(r, h["rad"]) - r["cam"])
            E["RERANK"][i] = np.linalg.norm(A.est_rerank(r, m, h["tau"]) - r["cam"])
            E["ORACLE"][i] = r["dist"].min()
    room = np.array([r["room"] for r in rows])
    grp = np.array([r["grp"] for r in rows])
    exam = np.array([W.TYPE[x] == "시험실" for x in room])
    for g in ("two", "four", "eight"):
        for ex, lab in ((True, "시험실"), (False, "그 밖")):
            ii = np.where((grp == g) & (exam == ex))[0]
            s = "  ".join(f"{k} {E[k][ii].mean():.2f}/{100 * np.mean(E[k][ii] <= 0.5):.0f}%" for k in E)
            d, lo, hi = W.paired(E["MAP"][ii], E["RERANK"][ii], room[ii])
            print(f"{g:5s} {lab:4s} n={len(ii):4d} {s} | RERANK-MAP {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        ii = np.where((grp == g) & ~exam)[0]
        print("      type share<=0.5 RERANK: " + " ".join(f"{t}{100*np.mean(v<=0.5):.0f}%" for t, v in W.by_type(E['RERANK'][ii], room[ii]).items())
              + " | MAP: " + " ".join(f"{t}{100*np.mean(v<=0.5):.0f}%" for t, v in W.by_type(E['MAP'][ii], room[ii]).items()))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0])
