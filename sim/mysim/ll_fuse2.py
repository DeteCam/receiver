"""Fusion of several spots with reliability weights / more spots (offline, stored maps).  [simulation]
Usage: python3 mysim/ll_fuse2.py <band> <cfg> <real pkl name>"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S
import real_scenes as R

band, cfg = sys.argv[1], sys.argv[2]
realp = os.path.join(S.ROOT, "results", sys.argv[3])
devp = os.path.join(S.ROOT, "results", f"dev_ll_{cfg}_{band}.pkl")
dev_res, real_res = S.load(devp), S.load(realp)
dev_rows = S.spot_table(dev_res, cfg, S.dev_spots, S.dev_looks)
model = S.train_model(dev_rows)
real_rows = S.spot_table(real_res, cfg, S.real_spots("all"), S.real_looks)
pb = model.predict_proba(S.xmat(real_rows))[:, 1]
score = {(r["scene"], r["seed"], r["cam"], r["node"]): (p, r["spread"]) for r, p in zip(real_rows, pb)}
# dev scores (leave-one-room-out) for the same rules on the old rooms
dev_pb = S.loro_scores(dev_rows)
dscore = {(r["scene"], r["seed"], r["cam"], r["node"]): (p, r["spread"]) for r, p in zip(dev_rows, dev_pb)}


def run(res, sc_score, groups_of, looks_of, rules):
    out = {}
    for key in sorted(res):
        sc, bd, seed = key
        mt, outs = res[key]
        if cfg not in outs or sc in R.EXAM:
            continue
        o = outs[cfg]
        LL = o["ll_all"].astype(np.float32); grid = o["grid"].astype(np.float64); cams = mt["cams"]
        for gname, base in groups_of(sc).items():
            for r in range(len(cams)):
                maps = [np.sum([LL[r, j] for j in looks_of(sc, i)], axis=0) for i in base]
                maps = [m - m.max() for m in maps]
                p = np.array([sc_score[(sc, seed, r, i)][0] for i in base])
                sp = np.array([sc_score[(sc, seed, r, i)][1] for i in base])
                for rn, w in rules(p, sp).items():
                    lt = sum(wi * m for wi, m in zip(w, maps))
                    pt = grid[int(np.argmax(lt))]
                    out.setdefault((gname, rn), []).append((sc, float(np.linalg.norm(pt[:2] - cams[r][:2])), float(np.sum(np.asarray(w) > 0))))
    return out


def rules(p, sp):
    n = len(p)
    d = {"equal": np.ones(n)}
    d["w=1-p"] = 1.0 - p
    d["w=(1-p)^2"] = (1.0 - p) ** 2
    w = 1.0 / np.maximum(sp, 10.0); d["w=1/spread"] = w * n / w.sum()
    for tau in (0.5, 0.35, 0.25):
        k = p <= tau
        if k.sum() < 2:
            k = np.zeros(n, bool); k[np.argsort(p)[:2]] = True
        d[f"drop p>{tau}"] = k.astype(float)
    k = np.ones(n); k[int(np.argmax(p))] = 0.0 if n > 2 else 1.0
    d["drop worst 1"] = k
    return d


def show(title, out, grp):
    print(f"\n## {title}")
    for (g, rn), v in out.items():
        if g != grp:
            continue
        e = np.array([x[1] for x in v]); used = np.mean([x[2] for x in v])
        per = {k: 100 * np.mean([x[1] <= 0.5 for x in v if R.KIND.get(x[0]) == k]) for k in ("lodging", "restroom", "locker")} if v[0][0] in R.KIND else {}
        print(f"  {rn:14s} within 0.5 m {100*np.mean(e<=0.5):5.1f} %  1 m {100*np.mean(e<=1.0):5.1f} %  median {np.median(e):.2f} m  spots used {used:.2f}  " +
              " ".join(f"{k} {x:.1f}" for k, x in per.items()))


print("# fusion with reliability weights (model trained on the old rooms)   [simulation]")
dv = run(dev_res, dscore, lambda sc: {"quadm": [4, 5, 6, 7], "all8": list(range(8))}, S.dev_looks, rules)
show("old rooms, 4 wall-centre spots", dv, "quadm")
show("old rooms, all 8 wall spots", dv, "all8")
def real_groups(sc):
    g = {"four": list(R.COMBO[sc]["four"]), "pair": list(R.COMBO[sc]["pair"])}
    g["all"] = [i for i in range(len(R.SPOTS[sc])) if R.SPOTS[sc][i][5] != "B"]
    g["all+B"] = list(range(len(R.SPOTS[sc])))
    return g
rv = run(real_res, score, real_groups, S.real_looks, rules)
for g, t in (("four", "realistic rooms, four spots"), ("pair", "realistic rooms, two spots"), ("all", "realistic rooms, every spot except inside the bathroom (5-7 spots)"), ("all+B", "realistic rooms, every spot (5-8)")):
    show(t, rv, g)
