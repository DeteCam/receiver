"""'Measure more only when unsure': start with two wall positions; if the fused map is spread out, add positions.
Thresholds are set on the development rooms (classroom_a, lodging_a, restroom_a, locker_a) for a target share of
upgrades; results are reported on the other 13 rooms.  A small learned variant predicts the two-position error
from map features (leave-one-room-out) and upgrades the worst predicted half.
python3 mysim/analyze_adaptive.py <variant> <cfg> [seeds]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as A
from sklearn.ensemble import HistGradientBoostingRegressor

DEV = {"classroom_a", "lodging_a", "restroom_a", "locker_a"}
UP = {"exam": "front5", "other": "quadm"}


def rows(res, cfg, h="A", seeds=None):
    out = []
    for (sc, band, sd), (meta, o) in sorted(res.items()):
        if seeds is not None and sd not in seeds:
            continue
        oc = o[cfg]
        g2 = f"{A.gname('two', sc)}{h}"
        g4 = f"{A.gname('quadm', sc)}{h}"
        e2, e4 = oc["pos_err"][f"{g2}_equal"], oc["pos_err"][f"{g4}_equal"]
        e8 = oc["pos_err"][f"all8{h}_equal"]
        s2, s4 = oc["gspread"][g2], oc["gspread"][g4]
        sp = oc["spread"]
        idx = [A_IDX[l] + (0 if h == "A" else 8) for l in LABS[A.gname('two', sc)]]
        for r in range(len(e2)):
            out.append(dict(sc=sc, band=band, seed=sd, e2=float(e2[r]), e4=float(e4[r]), e8=float(e8[r]),
                            s2=float(s2[r]), s4=float(s4[r]), spa=float(sp[r, idx[0]]), spb=float(sp[r, idx[1]])))
    return out


LABS = {"c01": ("c0", "c1"), "c03": ("c0", "c3"), "c23": ("c2", "c3"), "diag": ("c0", "c2")}
A_IDX = {l: i for i, l in enumerate(["c0", "c1", "c2", "c3", "m0", "m1", "m2", "m3"])}


def policy(rs, score, frac_dev):
    dev = [i for i, r in enumerate(rs) if r["sc"] in DEV]
    thr = np.quantile([score[i] for i in dev], 1 - frac_dev)
    ev = [i for i, r in enumerate(rs) if r["sc"] not in DEV]
    e = np.array([rs[i]["e4"] if score[i] >= thr else rs[i]["e2"] for i in ev])
    moved = np.mean([score[i] >= thr for i in ev])
    return e, moved, ev


if __name__ == "__main__":
    variant, cfg = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    res = A.load(variant, cfg)
    rs = rows(res, cfg, seeds=seeds)
    room = np.array([r["sc"] for r in rs])
    ev = [i for i, r in enumerate(rs) if r["sc"] not in DEV]
    e2 = np.array([rs[i]["e2"] for i in ev]); e4 = np.array([rs[i]["e4"] for i in ev])
    re = room[ev]
    print(f"eval cameras {len(ev)}")
    print(f"  always two      {A.short(e2)}")
    print(f"  always four     {A.short(e4)}")
    # learned error predictor (leave-one-room-out) on map features
    X = np.array([[np.log(r["s2"] + 1e-3), np.log(r["s4"] + 1e-3), r["spa"], r["spb"]] for r in rs])
    y = np.log(np.array([r["e2"] for r in rs]) + 0.1)
    pred = np.zeros(len(rs))
    for rm in sorted(set(room)):
        tr, te = room != rm, room == rm
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    s2 = np.array([r["s2"] for r in rs])
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 0.75):
        er, mv, _ = policy(rs, s2, frac)
        ea, mva, _ = policy(rs, pred, frac)
        rnd = np.mean([np.where(rng.random(len(ev)) < frac, e4, e2).mean() for _ in range(50)])
        print(f"  upgrade ~{int(frac * 100)}%: spread rule {A.short(er)} (moved {100 * mv:.0f}%) | "
              f"AI {A.short(ea)} (moved {100 * mva:.0f}%) | random mean {rnd:.2f}")
        bt = A.by_type(ea, re)
        print("      AI by type: " + " | ".join(f"{t} {A.short(v)}" for t, v in bt.items()))
