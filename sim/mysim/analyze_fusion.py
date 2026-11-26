"""Offline fusion studies on stored per-node maps (exp_wall.py ... ll).
Rules are tuned on the development rooms only (classroom_a, lodging_a, restroom_a, locker_a) and reported on the
other 13 rooms (9 evaluation + 4 never-seen).  python3 mysim/analyze_fusion.py <variant> <cfg> [seeds]"""
import sys, os, glob, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as A

DEV = {"classroom_a", "lodging_a", "restroom_a", "locker_a"}
LAB = ["c0", "c1", "c2", "c3", "m0", "m1", "m2", "m3"]
# roles relative to the board wall (exam rooms): which pinwheel label is the front corner on the board wall, etc.
# order: front corner on board wall, front corner on side wall, front mid, side mid near front corner 2,
#        back mid, side mid near front corner 1, back corner (side wall), back corner (back wall)
ROLE = {
    "y0": ["c0", "c1", "m0", "m1", "m2", "m3", "c3", "c2"],   # board on wall y=0 (classroom_a, lecture_a)
    "x0": ["c3", "c0", "m3", "m0", "m1", "m2", "c2", "c1"],   # board on wall x=0 (classroom_b, lecture_b)
    "yw": ["c2", "c3", "m2", "m3", "m0", "m1", "c1", "c0"],   # board on wall y=W (classroom_c)
}
BOARD_WALL = {"classroom_a": "y0", "lecture_a": "y0", "classroom_b": "x0", "lecture_b": "x0", "classroom_c": "yw"}


def load(variant, cfg):
    res = {}
    for f in sorted(glob.glob(A.R + f"exp_wall_{variant}_{cfg}_ll_p*.pkl")):
        with open(f, "rb") as fh:
            res.update(pickle.load(fh))
    return res


def fuse(L, idx, rule, p=None, sp=None):
    """L: (n_node, G) maps (max-normalised log-likelihoods). Returns fused map."""
    X = L[list(idx)].astype(np.float64)
    if rule == "sum":
        return X.sum(0)
    if rule == "floor":            # tempered: no single receiver can push a point below -p
        return np.maximum(X, -p).sum(0)
    if rule == "spreadw":          # weights 1 / spread (deg), floored
        w = 1.0 / np.maximum(np.asarray(sp)[list(idx)], p)
        return (X * (w / w.mean())[:, None]).sum(0)
    if rule == "trim":             # drop the receiver whose own best point is farthest from the others' fused best point
        if len(idx) < 3:
            return X.sum(0)
        return None                # handled by caller (needs grid)
    raise ValueError(rule)


def evaluate(res, cfg, groups_fn, rule, p=None, scenes=None, bands=("2g4", "5g"), seeds=None):
    errs, rooms = [], []
    for (sc, band, sd), (meta, o) in sorted(res.items()):
        if band not in bands or (scenes is not None and sc not in scenes) or (seeds is not None and sd not in seeds):
            continue
        oc = o[cfg]
        Lall, grid, sp = oc["ll_all"], oc["grid"], oc["spread"]
        cams = np.asarray(meta["cams"])
        idx = groups_fn(sc)
        for r in range(len(cams)):
            if rule == "trim" and len(idx) >= 3:
                X = Lall[r][list(idx)].astype(np.float64)
                tot = X.sum(0)
                best_i, best_d = None, -1
                for j in range(len(idx)):
                    others = tot - X[j]
                    po = grid[int(np.argmax(others))]
                    pj = grid[int(np.argmax(X[j]))]
                    d = np.linalg.norm(po[:2] - pj[:2])
                    if d > best_d:
                        best_d, best_i = d, j
                lt = tot - X[best_i] if best_d > p else tot
            else:
                lt = fuse(Lall[r], idx, rule, p, sp[r])
            pt = grid[int(np.argmax(lt))]
            errs.append(float(np.linalg.norm(pt[:2] - cams[r][:2])))
            rooms.append(sc)
    return np.array(errs), np.array(rooms)


def idx_of(labels, h="A"):
    off = 0 if h == "A" else 8
    return tuple(LAB.index(l) + off for l in labels)


def std_groups(kind, h="A"):
    def f(sc):
        g = A.gname(kind, sc)
        labs = {"c01": ("c0", "c1"), "c03": ("c0", "c3"), "c23": ("c2", "c3"), "diag": ("c0", "c2"),
                "c01m2": ("c0", "c1", "m2"), "c03m1": ("c0", "c3", "m1"), "c23m0": ("c2", "c3", "m0"),
                "quadc": ("c0", "c1", "c2", "c3"), "quadm": ("m0", "m1", "m2", "m3"), "all8": tuple(LAB)}[g]
        return idx_of(labs, h)
    return f


def role_groups(roles, h="A"):
    """exam rooms: positions picked by role relative to the board wall; other rooms: the same roles with y0 as 'front'."""
    def f(sc):
        rl = ROLE[BOARD_WALL.get(sc, "y0")]
        return idx_of([rl[i] for i in roles], h)
    return f


def report(e, room, label):
    bt = A.by_type(e, room)
    print(f"  {label:34s} all {A.short(e):18s} | " + " | ".join(f"{t} {A.short(v)}" for t, v in bt.items()))


if __name__ == "__main__":
    variant, cfg = sys.argv[1], sys.argv[2]
    res = load(variant, cfg)
    allrooms = sorted({k[0] for k in res})
    evalrooms = [s for s in allrooms if s not in DEV]
    print(f"rooms {len(allrooms)}, eval rooms {len(evalrooms)}")
    # --- 1. robust fusion rules, tuned on DEV ---
    for kind in ("two", "quadm", "all8"):
        gf = std_groups(kind)
        print(f"\n== fusion rule, group {kind} (A): DEV tuning")
        cands = [("sum", None)] + [("floor", p) for p in (0.5, 1.0, 1.5, 2.0, 3.0)] + \
                [("spreadw", p) for p in (2.0, 5.0, 10.0)] + ([("trim", p) for p in (0.5, 1.0, 2.0)] if kind != "two" else [])
        devscore = {}
        for rule, p in cands:
            e, room = evaluate(res, cfg, gf, rule, p, scenes=DEV)
            devscore[(rule, p)] = e.mean()
            print(f"  DEV {rule:8s} {str(p):5s} mean {e.mean():.3f}  <=0.5 {100 * np.mean(e <= 0.5):.0f}%")
        best = min(devscore, key=devscore.get)
        e0, room = evaluate(res, cfg, gf, "sum", None, scenes=evalrooms)
        e1, _ = evaluate(res, cfg, gf, best[0], best[1], scenes=evalrooms)
        print(f"  -> chosen on DEV: {best}")
        report(e0, room, "EVAL sum")
        report(e1, room, f"EVAL {best[0]} {best[1]}")
        d, lo, hi = A.paired(e0, e1, room)
        print(f"  paired delta (chosen - sum) {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # --- 2. exam rooms: position roles ---
    exam = [s for s in allrooms if A.TYPE[s] == "시험실"]
    print("\n== exam rooms, groups by role (0 front corner on board wall, 1 front corner on side wall, 2 front mid, "
          "3 side mid A, 4 back mid, 5 side mid B, 6 back corner side wall, 7 back corner back wall)")
    for roles in [(0, 1), (0, 2), (1, 2), (0, 1, 2), (0, 1, 3), (0, 1, 5), (0, 1, 2, 3), (0, 1, 3, 5), (0, 1, 2, 3, 5),
                  (0, 1, 4), (0, 1, 2, 3, 4, 5), tuple(range(8))]:
        e, room = evaluate(res, cfg, role_groups(roles), "sum", None, scenes=exam)
        print(f"  roles {str(roles):22s} {A.short(e)}")
