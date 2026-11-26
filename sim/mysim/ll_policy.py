"""Bearing procedures built on a reliability score (offline, stored maps).  Thresholds / model from the OLD rooms.
Usage: python3 mysim/ll_policy.py <band> <cfg> <real pkl name> [spot set four|all]   [simulation]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S
import real_scenes as R

band, cfg = sys.argv[1], sys.argv[2]
realp = os.path.join(S.ROOT, "results", sys.argv[3])
devp = os.path.join(S.ROOT, "results", f"dev_ll_{cfg}_{band}.pkl")
dev = S.spot_table(S.load(devp), cfg, S.dev_spots, S.dev_looks)
real = [r for r in S.spot_table(S.load(realp), cfg, S.real_spots("four"), S.real_looks) if r["scene"] not in R.EXAM]
model = S.train_model(dev)
sc_dev = {"spread": np.array([r["spread"] for r in dev]), "learned": S.loro_scores(dev)}
sc_real = {"spread": np.array([r["spread"] for r in real]), "learned": model.predict_proba(S.xmat(real))[:, 1]}
order_of = lambda sc: [R.SPOTS[sc][i][0] for i in R.COMBO[sc]["four"]]  # noqa: E731


def by_cam(rows, score):
    d = {}
    for r, s in zip(rows, score):
        d.setdefault((r["scene"], r["seed"], r["cam"]), {})[r["spot"]] = (s, r["err"])
    return d


def proc(d, thr, kmin, kmax):
    """measure kmin spots; while the best score so far is above thr and spots remain (up to kmax), measure one more;
    report the bearing of the most reliable spot measured"""
    errs, used = [], []
    for (sc, seed, cam), m in d.items():
        labs = [l for l in order_of(sc) if l in m]
        k = kmin
        while True:
            best = min(labs[:k], key=lambda l: m[l][0])
            if m[best][0] <= thr or k >= min(kmax, len(labs)):
                break
            k += 1
        errs.append(m[best][1]); used.append(k)
    return np.array(errs), np.array(used)


def st(e, u):
    return f"{e.mean():5.2f} {np.median(e):4.1f} {np.percentile(e,90):5.1f} {np.percentile(e,95):5.1f}  {100*np.mean(e>20):4.1f}%  {u.mean():.2f}"


e_all = np.array([r["err"] for r in real])
print("# bearing procedures on the realistic outside rooms (four-combo spots); columns: mean median p90 p95 | gross>20 | mean spots   [simulation]")
print(f"every measurement counted (plan metric)        {e_all.mean():5.2f} {np.median(e_all):4.1f} {np.percentile(e_all,90):5.1f} {np.percentile(e_all,95):5.1f}  {100*np.mean(e_all>20):4.1f}%  1.00")
for nm in ("spread", "learned"):
    d = by_cam(real, sc_real[nm])
    print(f"-- score: {nm}")
    for kmin, kmax in ((1, 4), (2, 4)):
        for acc in (0.9, 0.8, 0.7, 0.6, 0.5):
            thr = float(np.quantile(sc_dev[nm], acc))
            e, u = proc(d, thr, kmin, kmax)
            print(f"  start with {kmin}, up to {kmax}, accept level {int(100*acc)}% of old-room measurements: {st(e, u)}")
    for k in (2, 3, 4):
        e, u = proc(d, -1e9, k, k)
        print(f"  always {k} spots, report the most reliable one:                    {st(e, u)}")
# matched comparison: mean error at the same mean number of spots (interpolated)
print("-- learned vs spread at the same mean number of spots (start with 1, up to 4)")
for target in (1.25, 1.5, 1.75, 2.0):
    row = []
    for nm in ("spread", "learned"):
        d = by_cam(real, sc_real[nm]); pts = []
        for acc in np.linspace(0.3, 0.97, 40):
            e, u = proc(d, float(np.quantile(sc_dev[nm], acc)), 1, 4)
            pts.append((u.mean(), e.mean(), np.percentile(e, 95)))
        pts.sort()
        us = np.array([p[0] for p in pts]); row.append((np.interp(target, us, [p[1] for p in pts]), np.interp(target, us, [p[2] for p in pts])))
    print(f"  {target:.2f} spots: spread rule mean {row[0][0]:.2f} (p95 {row[0][1]:.1f})  |  learned mean {row[1][0]:.2f} (p95 {row[1][1]:.1f})")
