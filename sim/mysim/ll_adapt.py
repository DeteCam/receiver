"""'Measure more spots when unsure' (offline, stored maps).  Spots are added in a fixed order; after each spot the
fused map's spatial spread decides whether to stop.  The stop threshold is fixed on the OLD rooms.  [simulation]
Usage: python3 mysim/ll_adapt.py <band> <cfg> <real pkl name>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S
import real_scenes as R

band, cfg = sys.argv[1], sys.argv[2]
realp = os.path.join(S.ROOT, "results", sys.argv[3])
devp = os.path.join(S.ROOT, "results", f"dev_ll_{cfg}_{band}.pkl")


def seq(res, order_of, looks_of, kmin=2):
    """per camera: list over k of (position error, spatial spread) after fusing the first k spots"""
    out = []
    for key in sorted(res):
        sc, bd, seed = key
        mt, outs = res[key]
        if cfg not in outs or sc in R.EXAM or "ll_all" not in outs[cfg]:
            continue
        o = outs[cfg]
        LL = o["ll_all"].astype(np.float32); grid = o["grid"].astype(np.float64); cams = mt["cams"]
        order = order_of(sc)
        for r in range(len(cams)):
            lt = np.zeros(len(grid), np.float64); tr = []
            for k, i in enumerate(order):
                m = np.sum([LL[r, j] for j in looks_of(sc, i)], axis=0)
                lt = lt + (m - m.max())
                if k + 1 >= kmin:
                    c = int(np.argmax(lt)); pt = grid[c]
                    p = np.exp(lt - lt.max()); p /= p.sum()
                    sp = float(np.sqrt(np.sum(p * np.sum((grid[:, :2] - pt[:2]) ** 2, axis=1))))
                    tr.append((float(np.linalg.norm(pt[:2] - cams[r][:2])), sp))
            out.append((sc, tr))
    return out


def policy(tr_all, thr, kmin=2, kmax=None):
    errs, used, kinds = [], [], []
    for sc, tr in tr_all:
        n = len(tr) if kmax is None else min(len(tr), kmax - kmin + 1)
        pick = n - 1
        for k in range(n):
            if tr[k][1] <= thr:
                pick = k; break
        errs.append(tr[pick][0]); used.append(pick + kmin); kinds.append(R.KIND.get(sc, "old"))
    return np.array(errs), np.array(used), np.array(kinds)


dev = seq(S.load(devp), lambda sc: [4, 6, 5, 7, 0, 2, 1, 3], S.dev_looks)      # opposite wall centres first, then corners
def real_order(sc):
    four = list(R.COMBO[sc]["four"]); pair = list(R.COMBO[sc]["pair"])
    rest = [i for i in range(len(R.SPOTS[sc])) if i not in four and R.SPOTS[sc][i][5] != "B"]
    return pair + [i for i in four if i not in pair] + rest
real = seq(S.load(realp), real_order, S.real_looks)
print("# measure more spots when unsure: stop when the fused map's spread falls under a threshold fixed on the old rooms   [simulation]")
print("rule                         | old rooms: within 0.5 m, mean spots | realistic rooms: within 0.5 m (lodging / restroom / locker), within 1 m, mean spots")
def line(name, thr, kmax=None):
    e, u, _ = policy(dev, thr, kmax=kmax); er, ur, kr = policy(real, thr, kmax=kmax)
    per = " / ".join(f"{100*np.mean(er[kr==k]<=0.5):5.1f}" for k in ("lodging", "restroom", "locker"))
    print(f"{name:28s} | {100*np.mean(e<=0.5):5.1f} %  {u.mean():4.2f}        | {100*np.mean(er<=0.5):5.1f} % ({per})  {100*np.mean(er<=1.0):5.1f} %  {ur.mean():4.2f}")
for k in (2, 3, 4, 5, 6, 7, 8):
    line(f"always {k} spots", -1.0, kmax=k)
sp_dev = np.array([t[1] for _, tr in dev for t in tr])
for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    thr = float(np.quantile(sp_dev, q))
    line(f"stop at spread <= {thr:.2f} m", thr)
    line(f"  .. at most 6 spots", thr, kmax=6)
