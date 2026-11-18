"""숙소 방 계산(room_study.py)과 같은 전파 길 위에서 돌판 수신기를 잰다 (2026-11-18).
돌판 모형(rt_eval.py)은 옆에 둔 receiver-legacy/sim/rt 에서 불러온다.
    python room_plate.py measure [--plan tilt18] [--rep 2] [--jobs 11]
    python room_plate.py table   [--plan tilt18]
돌판: 안테나 하나 + 도는 알루미늄 판, 곡선 전체 맞추기(rt_study 의 plate 와 같다: tm 점수, 편파까지 맞춤).
돌판은 모든 쪽을 보므로 '방 가운데'는 한 번만 잰다 (자리 0). 후보 점과 오차 정의는 패널 계산과 같다.
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = HERE
OLD = os.path.abspath(os.path.join(HERE, "..", "..", "..", "receiver-legacy", "sim", "rt"))
sys.path.append(OLD)
import rt_eval as E                                   # noqa: E402
from rt_eval import PLANS                             # noqa: E402
import locate as K                                    # noqa: E402
import scene_room as R                                # noqa: E402
OUT = os.path.join(HERE, "study")
COMBOS = [("방 가운데 한 곳", (0,)), ("가운데 + 창턱", (0, 6)), ("가운데 + 현관 통로", (0, 10)), ("가운데 + 창턱 + 현관 통로", (0, 6, 10)),
          ("책상 위 한 곳", (5,)), ("책상 위 + 창턱", (5, 6)), ("책상 위 + 창턱 + 현관문 안쪽", (5, 6, 7)),
          ("욕실 안에서 한 번", (9,)), ("네 자리 (가운데·창턱·현관 통로·욕실 안)", (0, 6, 10, 9))]
USE = sorted({s for _, c in COMBOS for s in c})
_G = {}


def grid_dirs(step=3.0):
    v = []
    for el in np.arange(-87, 90, step):
        n = max(1, int(round(360 * np.cos(np.radians(el)) / step)))
        for az in np.arange(n) * 360.0 / n:
            v.append((np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))))
    return np.array(v)


def _init(X, plan):
    _G["pat"] = E.Pattern(); _G["tm"] = E.TM(_G["pat"], 160.0); _G["X"] = X; _G["plan"] = plan; _G["G"] = grid_dirs(); _G["D"] = {}


def _load(name):
    if name not in _G.setdefault("D", {}):
        d = np.load(os.path.join(NEW, "paths", name + ".npz")); _G["D"][name] = {k: d[k] for k in d.files if k != "meta"}
    return _G["D"][name]


def _trial(args):
    seed, name, c = args
    D = _load(name); X = _G["X"]; frames = PLANS[_G["plan"]]; rng = np.random.default_rng(seed); u = rng.normal(size=3); u /= np.linalg.norm(u)
    ns = len(D["spots"]); sc = np.zeros((ns, len(X)), np.float32); info = np.full((ns, 3), np.nan, np.float32)
    for s in USE:
        r2 = np.random.default_rng([seed, s])
        w = E.RTWorld(r2, D, s, c, frames, _G["pat"], u, rate=100.0, tv=0.0, snr=25.0)
        if not len(w.tau):
            continue
        pc = [w.scan(y, t)[1] for y, t in frames]
        pose = (w.base + r2.normal(0, 0.05, 3) * np.array([1, 1, 0.5]), w.head + r2.normal(0, 3.0))
        _, b = _G["tm"].score(K.to_dev(X, pose), [pc], frames); sc[s] = b[0]
        spot = D["spots"][s].astype(float); Xg = spot + 6.0 * _G["G"]; _, bg = _G["tm"].score(K.to_dev(Xg, pose), [pc], frames)
        d = _G["G"][int(np.argmax(bg[0]))]; ang = float(np.degrees(np.arccos(np.clip(d @ w.c_true, -1, 1))))
        info[s] = (w.los, ang, w.k_db)
    return c, sc.astype(np.float16), info


def measure(a):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True); X = R.surface_grid(); t0 = time.time(); jobs = []; meta = []
    for j, nm in enumerate(a.sets):
        D = _load(nm)
        for c in range(len(D["cams"]))[: a.ncam or None]:
            for k in range(a.rep):
                jobs.append((5000 + 1000 * j + 10 * c + k, nm, c)); meta.append((j, c))
    with mp.Pool(a.jobs, initializer=_init, initargs=(X, a.plan)) as pool:
        rows = pool.map(_trial, jobs, chunksize=2)
    cams = np.array([_load(a.sets[j])["cams"][c] for j, c in meta]); cat = np.array([_load(a.sets[j])["cat"][c] for j, c in meta])
    np.savez_compressed(os.path.join(OUT, "room_plate_%s.npz" % a.plan), sc=np.stack([r[1] for r in rows]), info=np.stack([r[2] for r in rows]), cams=cams, cat=cat, X=X.astype(np.float32),
                        cats=_load(a.sets[0])["cats"], names=_load(a.sets[0])["names"])
    print("plate %s %d번 후보 %d개 %.0f초" % (a.plan, len(rows), len(X), time.time() - t0), flush=True)


def table(a):
    Z = np.load(os.path.join(OUT, "room_plate_%s.npz" % a.plan)); X = Z["X"].astype(float); SC = Z["sc"].astype(np.float32); cams = Z["cams"]; cat = Z["cat"]
    cats = [str(c) for c in Z["cats"]]; names = [str(n) for n in Z["names"]]; bath = np.array([c.startswith("욕실") for c in cats])[cat]; n = len(cams); out = {"n": int(n), "plan": a.plan, "combos": {}, "bearing": {}, "by_cat": {}}
    print("== 돌판 (%s), 카메라 %d번, 후보 점 %d개. 오차 = 고른 점과 참 자리의 거리 (m)" % (a.plan, n, len(X)))
    print("%-34s %-4s %5s %5s %6s %6s %6s %7s" % ("재는 곳", "대상", "가운데", "80%", "30cm안", "50cm안", "1m안", "후보3 50cm"))
    E1 = {}
    for nm, combo in COMBOS:
        e = np.zeros(n); e3 = np.zeros(n)
        for i in range(n):
            t = np.zeros(len(X), np.float32)
            for s in combo:
                v = SC[i, s]; t += v - v.mean()
            o = np.argsort(-t); top = [o[0]]
            for q in o[1:400]:
                if len(top) >= 3: break
                if all(np.linalg.norm(X[q] - X[p]) > 0.6 for p in top): top.append(q)
            e[i] = np.linalg.norm(X[o[0]] - cams[i]); e3[i] = np.linalg.norm(X[top] - cams[i], axis=1).min()
        E1[nm] = e
        only_b = all(k == 9 for k in combo)
        for lab, m in (("방", ~bath), ("욕실", bath)):
            if (lab == "욕실" and 9 not in combo) or (lab == "방" and only_b) or not m.any():
                continue
            r = dict(median=float(np.median(e[m])), p80=float(np.percentile(e[m], 80)), in30=float((e[m] <= .3).mean()), in50=float((e[m] <= .5).mean()), in100=float((e[m] <= 1).mean()), top3_50=float((e3[m] <= .5).mean()))
            out["combos"]["%s | %s" % (nm, lab)] = r
            print("%-34s %-4s %5.2f %5.2f %5.0f%% %5.0f%% %5.0f%% %6.0f%%" % (nm[:34], lab, r["median"], r["p80"], 100 * r["in30"], 100 * r["in50"], 100 * r["in100"], 100 * r["top3_50"]))
    print("\n-- 한 곳에서 잰 방향 오차 (도, 가운데값 / 30도 넘게 틀린 비율)")
    info = Z["info"]
    for s in USE:
        los = info[:, s, 0] > 0.5; be = info[:, s, 1]; ok = np.isfinite(be)
        r = dict(los=float(np.median(be[ok & los])) if (ok & los).any() else None, nlos=float(np.median(be[ok & ~los])) if (ok & ~los).any() else None,
                 over30=float((be[ok] > 30).mean()), los_frac=float(los[ok].mean()))
        out["bearing"][names[s]] = r
        print("%-24s 직접 %5s  가려짐 %5s  30도 넘게 %3.0f%%  직접 비율 %3.0f%%" % (names[s], "%.1f" % r["los"] if r["los"] is not None else "-", "%.1f" % r["nlos"] if r["nlos"] is not None else "-", 100 * r["over30"], 100 * r["los_frac"]))
    for k, cn in enumerate(cats):
        m = cat == k
        if not m.any():
            continue
        out["by_cat"][cn] = {s: dict(median=float(np.median(E1[s][m])), in50=float((E1[s][m] <= .5).mean())) for s in ("방 가운데 한 곳", "가운데 + 창턱 + 현관 통로")}
    json.dump(out, open(os.path.join(OUT, "room_plate_%s.json" % a.plan), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--sets", nargs="+", default=["room_s1", "room_s2", "room_s3"]); ap.add_argument("--plan", default="tilt18")
    ap.add_argument("--rep", type=int, default=2); ap.add_argument("--ncam", type=int, default=0); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(); {"measure": measure, "table": table}[a.cmd](a)
