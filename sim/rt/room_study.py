"""숙소 방에서 패널로 숨긴 카메라의 위치를 얼마나 맞히나.
    python room_study.py measure [--sets room_s1 room_s2 room_s3] [--recv ura4x4p37.5h] [--tag ""] [--opt '{"imp": {...}}']
    python room_study.py table   [--tag ""]

재는 법은 교실과 같다 (rt_study 의 array_snap, array_score). 다른 점:
- 자리 대신 방의 겉면(벽·천장·욕실 벽·가구 위)에 20cm 간격으로 놓은 점들 가운데에서 고른다. 맞혔는지는 거리(m)로 본다
- 패널의 뒤쪽 반은 보지 못하므로, 패널 뒤에 있는 점에는 그 자리에서 잰 점수를 주지 않는다
- 여러 곳에서 잰 것을 겹치는 법 (--fuse):
    sum     점수를 그대로 더한다 (처음에 쓴 법. 견주려고 남긴다)
    center  곳마다 앞쪽 점들의 평균을 빼고 더한다. 뒤쪽 점은 0 (= 모른다)
    power   center 에 더해, 같은 자리에서 돌려 가며 잰 것들은 패널이 받은 세기가 가장 센 쪽보다 --drop dB 넘게 약한 쪽을 버린다. 기본값
            (카메라를 등진 쪽은 반사된 전파만 받아 엉뚱한 곳을 가리킨다. 등졌는지는 받은 세기로 안다)
- 패널의 자리와 기울기는 폰 AR 이 알려 준다고 본다 (자리 3cm, 좌우 2도, 기울기 1.5도 흩어짐)
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np                                      # noqa: E402
import rt_study as S                                    # noqa: E402
import scene_room as R                                  # noqa: E402  (장면을 만들지 않으면 sionna 를 쓰지 않는다)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "study")
FIVE = (0, 1, 2, 3, 4)
COMBOS = [("가운데에서 한 쪽만 (창 쪽)", (0,)), ("가운데에 서서 다섯 쪽으로 돌림", FIVE), ("다섯 쪽 + 창턱", FIVE + (6,)), ("다섯 쪽 + 현관 통로", FIVE + (10,)), ("다섯 쪽 + 창턱 + 현관 통로", FIVE + (6, 10)),
          ("책상 위 한 곳 (고정)", (5,)), ("책상 위 + 창턱 (고정 둘)", (5, 6)), ("창턱 + 현관문 안쪽 (고정 둘)", (6, 7)), ("책상 위 + 창턱 + 현관문 안쪽 (고정 셋)", (5, 6, 7)),
          ("욕실 문 앞에서 한 번", (8,)), ("욕실 안에서 한 번", (9,)), ("욕실 문 앞 + 욕실 안", (8, 9)), ("네 자리 모두 (가운데·창턱·현관 통로·욕실 안)", FIVE + (6, 10, 9))]
_G = {}


def _load(name):
    if name not in _G:
        d = np.load(os.path.join(HERE, "paths", name + ".npz")); _G[name] = {k: d[k] for k in d.files if k != "meta"}
    return _G[name]


def _init(X):
    _G["X"] = X


def _trial(args):
    """한 카메라를 모든 곳에서 잰다. 돌려준다: (카메라 번호, 곳마다의 점수 [S, N] float16, 곳마다 (직접파가 있나, 앞쪽에 있나, 방향 오차, 받은 세기 dB))"""
    seed, name, c, recv, opt = args
    D = _load(name); X = _G["X"]; rng = np.random.default_rng(seed); u = rng.normal(size=3); u /= np.linalg.norm(u)
    holo = recv.endswith("h"); nm = recv[:-1] if holo else recv; ns = len(D["spots"])
    sc = np.zeros((ns, len(X)), np.float32); info = np.full((ns, 4), np.nan, np.float32)
    for s in range(ns):
        r2 = np.random.default_rng([seed, s]); P = S.Paths(D, s, c, u); S.AIM = D["aims"][s].astype(float)
        if not len(P.tau):
            continue
        H, Rw, bore = S.array_snap(r2, P, nm, npk=opt.get("npk", 20), dur=opt.get("dur", 0.3), tv=0.0, snr=opt.get("snr", 25.0), holo=holo, mount="aim", imp=opt.get("imp"))
        spot, Rw, bore = S.pose_err(r2, P.spot.copy(), Rw, bore, opt.get("pos_sd", 0.03), opt.get("yaw_sd", 2.0), opt.get("tilt_sd", 1.5))
        v = S.array_score(H, Rw, bore, spot, X, opt.get("neff", 12.0)); d = X - spot; d /= np.linalg.norm(d, axis=1, keepdims=True); front = d @ bore
        v = np.where(front > 0.17, v, 0.0); sc[s] = v                                          # 패널 뒤쪽(과 옆 10도)은 모른다
        pw = 10 * np.log10(max(float(np.abs(H).mean()), 1e-30)) + (r2.normal(0, opt.get("pw_sd", 1.0)))            # 받은 세기 (기준 x 안테나의 크기라 세기와 같은 단위). 읽는 값의 흔들림 1 dB
        infront = float(P.c_true @ bore); info[s] = (P.los, infront, S.ang_err(d[int(np.argmax(v))], P.c_true)[2], pw)
    return c, sc.astype(np.float16), info


def measure(a):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True); opt = json.loads(a.opt); X = R.surface_grid(); _G["X"] = X; t0 = time.time()
    jobs = []; meta = []
    for j, nm in enumerate(a.sets):
        D = _load(nm)
        for c in range(len(D["cams"])):
            for k in range(a.rep):
                jobs.append((5000 + 1000 * j + 10 * c + k, nm, c, a.recv, opt)); meta.append((j, c))
    with mp.Pool(a.jobs, initializer=_init, initargs=(X,)) as pool:
        rows = pool.map(_trial, jobs, chunksize=4)
    cams = np.array([_load(a.sets[j])["cams"][c] for j, c in meta]); cat = np.array([_load(a.sets[j])["cat"][c] for j, c in meta])
    np.savez_compressed(os.path.join(OUT, "room_%s.npz" % (a.recv + (a.tag and "_" + a.tag))), sc=np.stack([r[1] for r in rows]), info=np.stack([r[2] for r in rows]), cams=cams, cat=cat, X=X.astype(np.float32),
                        cats=_load(a.sets[0])["cats"], names=_load(a.sets[0])["names"], spots=_load(a.sets[0])["spots"], opt=a.opt)
    print("%s %d번  후보 점 %d개  %.0f초" % (a.recv, len(rows), len(X), time.time() - t0), flush=True)


GROUPS = [FIVE]                                         # 같은 자리에서 돌려 가며 잰 묶음 (방 가운데의 다섯 쪽)
PW_DROP = 6.0


def fuse(sc, combo, pw=None, mode="power"):
    """곳들의 점수를 겹친 것 [N]. sc [S, N]: 0 = 그 곳에서는 모르는 점 (패널 뒤)"""
    use = list(combo)
    if mode == "power" and pw is not None:
        for g in GROUPS:
            gs = [s for s in use if s in g]
            if len(gs) > 1:
                top = np.nanmax(pw[gs]); use = [s for s in use if s not in gs or pw[s] >= top - PW_DROP]
    t = np.zeros(sc.shape[1], np.float32)
    for s in use:
        v = sc[s].astype(np.float32); fr = v != 0
        if mode == "sum" or not fr.any():
            t += v
        else:
            t += np.where(fr, v - v[fr].mean(), 0.0)
    return t


def estimate(sc, X, combo, k=3, sep=0.6, pw=None, mode="power"):
    """가장 높은 점. 돌려준다: (1등 점, 서로 sep 넘게 떨어진 위쪽 k 개 점)"""
    t = fuse(sc, combo, pw, mode); order = np.argsort(-t); top = [order[0]]
    for i in order[1:400]:
        if len(top) >= k: break
        if all(np.linalg.norm(X[i] - X[j]) > sep for j in top): top.append(i)
    return X[order[0]], X[top]


def table(a):
    global PW_DROP
    PW_DROP = a.drop
    f = os.path.join(OUT, "room_%s.npz" % (a.recv + (a.tag and "_" + a.tag))); Z = np.load(f); X = Z["X"].astype(float); cams = Z["cams"]; cat = Z["cat"]; cats = [str(c) for c in Z["cats"]]; names = [str(n) for n in Z["names"]]
    SC = Z["sc"]; PW = Z["info"][:, :, 3] if Z["info"].shape[2] > 3 else np.zeros(Z["info"].shape[:2]); n = len(cams); bath = np.array([c.startswith("욕실") for c in cats])[cat]; out = {"n": int(n), "candidates": int(len(X)), "fuse": a.fuse, "combos": {}, "bearing": {}, "by_cat": {}}
    print("== %s  카메라 %d번, 후보 점 %d개, 겹치는 법 %s%s. 오차 = 고른 점과 참 자리의 거리 (m)" % (a.recv, n, len(X), a.fuse, " (%g dB)" % a.drop if a.fuse == "power" else ""))
    print("%-34s %-6s %5s %5s %6s %6s %6s %7s" % ("재는 곳", "대상", "가운데", "80%", "30cm안", "50cm안", "1m안", "후보3 50cm"))
    E = {}
    for nm, combo in COMBOS:
        e = np.zeros(n); e3 = np.zeros(n); est = np.zeros((n, 3))
        for i in range(n):
            p1, p3 = estimate(SC[i], X, combo, pw=PW[i], mode=a.fuse); e[i] = np.linalg.norm(p1 - cams[i]); e3[i] = np.linalg.norm(p3 - cams[i], axis=1).min(); est[i] = p1
        E[nm] = (e, est)
        has_b = any(k in combo for k in (8, 9)); only_b = all(k in (8, 9) for k in combo)
        for lab, m in (("방", ~bath), ("욕실", bath)):
            if (lab == "욕실" and not has_b) or (lab == "방" and only_b):
                continue
            r = dict(median=float(np.median(e[m])), p80=float(np.percentile(e[m], 80)), in30=float((e[m] <= 0.3).mean()), in50=float((e[m] <= 0.5).mean()), in100=float((e[m] <= 1.0).mean()), top3_50=float((e3[m] <= 0.5).mean()))
            out["combos"]["%s | %s" % (nm, lab)] = r
            print("%-34s %-6s %5.2f %5.2f %5.0f%% %5.0f%% %5.0f%% %6.0f%%" % (nm[:34], lab, r["median"], r["p80"], 100 * r["in30"], 100 * r["in50"], 100 * r["in100"], 100 * r["top3_50"]))
    sfx = a.recv + (a.tag and "_" + a.tag) + ("" if a.fuse == "power" else "_" + a.fuse) + ("" if a.drop == 6.0 else "_d%g" % a.drop)
    info = Z["info"]
    if not a.brief:
        print("\n-- 한 곳에서 잰 방향의 오차 (도, 가운데값): 카메라가 패널 앞쪽 60도 안에 있을 때")
    for s, nm in enumerate(names):
        fr = info[:, s, 1] > 0.5; los = info[:, s, 0] > 0.5; be = info[:, s, 2]
        a1 = be[fr & los]; a2 = be[fr & ~los]
        out["bearing"][nm] = dict(los=float(np.median(a1)) if len(a1) else None, nlos=float(np.median(a2)) if len(a2) else None, los_frac=float(los[fr].mean()) if fr.any() else None, n=int(fr.sum()))
        if not a.brief:
            print("%-24s 직접 보일 때 %5.1f (%3d번)   가려졌을 때 %5.1f (%3d번)" % (nm, np.median(a1) if len(a1) else np.nan, len(a1), np.median(a2) if len(a2) else np.nan, len(a2)))
    print("\n-- 숨긴 곳마다 (가운데값 m / 50cm 안 %)")
    show = ["가운데에 서서 다섯 쪽으로 돌림", "다섯 쪽 + 창턱 + 현관 통로", "책상 위 + 창턱 (고정 둘)", "네 자리 모두 (가운데·창턱·현관 통로·욕실 안)"]
    print("%-34s " % "숨긴 곳" + " | ".join("%-20s" % s[:20] for s in show))
    for k, cn in enumerate(cats):
        m = cat == k; row = []
        for s in show:
            e = E[s][0][m]; row.append("%4.2f / %3.0f%%" % (np.median(e), 100 * (e <= 0.5).mean())); out["by_cat"].setdefault(cn, {})[s] = dict(median=float(np.median(e)), in50=float((e <= 0.5).mean()))
        print("%-34s " % cn[:34] + " | ".join("%-20s" % r for r in row))
    json.dump(out, open(os.path.join(OUT, "room_%s.json" % sfx), "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(OUT, "room_est_%s.npz" % sfx), cams=cams, cat=cat, **{"e%d" % i: E[nm][0] for i, (nm, _) in enumerate(COMBOS)}, **{"p%d" % i: E[nm][1] for i, (nm, _) in enumerate(COMBOS)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--sets", nargs="+", default=["room_s1", "room_s2", "room_s3"]); ap.add_argument("--recv", default="ura4x4p37.5h")
    ap.add_argument("--fuse", default="power", choices=["sum", "center", "power"]); ap.add_argument("--drop", type=float, default=6.0); ap.add_argument("--brief", action="store_true")
    ap.add_argument("--rep", type=int, default=2); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--opt", default="{}"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); {"measure": measure, "table": table}[a.cmd](a)
