"""추정 구간(부피) 계산: 카메라가 겉면에 있다고 가정하지 않고, 두 곳 이상에서 잰 방향선이 겹치는 부피로 자리를 찾는다 (2026-11-19).
패널과 돌판을 같은 전파 길, 같은 카메라, 같은 겹치기·같은 지표로 견준다.

    python vol_study.py paths   --scene room|lec              전파 길 묶음 (GPU). exam 은 배치 계산의 kr40_cv_s* 를 그대로 쓴다
    python vol_study.py measure --scene room|lec|exam --dev panel|plate [--jobs 11]
    python vol_study.py table   --scene room|lec|exam

장면
  room  숙소 방 (scene_room). 겉면 아홉 곳 + 겉면에 붙지 않은 네 곳 (책상 위 물건, 협탁, 걸어 둔 가방, 바닥 가방)
  lec   빈 교실(kr40, 7.5 x 9 m, 철판 칠판·철제 사물함)에 숨긴 몰카. 손에 들고 세 곳(앞·가운데·뒤)에서 잰다
  exam  시험 중 교실(사람 40명). 책상 위, 가슴, 안경 자리의 초소형 카메라. 벽에 건 두 곳에서 잰다

읽는 법 (두 장치 같다)
  한 곳의 점수는 방향만의 함수다 (패널은 평면파 조향, 돌판은 판의 회절 곡선). 2도 간격 방향 격자에서 점수를 내고, 방 안 칸(숙소 10 cm, 교실 15 cm)마다
  그 칸 쪽 방향의 점수를 준다. 곳마다 평균을 빼서 더한다 (패널 뒤쪽은 모름 = 0). 손에 든 패널의 다섯 쪽은 room_study 와 같이 센 쪽만 쓴다.
  추정 구간 = 겹친 점수가 (최고 - 델타) 위인 칸들. 델타는 장치·조합마다 '참 자리가 구간 안에 드는 비율 80%' 가 되게 맞춘다.
  지표: 구간의 길이 (가장 긴 축의 폭), 부피, 최고점의 오차, 50 cm 안 비율, 후보 셋 가운데 하나가 50 cm 안.
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "study"); PATHS = os.path.join(HERE, "paths")
OLD = os.path.abspath(os.path.join(HERE, "..", "..", "..", "receiver-legacy", "sim", "rt"))
Q = np.geomspace(0.002, 0.8, 24)                                         # 델타 / (최고 - 가운데값)
STEP = float(os.environ.get("VOL_STEP", "2"))                         # 방향 격자 간격 (도)
TAG = ("" if STEP == 2.0 else "_d%g" % STEP) + ("" if os.environ.get("VOL_ROTMODE", "full") == "full" else "_" + os.environ["VOL_ROTMODE"])

# ---------- 장면마다: 길 묶음, 칸, 곳, 조합 ----------
LEC_PLACES = [("앞", (2.5, 1.6, 1.3)), ("가운데", (3.375, 4.85, 1.3)), ("뒤", (3.75, 8.2, 1.3))]


def lec_spots():
    """손에 든 패널: 곳마다 다섯 쪽 (칠판, 뒷벽, 창, 복도, 천장). 돌판은 곳마다 첫 줄 하나만 쓴다"""
    sp, aims, names = [], [], []
    for nm, (x, y, z) in LEC_PLACES:
        for dn, aim in (("칠판 쪽", (x, 0.0, 1.4)), ("뒷벽 쪽", (x, 9.0, 1.2)), ("창 쪽", (0.0, y, 1.4)), ("복도 쪽", (7.5, y, 1.4)), ("천장 쪽", (x, y + 0.3, 2.7))):
            sp.append((x, y, z)); aims.append(aim); names.append("%s에서 %s" % (nm, dn))
    return sp, aims, names


def lec_hide(rng, per):
    W, D = 7.5, 9.0
    from scene_kr import SEATS

    def board(r):
        if r.random() < 0.6: return (r.uniform(1.4, 6.1), r.uniform(0.05, 0.12), r.uniform(0.88, 0.96))
        return (r.choice([1.30, 6.20]), 0.06, r.uniform(0.95, 2.05))

    def back(r):
        if r.random() < 0.5: return (r.uniform(0.5, 7.0), D - 0.04, r.uniform(1.4, 2.3))
        return (r.uniform(0.5, 7.0), r.uniform(8.55, 8.95), r.uniform(1.25, 1.40))

    def desk(r):
        x, y = SEATS[r.integers(len(SEATS))]
        if r.random() < 0.6: return (x + r.uniform(-0.25, 0.25), y + r.uniform(-0.15, 0.15), r.uniform(0.78, 1.0))
        return (x + r.uniform(-0.35, 0.35), y + 0.30, r.uniform(0.15, 0.45))
    H = {"칠판 위 시계·스피커": lambda r: (r.uniform(1.5, 6.0), 0.05, r.uniform(2.2, 2.55)),
         "칠판 둘레 (분필받이·테두리)": board,
         "빔프로젝터 (천장)": lambda r: (r.uniform(3.3, 4.2), r.uniform(2.8, 4.0), r.uniform(2.45, 2.62)),
         "천장 화재감지기": lambda r: (r.uniform(0.8, 6.7), r.uniform(1.0, 8.0), 2.66),
         "복도 쪽 벽 높은 곳 (스피커·에어컨)": lambda r: (W - 0.05, r.uniform(1.5, 7.5), r.uniform(2.15, 2.6)),
         "창 위 커튼 상자": lambda r: (r.uniform(0.04, 0.15), r.uniform(0.6, 8.4), r.uniform(2.42, 2.62)),
         "뒷벽 게시판·사물함 위": back,
         "교탁 위 물건 (공중)": lambda r: (r.uniform(3.4, 4.1), r.uniform(0.95, 1.35), r.uniform(1.05, 1.30)),
         "책상 위·가방 속 (공중)": desk}
    P = [H[c](rng) for c in H for _ in range(per)]
    return np.array(P, float), np.repeat(np.arange(len(H)), per), list(H)


def grid(scene):
    if scene == "room":
        st = 0.10; W, D, C = 3.6, 5.4, 2.4
    else:
        st = 0.15; W, D, C = 7.5, 9.0, 2.7
    xs = np.arange(st / 2, W, st); ys = np.arange(st / 2, D, st); zs = np.arange(st / 2, C, st)
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return X, st


FIVE = (0, 1, 2, 3, 4)
SC = {
    "room": dict(sets=["room_v_s1", "room_v_s2", "room_v_s3"], k=1,
                 panel=[0, 1, 2, 3, 4, 6, 8, 9, 10], plate=[0, 6, 8, 9, 10], groups=[FIVE],
                 combos=[("한 곳 (방 가운데)", {"panel": FIVE, "plate": (0,)}, "방"),
                         ("두 곳 (가운데 + 창턱)", {"panel": FIVE + (6,), "plate": (0, 6)}, "방"),
                         ("세 곳 (가운데 + 창턱 + 현관 통로)", {"panel": FIVE + (6, 10), "plate": (0, 6, 10)}, "방"),
                         ("욕실 한 곳 (욕실 안)", {"panel": (9,), "plate": (9,)}, "욕실"),
                         ("욕실 두 곳 (문 앞 + 욕실 안)", {"panel": (8, 9), "plate": (8, 9)}, "욕실"),
                         ("네 곳 (가운데·창턱·현관 통로·욕실 안)", {"panel": FIVE + (6, 10, 9), "plate": (0, 6, 10, 9)}, "모두")]),
    "lec": dict(sets=["lecmk_s1", "lecmk_s2", "lecmk_s3"], k=1,
                panel=list(range(15)), plate=[0, 5, 10], groups=[(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14)],
                combos=[("한 곳 (가운데)", {"panel": (5, 6, 7, 8, 9), "plate": (5,)}, "모두"),
                        ("두 곳 (앞 + 뒤)", {"panel": tuple(range(5)) + tuple(range(10, 15)), "plate": (0, 10)}, "모두"),
                        ("두 곳 (가운데 + 뒤)", {"panel": tuple(range(5, 15)), "plate": (5, 10)}, "모두"),
                        ("세 곳 (앞 + 가운데 + 뒤)", {"panel": tuple(range(15)), "plate": (0, 5, 10)}, "모두")]),
    "exam": dict(sets=["kr40_cv_s%d" % i for i in range(1, 9)], k=8,
                 panel=[2, 3, 4, 5], plate=[2, 3, 4, 5], groups=[],
                 combos=[("창 2.5 m + 복도 2.5 m (데모 배치)", {"panel": (5, 4), "plate": (5, 4)}, "모두"),
                         ("두 옆벽 가운데 (4.85 m)", {"panel": (2, 3), "plate": (2, 3)}, "모두"),
                         ("넷 모두", {"panel": (2, 3, 4, 5), "plate": (2, 3, 4, 5)}, "모두")]),
}
KS = (1, 4, 8)
ROT = 8                                                                 # panelrot: 받침을 돌려 45도마다 + 천장 쪽 하나


ROTMODE = os.environ.get("VOL_ROTMODE", "full")                        # full: 수평 8 + 천장 1, flat: 받침만 돌림(수평 8, 기울임 0), tilt<도>: 받침만 돌림, 위로 기울임


def rot_aims(spot):
    x, y, z = spot
    if ROTMODE.startswith("tilt"):
        t = np.tan(np.radians(float(ROTMODE[4:])))
        A = [(x + 2.0 * np.cos(np.radians(a)), y + 2.0 * np.sin(np.radians(a)), z + 2.0 * t) for a in np.arange(ROT) * 360.0 / ROT]
        return A + [A[0]]
    if ROTMODE == "flat":
        A = [(x + 2.0 * np.cos(np.radians(a)), y + 2.0 * np.sin(np.radians(a)), z) for a in np.arange(ROT) * 360.0 / ROT]
        return A + [A[0]]
    A = [(x + 2.0 * np.cos(np.radians(a)), y + 2.0 * np.sin(np.radians(a)), max(z - 0.3, 0.5)) for a in np.arange(ROT) * 360.0 / ROT]
    return A + [(x + 0.01, y + 0.3, z + 3.0)]                                                          # exam: 사진 몇 장을 모으나 (패널만. 돌판은 한 번에 110초 이어진 신호가 있어야 해서 1)


# ---------- 방향 격자 ----------
def dir_grid(step=2.0):
    els = np.arange(-89.0, 90.0, step); ns = [max(1, int(round(360 * np.cos(np.radians(e)) / step))) for e in els]
    off = np.concatenate([[0], np.cumsum(ns)[:-1]]); G = []
    for e, n in zip(els, ns):
        az = np.arange(n) * 360.0 / n; ce = np.cos(np.radians(e))
        G += [(ce * np.cos(np.radians(a)), ce * np.sin(np.radians(a)), np.sin(np.radians(e))) for a in az]
    return np.array(G), els, np.array(ns), off, step


def dir_index(V, DG):
    """단위 벡터 V [N,3] -> 가장 가까운 격자 방향 번호"""
    _, els, ns, off, step = DG
    el = np.degrees(np.arcsin(np.clip(V[:, 2], -1, 1))); k = np.clip(np.round((el - els[0]) / step).astype(int), 0, len(els) - 1)
    az = np.degrees(np.arctan2(V[:, 1], V[:, 0])) % 360.0; n = ns[k]; j = np.round(az / (360.0 / n)).astype(int) % n
    return off[k] + j


# ---------- 일꾼 ----------
_G = {}


def _init(scene, dev):
    _G.update(scene=scene, dev=dev, D={}, idx={}); X, st = grid(scene); _G["X"] = X; _G["st"] = st; _G["DG"] = dir_grid(STEP)
    if dev == "plate":
        sys.path.append(OLD)
        import rt_eval as E
        import locate as K
        _G["E"] = E; _G["K"] = K; _G["pat"] = E.Pattern(); _G["tm"] = E.TM(_G["pat"], 160.0); _G["frames"] = E.PLANS["tilt18"]


def _load(name):
    if name not in _G["D"]:
        d = np.load(os.path.join(PATHS, name + ".npz")); _G["D"][name] = {k: d[k] for k in d.files if k not in ("meta",)}
    return _G["D"][name]


def _vox_index(spot):
    key = tuple(np.round(spot, 3))
    if key not in _G["idx"]:
        v = _G["X"] - np.asarray(spot, float); v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9); _G["idx"][key] = dir_index(v, _G["DG"])
    return _G["idx"][key]


def _panel_dir(D, s, c, u, r2, aim=None):
    import rt_study as S
    if aim is not None:
        S.AIM = np.asarray(aim, float)
    elif "aims" in D:
        S.AIM = D["aims"][s].astype(float)
    else:
        S.AIM = np.array([3.75, 4.85, 0.9])
    P = S.Paths(D, s, c, u)
    if not len(P.tau):
        return None, -99.0, P
    H, Rw, bore = S.array_snap(r2, P, "ura4x4p37.5", npk=20, dur=0.3, tv=0.0, snr=25.0, holo=True, mount="aim")
    spot, Rw, bore = S.pose_err(r2, P.spot.copy(), Rw, bore, 0.03, 2.0, 1.5)
    G = _G["DG"][0]; v = S.array_score(H, Rw, bore, spot, spot + 6.0 * G, 12.0)
    v = np.where(G @ bore > 0.17, v, 0.0)
    pw = 10 * np.log10(max(float(np.abs(H).mean()), 1e-30)) + r2.normal(0, 1.0)
    return v.astype(np.float32), pw, P


def _plate_dir(D, s, c, u, r2):
    E, K = _G["E"], _G["K"]; frames = _G["frames"]
    w = E.RTWorld(r2, D, s, c, frames, _G["pat"], u, rate=100.0, tv=0.0, snr=25.0)
    if not len(w.tau):
        return None, -99.0, None
    pc = [w.scan(y, t)[1] for y, t in frames]
    pose = (w.base + r2.normal(0, 0.05, 3) * np.array([1, 1, 0.5]), w.head + r2.normal(0, 3.0))
    spot = D["spots"][s].astype(float); G = _G["DG"][0]
    _, b = _G["tm"].score(K.to_dev(spot + 6.0 * G, pose), [pc], frames)
    return np.asarray(b[0], np.float32), 0.0, None


def _fuse(vox, pw, combo, groups):
    use = list(combo)
    for g in groups:
        gs = [s for s in use if s in g and s in vox]
        if len(gs) > 1:
            top = max(pw[s] for s in gs); use = [s for s in use if s not in gs or pw[s] >= top - 6.0]
    t = np.zeros(len(_G["X"]), np.float32)
    for s in use:
        if s not in vox:
            continue
        v = vox[s]; fr = v != 0
        if fr.any():
            t += np.where(fr, v - v[fr].mean(), 0.0)
    return t


def _metrics(t, cam, seats=None, seat_true=None):
    X = _G["X"]; o = np.argsort(-t); tmax = float(t[o[0]]); spread = max(tmax - float(np.median(t)), 1e-6)
    it = int(np.argmin(np.linalg.norm(X - cam, axis=1))); gap = (tmax - float(t[it])) / spread
    e1 = float(np.linalg.norm(X[o[0]] - cam)); top = [o[0]]
    for i in o[1:3000]:
        if len(top) >= 3: break
        if all(np.linalg.norm(X[i] - X[j]) > 0.6 for j in top): top.append(i)
    e3 = float(min(np.linalg.norm(X[j] - cam) for j in top))
    ext = np.zeros(len(Q), np.float32); vol = np.zeros(len(Q), np.float32); rng = np.random.default_rng(0)
    for k, q in enumerate(Q):
        m = t >= tmax - q * spread; n = int(m.sum()); vol[k] = n * _G["st"] ** 3
        if n < 2:
            ext[k] = _G["st"]; continue
        P = X[m]
        if n > 2500: P = P[rng.choice(n, 2500, replace=False)]
        P = P - P.mean(0); w, V = np.linalg.eigh(P.T @ P); a = P @ V[:, -1]; ext[k] = float(a.max() - a.min()) + _G["st"]
    r = dict(gap=gap, e1=e1, e3=e3, ext=ext, vol=vol)
    if seats is not None:                                                # 시험장: 칸 -> 가까운 자리 (xy 0.8 m 안)
        sl = []
        for i in o[:6000]:
            d = np.linalg.norm(seats - X[i, :2], axis=1); j = int(np.argmin(d))
            if d[j] < 0.8 and j not in sl: sl.append(j)
            if len(sl) >= 3: break
        r["seat1"] = bool(sl and sl[0] == seat_true); r["seat3"] = seat_true in sl
        sx, sy = seats[seat_true]; r["seat_nb"] = bool(sl and np.hypot(*(seats[sl[0]] - seats[seat_true])) < 1.2)
    return r


def _trial(args):
    scene, dev, c, rep = args; cfg = SC[scene]; spots = cfg["plate" if dev == "panelrot" else dev]; sets = cfg["sets"][: (cfg["k"] if dev.startswith("panel") else 1)] if scene == "exam" else [cfg["sets"][c[0]]]
    cam_i = c[1] if scene != "exam" else c; seed = 7000 + 97 * (cam_i if scene == "exam" else c[0] * 1000 + c[1]) + rep
    rng = np.random.default_rng(seed); u = rng.normal(size=3); u /= np.linalg.norm(u)
    per_set = []
    for name in sets:
        D = _load(name); vox = {}; pw = {}
        for s in spots:
            if dev == "panelrot":
                sp = D["spots"][s].astype(float)
                for k, aim in enumerate(rot_aims(sp)):
                    r2 = np.random.default_rng([seed, s, k, len(per_set)])
                    v, p, _ = _panel_dir(D, s, cam_i, u, r2, aim)
                    if v is None: continue
                    vox[s * 100 + k] = v[_vox_index(sp)]; pw[s * 100 + k] = p
                continue
            r2 = np.random.default_rng([seed, s, len(per_set)])
            v, p, _ = (_panel_dir if dev == "panel" else _plate_dir)(D, s, cam_i, u, r2)
            if v is None: continue
            vox[s] = v[_vox_index(D["spots"][s].astype(float))]; pw[s] = p
        per_set.append((vox, pw))
    D0 = _load(sets[0]); cam = D0["cams"][cam_i].astype(float); out = {}
    seats = None; seat_true = None
    if scene == "exam":
        seats = D0["seats"] + np.array([0.0, 0.225]); seat_true = cam_i // 4
    for nm, combos, _ in cfg["combos"]:
        if dev == "panelrot":
            combo = tuple(s * 100 + k for s in combos["plate"] for k in range(ROT + 1)); groups = [tuple(s * 100 + k for k in range(ROT + 1)) for s in combos["plate"]]
        else:
            combo = combos[dev]; groups = cfg["groups"]
        ks = KS if (scene == "exam" and dev.startswith("panel")) else (1,)
        for K_ in ks:
            t = sum(_fuse(vx, p, combo, groups) for vx, p in per_set[:K_])
            out[(nm, K_)] = _metrics(t, cam, seats, seat_true)
    return out


def _jobs(scene, dev, rep):
    cfg = SC[scene]
    if scene == "exam":
        D = np.load(os.path.join(PATHS, cfg["sets"][0] + ".npz")); return [(scene, dev, c, rep) for c in range(len(D["cams"]))]
    J = []
    for j, nm in enumerate(cfg["sets"]):
        D = np.load(os.path.join(PATHS, nm + ".npz"))
        J += [(scene, dev, (j, c), rep) for c in range(len(D["cams"]))]
    return J


def measure(a):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); jobs = []
    for r in range(a.rep):
        jobs += _jobs(a.scene, a.dev, r)
    if a.ncam:
        jobs = jobs[:: max(1, len(jobs) // a.ncam)]
    with mp.Pool(a.jobs, initializer=_init, initargs=(a.scene, a.dev)) as pool:
        rows = pool.map(_trial, jobs, chunksize=1)
    cfg = SC[a.scene]; keys = list(rows[0].keys()); res = {}
    for k in keys:
        res["%s|%d" % k] = {f: np.array([r[k][f] for r in rows]) for f in rows[0][k]}
    if a.scene == "exam":
        cam = np.array([np.load(os.path.join(PATHS, cfg["sets"][0] + ".npz"))["cams"][j[2]] for j in jobs]); cat = np.array([j[2] % 4 for j in jobs]); cats = ["책상 위 왼쪽", "책상 위 오른쪽", "가슴", "안경"]
    else:
        cam = np.array([np.load(os.path.join(PATHS, cfg["sets"][j[2][0]] + ".npz"))["cams"][j[2][1]] for j in jobs])
        cat = np.array([np.load(os.path.join(PATHS, cfg["sets"][j[2][0]] + ".npz"))["cat"][j[2][1]] for j in jobs])
        cats = [str(x) for x in np.load(os.path.join(PATHS, cfg["sets"][0] + ".npz"))["cats"]]
    np.savez_compressed(os.path.join(OUT, "vol_%s_%s%s.npz" % (a.scene, a.dev, TAG)), cams=cam, cat=cat, cats=np.array(cats), Q=Q,
                        **{"%s__%s" % (k, f): v for k, d in res.items() for f, v in d.items()})
    print("%s %s %d번 %.0f초" % (a.scene, a.dev, len(rows), time.time() - t0), flush=True)


def _bath_mask(scene, cats, cat):
    if scene != "room":
        return np.ones(len(cat), bool), np.zeros(len(cat), bool)
    b = np.array([c.startswith("욕실") for c in cats])[cat]; return ~b, b


def table(a):
    cfg = SC[a.scene]; out = {"scene": a.scene, "rows": []}
    print("== %s. 구간 = 참 자리가 80%% 들도록 맞춘 추정 구간. 길이 = 가장 긴 축의 폭 (m), 부피 (리터)" % a.scene)
    hdr = "%-34s %-5s %-4s %6s %8s %8s %6s %7s %7s" % ("재는 곳", "장치", "대상", "사진", "구간길이", "부피L", "최고점", "50cm안", "후보3")
    if a.scene == "exam":
        hdr += " %6s %6s %6s" % ("자리", "옆자리", "3자리")
    print(hdr)
    for dev in ("panel", "panelrot", "plate"):
        f = os.path.join(OUT, "vol_%s_%s%s.npz" % (a.scene, dev, TAG))
        if not os.path.exists(f):
            continue
        Z = np.load(f); cats = [str(c) for c in Z["cats"]]; cat = Z["cat"]; room_m, bath_m = _bath_mask(a.scene, cats, cat)
        for nm, _, tgt in cfg["combos"]:
            for K_ in (KS if (a.scene == "exam" and dev.startswith("panel")) else (1,)):
                key = "%s|%d" % (nm, K_)
                if key + "__gap" not in Z.files:
                    continue
                gap = Z[key + "__gap"]; ext = Z[key + "__ext"]; vol = Z[key + "__vol"]; e1 = Z[key + "__e1"]; e3 = Z[key + "__e3"]
                masks = {"방": [("방", room_m)], "욕실": [("욕실", bath_m)], "모두": [("방", room_m), ("욕실", bath_m)] if a.scene == "room" else [("모두", room_m)]}[tgt]
                for lab, m in masks:
                    if not m.any():
                        continue
                    cov = np.array([(gap[m] <= q).mean() for q in Q]); k = int(np.argmax(cov >= 0.8)) if (cov >= 0.8).any() else len(Q) - 1
                    r = dict(name=nm, dev=dev, target=lab, k=K_, n=int(m.sum()), q=float(Q[k]), cover=float(cov[k]), ext=float(np.median(ext[m, k])), ext80=float(np.percentile(ext[m, k], 80)),
                             vol_l=float(np.median(vol[m, k]) * 1000), e1=float(np.median(e1[m])), in50=float((e1[m] <= 0.5).mean()), top3=float((e3[m] <= 0.5).mean()),
                             cover_by_q=cov.tolist(), ext_by_q=np.median(ext[m], 0).tolist())
                    line = "%-34s %-5s %-4s %6d %8.2f %8.0f %6.2f %6.0f%% %6.0f%%" % (nm[:34], {"panel": "패널", "panelrot": "돌림", "plate": "돌판"}[dev], lab, K_, r["ext"], r["vol_l"], r["e1"], 100 * r["in50"], 100 * r["top3"])
                    if a.scene == "exam":
                        for f_ in ("seat1", "seat_nb", "seat3"):
                            r[f_] = float(Z[key + "__" + f_][m].mean())
                        line += " %5.0f%% %5.0f%% %5.0f%%" % (100 * r["seat1"], 100 * r["seat_nb"], 100 * r["seat3"])
                    print(line); out["rows"].append(r)
    # 숨긴 곳마다 (가장 많이 쓰는 조합: 둘째 조합, 사진 1장)
    nm = cfg["combos"][1 if a.scene != "exam" else 0][0]; print("\n-- 숨긴 곳마다: %s (최고점 오차 가운데값 m / 50 cm 안)" % nm); out["by_cat"] = {}
    for dev in ("panel", "panelrot", "plate"):
        f = os.path.join(OUT, "vol_%s_%s%s.npz" % (a.scene, dev, TAG))
        if not os.path.exists(f):
            continue
        Z = np.load(f); cats = [str(c) for c in Z["cats"]]; cat = Z["cat"]; e1 = Z["%s|1__e1" % nm]
        for i, cn in enumerate(cats):
            m = cat == i
            if m.any():
                out["by_cat"].setdefault(cn, {})[dev] = dict(e1=float(np.median(e1[m])), in50=float((e1[m] <= 0.5).mean()))
    for cn, d in out["by_cat"].items():
        print("%-34s " % cn[:34] + " | ".join("%s %4.2f / %3.0f%%" % ({"panel": "패널", "panelrot": "돌림", "plate": "돌판"}[k], v["e1"], 100 * v["in50"]) for k, v in d.items()))
    json.dump(out, open(os.path.join(OUT, "vol_%s%s.json" % (a.scene, TAG)), "w"), ensure_ascii=False, indent=1)


def paths(a):
    import subprocess
    if a.scene == "room":
        for s in (1, 2, 3):
            if not os.path.exists(os.path.join(PATHS, "room_v_s%d.npz" % s)):
                subprocess.run([sys.executable, "room_paths.py", "room_v_s%d" % s, "--seed", str(400 + s), "--per", "12", "--free"], cwd=HERE, check=True)
        return
    import mitsuba as mi
    import sionna.rt as rt
    import scene_kr as SK
    from rt_paths import solve_spot, FC
    SK.set_room("kr40"); sp, aims, names = lec_spots()
    for s in (1, 2, 3):
        name = "lecmk_s%d" % s
        if os.path.exists(os.path.join(PATHS, name + ".npz")):
            continue
        rng = np.random.default_rng(500 + s); cams, cat, cats = lec_hide(rng, 12)
        scene = SK.build(people=False); scene.frequency = FC
        arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH"); scene.tx_array = arr; scene.rx_array = arr
        for i, c in enumerate(cams): scene.add(rt.Receiver("c%03d" % i, position=mi.Point3f(*[float(v) for v in c])))
        solver = rt.PathSolver(); res = []; t0 = time.time(); none = np.array([-1]); done = {}
        for k, pos in enumerate(sp):
            if pos not in done:
                done[pos], _ = solve_spot(scene, solver, pos, cams, 3, True, 10 ** 6, none)
            res.append(done[pos]); print("%2d/%d %s (%.0f초)" % (k + 1, len(sp), names[k], time.time() - t0), flush=True)
        P = max(o["A"].shape[1] for o in res)

        def pad(x):
            return np.pad(x, [(0, 0), (0, P - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
        np.savez_compressed(os.path.join(PATHS, name + ".npz"), spots=np.array(sp), aims=np.array(aims), names=np.array(names), cams=cams, cat=cat, cats=np.array(cats), fc=FC,
                            **{k: np.stack([pad(o[k]) for o in res]) for k in res[0]})
        print("저장", name, "%.0f초" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--scene", required=True, choices=list(SC)); ap.add_argument("--dev", default="panel", choices=["panel", "plate", "panelrot"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--rep", type=int, default=1); ap.add_argument("--ncam", type=int, default=0)
    a = ap.parse_args(); {"measure": measure, "table": table, "paths": paths}[a.cmd](a)


if __name__ == "__main__":
    main()
