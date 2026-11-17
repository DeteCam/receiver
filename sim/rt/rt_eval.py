"""광선 추적(Sionna RT)으로 구한 교실의 전파 길 + NEC2 로 푼 수신기 무늬 표로 '재는 과정'을 만들어, 장치 코드에 그대로 넣는다.

    python rt_eval.py --list                 경우 이름 보기
    python rt_eval.py [--n 200] [--only 이름 ...] [--jobs 10]        결과: rt_eval_result.json

만드는 것: 패킷마다의 CSI 세기 [패킷, 서브캐리어 52] (판 각도, 길마다의 지연·방향·편파, 사람 몸에 닿는 길의 흔들림, 잡음)
읽는 법 셋을 견준다
  trough  지금 장치 코드: analyze3.scan_curve (패킷 -> 곡선) -> fusion.Fusion.read (곡선 -> 골 가운데) -> locate.score_points
  tm      곡선 전체 맞추기: 후보 점마다 '직접파만 있다면 보였을 곡선'을 무늬 표에서 꺼내 잰 곡선과 견준다 (두 편파의 세기 합)
  tmp     곡선 전체 맞추기 + 편파: 한 곳에서 잰 스캔들이 같은 편파를 봤다고 두고, 편파 14가지 가운데 가장 잘 맞는 것으로 견준다
자리 고르기는 장치 코드(locate.ExamSession)의 방법 그대로: 곳마다의 점수를 곱하고, 한 곳이 통째로 틀릴 가능성을 깔아 둔다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                      # 일꾼 여럿이 저마다 여러 가닥을 쓰면 서로 밀려 수십 배 느려진다
import sys, json, time, argparse                        # noqa: E402
import numpy as np                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.dirname(HERE); ROOT = os.path.dirname(SIM)
for _p in (os.path.join(ROOT, "firmware", "pi"), os.path.join(ROOT, "firmware", "core"), os.path.join(SIM, "em"), os.path.join(SIM, "em", "model")):
    sys.path.insert(0, _p)
import analyze3 as A3                                   # noqa: E402  장치 코드
from estimator import frame                             # noqa: E402
from fusion import Fusion, antenna_pos, origin          # noqa: E402
import locate as K                                      # noqa: E402
import tables as TB                                     # noqa: E402  NEC2 표

C0 = 299792458.0
REV_S = 4096 * 0.0015                                   # 판 한 바퀴 6.14초 (28BYJ-48)
SUB = np.r_[np.arange(-26, 0), np.arange(1, 27)] * 312.5e3
TILT3 = float(np.degrees(np.arctan(np.sqrt(2.0))))
Y6 = (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0)
PLANS = {"tilt3": [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], "tilt6": [(y, TILT3) for y in Y6],
         "tilt12": [(y, p) for p in (40.0, 70.0) for y in Y6],
         "tilt18": [(y, p) for i, p in enumerate((30.0, TILT3, 78.0)) for y in (Y6 if i % 2 == 0 else Y6[::-1])],
         "tilt24": [(y, p) for i, p in enumerate((25.0, 45.0, 62.0, 80.0)) for y in (Y6 if i % 2 == 0 else Y6[::-1])],
         "tilt36": [(y, p) for i, p in enumerate((20.0, 32.0, 44.0, TILT3, 67.0, 80.0)) for y in (Y6 if i % 2 == 0 else Y6[::-1])],
         "flat6": [(y, 85.0) for y in Y6], "up6": [(y, 20.0) for y in Y6], "up12": [(y, p) for p in (15.0, 35.0) for y in Y6]}
POSE = dict(pos=0.05, yaw=3.0)                          # 폰 AR 이 잰 수신기 자리·향한 쪽의 오차 (m, 도)
KIND = {"책상": (0, 1), "가슴": (2,), "안경": (3,)}
XS = 0.6 + np.arange(6) * 1.1 + 0.025; YS = 2.0 + np.arange(7) * 0.95
POLS = [(1.0, 0.0)] + [(np.cos(np.radians(c)), np.sin(np.radians(c)) * np.exp(1j * np.radians(d))) for c in (30.0, 60.0) for d in (0.0, 90.0, 180.0, 270.0)] + [(0.0, 1.0)] + \
       [(np.sqrt(0.5), np.sqrt(0.5) * np.exp(1j * np.radians(d))) for d in (45.0, 135.0, 225.0, 315.0)]


def rotz(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a); v = np.asarray(v, float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class Pattern:
    """수신기 전체의 무늬 표 E[판 각 72, theta 37, phi 72, 편파 2] (머리 좌표). 받침 각이 가장 가까운 표를 쓴다"""
    def __init__(self, names=None):
        names = names or {-120.0: "dev_y-120", 0.0: "dev_y+000", 120.0: "dev_y+120"}
        self.T = {k: TB.load(v)[0] for k, v in names.items()}; self.keys = np.array(list(self.T))

    def get(self, yaw, dh):
        E = self.T[float(self.keys[int(np.argmin(np.abs((self.keys - yaw + 180.0) % 360.0 - 180.0)))])]
        th = np.degrees(np.arccos(np.clip(dh[:, 2], -1, 1))); ph = np.degrees(np.arctan2(dh[:, 1], dh[:, 0])) % 360.0
        ft = th / 5.0; i0 = np.clip(np.floor(ft), 0, 35).astype(int); wt = (ft - i0)[None, :, None]
        fp = ph / 5.0; j0 = np.floor(fp).astype(int) % 72; j1 = (j0 + 1) % 72; wp = (fp - np.floor(fp))[None, :, None]
        e = (1 - wt) * ((1 - wp) * E[:, i0, j0] + wp * E[:, i0, j1]) + wt * ((1 - wp) * E[:, i0 + 1, j0] + wp * E[:, i0 + 1, j1])
        t, p = np.radians(th), np.radians(ph)
        tu = np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)], -1); pu = np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], -1)
        return e, tu, pu                                   # e[72, P, 2]

    def world(self, yaw, fr, d, th_hat, ph_hat):
        """방향 d (P,3) 와 그 방향의 세계(또는 수신기) 좌표 편파 기준 (theta, phi) 에 대한 무늬 g[72, P, 2]. fr = frame(...) 같은 좌표"""
        a, ex, ey = fr
        dh = np.stack([d @ ex, d @ ey, d @ a], -1); e, tu, pu = self.get(yaw, dh)
        et = np.stack([th_hat @ ex, th_hat @ ey, th_hat @ a], -1); ep = np.stack([ph_hat @ ex, ph_hat @ ey, ph_hat @ a], -1)
        return np.stack([e[:, :, 0] * (et * tu).sum(-1) + e[:, :, 1] * (et * pu).sum(-1), e[:, :, 0] * (ep * tu).sum(-1) + e[:, :, 1] * (ep * pu).sum(-1)], -1)


def sph(th, ph):
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    return np.stack([st * cp, st * sp, ct], -1), np.stack([ct * cp, ct * sp, -st], -1), np.stack([-sp, cp, np.zeros_like(sp)], -1)


def sph_v(v):
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return sph(np.arccos(np.clip(v[..., 2], -1, 1)), np.arctan2(v[..., 1], v[..., 0]))


def power_curve(tt, C, rec, bin_deg=5.0):
    """서브캐리어 전부의 세기 평균(선형)으로 만든 곡선 (dB, 가운데값 기준). 반사끼리 섞여 생기는 들쭉날쭉이 평균으로 줄어든다"""
    phi = A3.plate_angle(rec, tt); nb = int(round(360.0 / bin_deg)); idx = np.minimum((phi / bin_deg).astype(int), nb - 1)
    p = (10 ** (C / 10.0)).mean(1); s = np.bincount(idx, p, nb); n = np.bincount(idx, minlength=nb)
    lv = np.where(n > 0, 10 * np.log10(np.maximum(s, 1e-30) / np.maximum(n, 1)), np.nan)
    return (np.arange(nb) + 0.5) * bin_deg, lv - np.nanmedian(lv)


class RTWorld:
    """한 곳에 놓은 수신기 한 대. D: 길 묶음(npz), s: 놓은 곳 번호, c: 카메라 점 번호"""
    def __init__(self, rng, D, s, c, frames, pat, u, rate=100.0, tv=1.0, snr=30.0, los_only=False, pat_true=None):
        ok = D["ok"][s, c].copy()
        if los_only:
            ok &= D["nint"][s, c] == 0
        A = D["A"][s, c][ok].astype(complex); self.tau = D["tau"][s, c][ok].astype(float); self.body = D["body"][s, c][ok]
        self.d, self.th_hat, self.ph_hat = sph(D["tht"][s, c][ok].astype(float), D["pht"][s, c][ok].astype(float))
        _, tr, pr = sph(D["thr"][s, c][ok].astype(float), D["phr"][s, c][ok].astype(float))
        gc = np.stack([tr @ u, pr @ u], -1)                                    # 카메라 안테나 = 짧은 쌍극자 (축 u)
        self.cj = np.einsum("pi,pij->pj", gc, A)                               # 길마다, 수신기 편파 2개에 대한 계수
        self.fk = float(D["fc"]) + SUB
        self.phase = np.exp(-2j * np.pi * self.tau[:, None] * self.fk[None, :])
        self.rng = rng; self.pat = pat_true or pat; self.rate = rate; self.tv = tv; self.snr = snr
        self.src = D["spots"][s].astype(float); self.cam = D["cams"][c].astype(float); self.head = float(rng.uniform(0, 360))
        self.o = origin(frames); self.base = self.src - rotz(self.o, self.head)
        v = self.cam - self.src; self.c_true = v / np.linalg.norm(v); nint = D["nint"][s, c][ok]; self.los = bool((nint == 0).any())
        pw = (np.abs(self.cj) ** 2).sum(1); self.k_db = float(10 * np.log10(pw[nint == 0].sum() / max(pw.sum(), 1e-30) + 1e-9)) if self.los else -99.0

    def packets(self, yaw, tilt):
        rng = self.rng; y = yaw + rng.normal(0, 1.0); t = tilt + rng.normal(0, 1.0); zero = rng.normal(0, 2.0)
        g = self.pat.world(yaw, frame(self.head + y, t), self.d, self.th_hat, self.ph_hat)
        w72 = g[:, :, 0] * self.cj[None, :, 0] + g[:, :, 1] * self.cj[None, :, 1]             # [판 각 72, 길]
        dr = rotz(antenna_pos(yaw, tilt) - self.o, self.head)                  # 이 스캔의 안테나 자리 (기준점에서 벗어난 만큼)
        geo = np.exp(2j * np.pi * (self.d @ dr)[:, None] * self.fk[None, :] / C0)
        n = max(int(rng.poisson(self.rate * REV_S)), 40); tt = np.sort(rng.uniform(0, REV_S, n)); spin0 = float(rng.uniform(0, 360))
        phi = (spin0 + 360.0 * tt / REV_S) % 360.0                             # 장치가 아는 판 각
        x = ((phi + zero) % 360.0) / 5.0; k0 = np.floor(x).astype(int) % 72; k1 = (k0 + 1) % 72; w = (x - np.floor(x))[:, None]
        W = (1 - w) * w72[k0] + w * w72[k1]                                    # [패킷, 길]
        if self.tv > 0 and self.body.any():                                    # 몸에 닿은 길: 사람이 조금씩 움직여 스캔마다, 스캔 안에서도 흔들린다
            nb = int(self.body.sum()); ph0 = rng.normal(0, 0.6 * self.tv, nb); amp = np.abs(rng.normal(0, 0.3 * self.tv, nb)); m = rng.integers(1, 4, nb); ps = rng.uniform(0, 2 * np.pi, nb)
            f = (1 + rng.normal(0, 0.15 * self.tv, nb))[None, :] * np.exp(1j * (ph0[None, :] + amp[None, :] * np.sin(2 * np.pi * m[None, :] * tt[:, None] / REV_S + ps[None, :])))
            W = W.copy(); W[:, self.body] *= f
        H = W @ (self.phase * geo)                                             # [패킷, 서브캐리어]
        sig = np.median(np.abs(H) ** 2); na = np.sqrt(max(sig, 1e-24) * 10 ** (-self.snr / 10) / 2)
        H = H + na * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
        C = 20 * np.log10(np.maximum(np.abs(H), 1e-12)) + rng.normal(0, 0.5, (n, 1)) + rng.normal(0, 0.5, H.shape)
        return tt, C, dict(t=[0.0, REV_S], k=[0.0, 4096.0], spin0=spin0, per_rev=4096.0)

    def scan(self, yaw, tilt):
        """(장치 코드의 곡선, 세기 평균 곡선)"""
        tt, C, rec = self.packets(yaw, tilt)
        ph, lv, _ = A3.scan_curve(tt, C, rec)
        return (ph, lv), power_curve(tt, C, rec)


def _gain(Tc, m):
    """잰 곡선 m[B] 을 틀 Tc[B, N] (dB, 평균을 뺀 것) 으로 설명했을 때 줄어드는 제곱 오차. 틀의 크기는 0~1.2배 사이에서 맞춘다"""
    mc = m - m.mean()
    al = np.clip((Tc * mc[:, None]).sum(0) / np.maximum((Tc ** 2).sum(0), 1e-9), 0.0, 1.2)
    return (mc ** 2).sum() - ((mc[:, None] - al * Tc) ** 2).sum(0)


class TM:
    """곡선 전체 맞추기. score(...) -> 곡선 묶음마다 (편파를 모를 때의 점수, 편파까지 맞춘 점수). 후보 점마다의 로그 점수"""
    def __init__(self, pat, den=160.0):
        self.pat = pat; self.den = den

    def score(self, Xdev, curve_sets, frames, pol=True):
        N = len(Xdev); S = len(curve_sets); s0 = np.zeros((S, N)); sp = np.zeros((S, len(POLS), N))
        for i, (yaw, tilt) in enumerate(frames):
            d, th_hat, ph_hat = sph_v(Xdev - antenna_pos(yaw, tilt))
            g = self.pat.world(yaw, frame(yaw, tilt), d, th_hat, ph_hat)       # [72, N, 2] (수신기 좌표의 편파 기준)
            g = 0.5 * (g + np.roll(g, -1, 0))                                  # 칸 가운데 (2.5, 7.5, ...) 로
            T = [10 * np.log10((np.abs(g) ** 2).sum(-1) + 1e-30)]
            if pol:
                T += [10 * np.log10(np.abs(a * g[:, :, 0] + b * g[:, :, 1]) ** 2 + 1e-30) for a, b in POLS]
            for j, curves in enumerate(curve_sets):
                lv = curves[i][1]; okb = np.isfinite(lv)
                if okb.sum() < 36:
                    continue
                m = lv[okb]
                for k, Tk in enumerate(T):
                    Tk = Tk[okb]; gk = _gain(Tk - Tk.mean(0), m)
                    if k == 0:
                        s0[j] += gk
                    else:
                        sp[j, k - 1] += gk
        return s0 / self.den, sp.max(1) / self.den


def seat_rc(xy, seats_xy=None):
    xs, ys = (XS, YS) if seats_xy is None else (np.unique(np.round(seats_xy[:, 0], 3)), np.unique(np.round(seats_xy[:, 1], 3)))
    return int(np.argmin(np.abs(ys - xy[1]))), int(np.argmin(np.abs(xs - xy[0])))


def post_seat(pts, logs, q=K.Q_BAD, temp=1.0):
    """곳마다의 점수(로그)를 장치와 같은 방법으로 곱한다 (한 곳이 통째로 틀릴 가능성 q)"""
    p = np.ones(len(pts.X))
    for s in logs:
        l = np.exp((s - s.max()) / temp); p *= l + q / (1.0 - q) * l.mean(); p /= p.max()
    p = np.bincount(pts.owner, p, minlength=pts.n); return p / p.sum()


EST = ("trough", "tm", "tmp", "tmw", "tmpw")            # w = 세기 평균 곡선으로


def trial(seed, D, cfg, pat, fus, tm):
    rng = np.random.default_rng(seed); frames = PLANS[cfg["plan"]]
    seats_xy = D["seats"]; empty = set(D["empty"].tolist()) if "empty" in D else set()
    if len(empty) >= len(seats_xy):
        empty = set()                                                          # 빈 교실: 카메라만 책상 위에
    while True:
        seat = int(rng.integers(len(seats_xy)))
        if seat not in empty:
            break
    c = seat * 4 + int(rng.choice(KIND[cfg["kind"]]))
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    allow = cfg.get("spots") or list(range(len(D["spots"])))
    seats = [dict(id=str(k), x=float(x), y=float(y) + 0.225) for k, (x, y) in enumerate(seats_xy)]
    spots = [dict(name="곳 %d" % i, x=float(D["spots"][i][0]), y=float(D["spots"][i][1]), z=float(D["spots"][i][2])) for i in allow]
    E = K.ExamSession(seats, spots, (0.0, -1.0)); logs = dict((e, []) for e in EST); seen = []; kdb = []
    order = cfg.get("order"); pick_by = cfg.get("pick", "tmpw"); start = cfg.get("start")
    pat_true = _G.get("pat_true") if cfg.get("mismatch") else None
    for k in range(cfg["npos"]):
        if order is not None:
            if k >= len(order):
                break
            idx = order[k]
        elif k == 0 and start is not None:
            idx = start
        else:
            idx = E.next_spot(post_seat(E.pts, logs[pick_by]) if k > 0 else None)
            if idx is None:
                break
        w = RTWorld(rng, D, allow[idx], c, frames, pat, u, rate=cfg.get("rate", 100.0), tv=cfg.get("tv", 1.0), snr=cfg.get("snr", 30.0), los_only=cfg.get("los_only", False), pat_true=pat_true)
        both = [w.scan(y, t) for y, t in frames]; curves = [b[0] for b in both]; pcurves = [b[1] for b in both]; obs = fus.read(curves)
        pose = (w.base + rng.normal(0, cfg.get("pose_pos", POSE["pos"]), 3) * np.array([1, 1, 0.5]), w.head + rng.normal(0, cfg.get("pose_yaw", POSE["yaw"])))
        E.add(pose, obs, frames, None, spot=idx)
        Xd = K.to_dev(E.pts.X, pose)
        logs["trough"].append(K.score_points(Xd, obs, frames, K.SIG_SCALE * (K.SIG_MANY if len(frames) >= K.MANY_N else 1.0)))
        a, b = tm.score(Xd, [curves, pcurves], frames); logs["tm"].append(a[0]); logs["tmp"].append(b[0]); logs["tmw"].append(a[1]); logs["tmpw"].append(b[1])
        seen.append(float(np.mean([o["seen"] for o in obs]))); kdb.append(w.k_db)
    out = dict(seat=seat, seen=seen, kdb=kdb, used=[int(i) for i in E.used])
    n_done = len(logs["tm"]); temp = cfg.get("temp", 1.0)
    for name in EST:
        for npos in sorted(set([1, 2, 3, 4, 5, 6, 8, 12, n_done])):
            if npos > n_done:
                continue
            p = post_seat(E.pts, logs[name][:npos], temp=1.0 if name == "trough" else temp); o = np.argsort(-p); best = int(o[0])
            (r0, c0), (r1, c1) = seat_rc(seats_xy[seat], seats_xy), seat_rc(seats_xy[best], seats_xy)
            out["%s@%d" % (name, npos)] = (best == seat, abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1, seat in o[:3].tolist(), float(np.hypot(*(seats_xy[seat] - seats_xy[best]))), float(p[seat]), float(p[o[:3]].sum()))
    return out


_G = {}


def _init(den):
    _G["pat"] = Pattern(); _G["fus"] = Fusion(); _G["tm"] = TM(_G["pat"], den or 160.0); _G["D"] = {}; _G["den0"] = den or 160.0
    try:                                                                       # 표와 실물이 다를 때를 흉내: 만드는 쪽은 '머리만' 표, 맞추는 쪽은 전체 표
        _G["pat_true"] = Pattern({0.0: "head"})
    except Exception:                                                          # noqa: BLE001
        _G["pat_true"] = None


def _load(name):
    if name not in _G["D"]:
        d = np.load(os.path.join(HERE, "paths", name + ".npz")); _G["D"][name] = {k: d[k] for k in d.files if k not in ("meta", "people")}
    return _G["D"][name]


def _one(args):
    seed, cfg = args
    data = cfg["data"]; name = data[seed % len(data)] if isinstance(data, (list, tuple)) else data
    try:
        if cfg.get("den"):
            _G["tm"].den = cfg["den"]
        r = trial(seed, _load(name), cfg, _G["pat"], _G["fus"], _G["tm"]); _G["tm"].den = _G.get("den0", _G["tm"].den)
        return r
    except Exception as ex:                                                    # noqa: BLE001
        import traceback
        return dict(error=repr(ex) + traceback.format_exc()[-300:])


def summarize(rows):
    rows = [r for r in rows if "error" not in r]; out = dict(n=len(rows))
    if not rows:
        return out
    out["seen"] = round(float(np.mean([x for r in rows for x in r["seen"]]) * 100), 1); k = np.array([x for r in rows for x in r["kdb"]])
    out["los"] = round(float(np.mean(k > -90) * 100), 1); out["kdb_med"] = round(float(np.median(k[k > -90])), 1) if (k > -90).any() else None
    for key in sorted(k_ for k_ in rows[0] if "@" in k_):
        v = np.array([r[key] for r in rows if key in r], float)
        out[key] = [round(float(v[:, 0].mean() * 100), 1), round(float(v[:, 1].mean() * 100), 1), round(float(v[:, 2].mean() * 100), 1), round(float(np.mean(v[:, 3])), 2), round(float(v[:, 5].mean() * 100), 1)]
    return out


S1 = ["kr40_h24_s1", "kr40_h24_s2", "kr40_h24_s3"]
CORN, BACK2, FRONT3, SIDE, BACK3 = [0, 1, 2, 3], [2, 3], [15, 16, 17], list(range(4, 12)), [12, 13, 14]


def cases():
    """경우 이름 -> 설정. 위에서부터 차례로 돈다 (중요한 것 먼저)"""
    c = {}
    base = dict(data=S1, plan="tilt18", npos=6, kind="책상")
    for kind in ("책상", "가슴", "안경"):
        c["기준 2.4m · 18번 · 8곳까지 | %s" % kind] = dict(base, kind=kind, npos=8, nx=3)
    for kind in ("책상", "가슴"):
        c["천장 가운데 한 곳 2.5m | %s" % kind] = dict(base, data="kr40_ceil_s1", spots=[0], npos=1, kind=kind, nx=3)
        c["천장 세 곳 2.5m | %s" % kind] = dict(base, data="kr40_ceil_s1", npos=3, kind=kind)
    for h, nm in ((1.2, "kr40_h12_s1"), (1.6, "kr40_h16_s1"), (2.0, "kr40_h20_s1"), (2.2, "kr40_h22_s1"), (2.4, "kr40_h24_s1"), (2.6, "kr40_h26_s1")):
        for kind in ("책상", "가슴"):
            c["높이 %.1fm | %s" % (h, kind)] = dict(base, data=nm, kind=kind)
    for plan, npos in (("tilt3", 18), ("tilt6", 12), ("tilt12", 6), ("tilt18", 4), ("tilt36", 2)):
        c["같은 시간(스캔 약 72번): %d곳 x %s | 책상" % (npos, plan)] = dict(base, plan=plan, npos=npos)
    for plan in ("tilt3", "tilt6", "tilt12", "tilt36", "flat6", "up12"):
        c["재는 횟수 %s | 책상" % plan] = dict(base, plan=plan)
    for nm, sp, kinds in (("네 모서리만", CORN, ("책상", "가슴")), ("뒤 두 모서리만", BACK2, ("책상", "가슴")), ("옆벽 8곳만", SIDE, ("책상",)), ("교탁 쪽 3곳만", FRONT3, ("책상",)),
                          ("모서리 + 교탁", CORN + FRONT3, ("책상",)), ("교탁 + 뒷벽", FRONT3 + BACK3, ("책상",))):
        for kind in kinds:
            c["놓는 곳 %s | %s" % (nm, kind)] = dict(base, spots=sp, npos=min(6, len(sp)), kind=kind)
    c["정해진 차례: 한 바퀴 6곳 | 책상"] = dict(base, order=[16, 8, 10, 13, 6, 4]); c["정해진 차례: 한 바퀴 8곳 | 책상"] = dict(base, order=[16, 8, 10, 3, 13, 2, 6, 4], npos=8)
    c["정해진 차례: 네 모서리 + 옆벽 가운데 둘 | 책상"] = dict(base, order=[0, 1, 2, 3, 5, 9]); c["정해진 차례: 한 바퀴 14곳 x 6번 | 책상"] = dict(base, order=[15, 16, 17, 8, 9, 10, 11, 14, 13, 12, 7, 6, 5, 4], npos=14, plan="tilt6")
    for kind in ("책상", "가슴"):
        c["강의실 72석 2.4m · 8곳까지 | %s" % kind] = dict(base, data=["lec72_h24_s1", "lec72_h24_s2"], kind=kind, npos=8)
    c["강의실 72석 2.7m · 8곳까지 | 책상"] = dict(base, data="lec72_h27_s1", npos=8); c["강의실 72석 천장 가운데 한 곳 2.8m | 책상"] = dict(base, data="lec72_ceil_s1", spots=[0], npos=1)
    c["강의실 72석 천장 세 곳 2.8m | 책상"] = dict(base, data="lec72_ceil_s1", npos=3)
    for kind in ("책상", "가슴"):
        c["빈 교실 2.4m | %s" % kind] = dict(base, data="kr00_h24", kind=kind); c["20명 2.4m | %s" % kind] = dict(base, data="kr20_h24_s1", kind=kind)
    c["통로 가운데 1.2m (견줌) | 책상"] = dict(base, data="kr40_aisle_s1", npos=5)
    c["같은 자리 견줌 (s1만) | 책상"] = dict(base, data="kr40_h24_s1"); c["길 계산 깊이 5 | 책상"] = dict(base, data="kr40_h24_s1_d5"); c["회절 없이 | 책상"] = dict(base, data="kr40_h24_s1_nodiff")
    c["천장이 텍스(석고판) | 책상"] = dict(base, data="kr40_h24_s1_board"); c["직접파만 (이상) | 책상"] = dict(base, data="kr40_h24_s1", los_only=True)
    c["표와 실물이 다름 (머리만 표) | 책상"] = dict(base, mismatch=True); c["자리 오차 큼 (0.15m, 8도) | 책상"] = dict(base, pose_pos=0.15, pose_yaw=8.0)
    c["사람 안 움직임 | 책상"] = dict(base, tv=0.0); c["사람 많이 움직임 | 책상"] = dict(base, tv=2.0)
    c["패킷 초당 30 | 책상"] = dict(base, rate=30.0); c["패킷 초당 300 | 책상"] = dict(base, rate=300.0); c["잡음 큼 (SNR 15dB) | 책상"] = dict(base, snr=15.0)
    for pk in ("trough", "tm"):
        c["다음 곳을 %s 로 고름 | 책상" % pk] = dict(base, pick=pk)
    for den in (40.0, 80.0, 320.0, 640.0):
        for kind in ("책상", "가슴"):
            c["맞춤 세기 den=%g | %s" % (den, kind)] = dict(base, den=den, kind=kind)
    for kind in ("책상", "가슴", "안경"):
        c["권하는 방법: 교탁에서 시작 + 12번 x 6곳 | %s" % kind] = dict(base, start=16, plan="tilt12", kind=kind, nx=3)
        c["나쁜 조건 한꺼번에 (표 다름 + 자리 오차 큼 + 패킷 30 + 많이 움직임) | %s" % kind] = dict(base, mismatch=True, pose_pos=0.15, pose_yaw=8.0, rate=30.0, tv=2.0, kind=kind, nx=2)
    c["광선 수 4배 (수렴 확인) | 책상"] = dict(base, data="kr40_h24_s1_hi")
    for h, nm in ((1.6, "kr40_h16_s1"), (2.0, "kr40_h20_s1"), (2.4, "kr40_h24_s1"), (2.6, "kr40_h26_s1")):
        c["높이 %.1fm | 안경" % h] = dict(base, data=nm, kind="안경")
    c["강의실 72석 2.7m · 8곳까지 | 가슴"] = dict(base, data="lec72_h27_s1", npos=8, kind="가슴"); c["천장 가운데 한 곳 2.5m | 안경"] = dict(base, data="kr40_ceil_s1", spots=[0], npos=1, kind="안경")
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--only", nargs="*"); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--list", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--out", default="rt_eval_result.json"); ap.add_argument("--den", type=float, default=160.0); ap.add_argument("--show", default="tmpw")
    a = ap.parse_args(); C = cases()
    if a.list:
        print("\n".join(C)); return
    names = [k for k in C if (not a.only or any(o in k for o in a.only))]
    have = lambda d: all(os.path.exists(os.path.join(HERE, "paths", x + ".npz")) for x in (d if isinstance(d, (list, tuple)) else [d]))
    f = os.path.join(HERE, a.out); res = json.load(open(f)) if os.path.exists(f) else {}
    if a.resume:
        names = [k for k in names if k not in res]                             # 끊긴 뒤 이어서: 이미 있는 경우는 건너뛴다
    import multiprocessing as mp
    _G["den0"] = a.den
    with mp.Pool(a.jobs, initializer=_init, initargs=(a.den,)) as pool:
        for nm in names:
            if not have(C[nm]["data"]):
                print("길 묶음이 아직 없다:", nm, flush=True); continue
            t0 = time.time(); rows = pool.map(_one, [(1000 + i, C[nm]) for i in range(a.n * C[nm].get("nx", 1))], chunksize=2)
            bad = [r["error"] for r in rows if "error" in r]
            s = summarize(rows); s["cfg"] = {k: v for k, v in C[nm].items() if k != "data"}; res[nm] = s; json.dump(res, open(f, "w"), ensure_ascii=False, indent=1)
            npos = min(C[nm]["npos"], 6); sh = lambda e: s.get("%s@%d" % (e, npos), ["-"] * 3)[:3]
            print("%-38s n=%d 직접파 %3.0f%% K %s | 골찾기 %s | tm %s | tmp %s | tmw %s | tmpw %s | %.0f초 %s" % (
                nm, s["n"], s.get("los", -1), s.get("kdb_med"), sh("trough"), sh("tm"), sh("tmp"), sh("tmw"), sh("tmpw"), time.time() - t0, bad[:1]), flush=True)


if __name__ == "__main__":
    main()
