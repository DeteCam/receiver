"""좁은 방(화장실)과 사람이 가릴 때를 흉내 낸다.  python3 room.py [횟수]   결과: room_result.json

montecarlo.py 는 반사를 '세기 범위'로만 넣었다. 여기서는 방 크기와 벽 재질을 정하고 반사를 하나하나 계산한다.
  - 방은 상자. 직접파 + 벽·바닥·천장에 1~3번 부딪힌 반사파를 거울상(image) 방법으로 모두 구한다 (63개).
  - 반사 세기는 재질과 부딪히는 각으로 정한다 (프레넬 식). 타일·콘크리트는 세고, 석고보드는 약하고, 거울은 전부 되튄다.
  - 사람은 반지름 0.17m, 키 1.7m 기둥. 그 옆을 지나는 파는 기둥 양쪽으로 돌아 들어온다 (띠 모양 가림막의 회절).
  - 수신기 안테나는 받침 축에서 4.8cm 떨어져 있어서 받침을 돌릴 때마다 자리가 바뀐다 (가까운 카메라일수록 방향이 달라 보인다). 이것도 넣었다.
  - 판의 가림, 누설, 패킷 수, 관절 오차, 골 모양(W, V)은 montecarlo.py 와 같다. 방향 계산도 같은 코드(fusion.py)다.
넣지 않은 것: 가구·세면대·변기 같은 물건의 모양(약한 흩어짐으로만), 편파의 자세한 계산, 재는 도중에 사람이 움직여 생기는 빠른 흔들림.
"""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
from model import Plate, LAM
from estimator import unit, azel, frame, to_scan
from fusion import Fusion, consistency
import montecarlo as mc

HERE = os.path.dirname(os.path.abspath(__file__))
LAM_M = LAM / 1000.0
TILT3 = mc.TILT3
ANT_R, ANT_H = 0.020, 0.073          # 안테나: 끄덕임 축에서 뒤로 2cm, 위로 7.3cm (p2.py 의 Y_T, S_T)
AXIS_H = 0.120                       # 끄덕임 축 높이
DISTS = (500.0, 800.0, 1500.0, 3000.0, 5000.0)
DF = np.linspace(-8.75e6, 8.75e6, 8)   # 20MHz 채널 안에서 고른 주파수 8개 (서브캐리어 묶음). 분석 코드처럼 서브캐리어마다 곡선을 만들어 가운데값을 쓴다
C0 = 299792458.0
_PL = None

# 재질: 복소 유전율 (2.4GHz, ITU-R P.2040 의 값에서). None 은 금속(거울 뒤 은막)
EPS = dict(concrete=5.31 - 0.49j, tile=6.0 - 0.5j, brick=3.75 - 0.28j, plaster=2.94 - 0.16j, wood=1.99 - 0.09j, glass=6.27 - 0.09j, metal=None)
# 얇은 판은 두께까지 넣는다: ("slab", 유전율, 두께 mm). 판 속에서 오간 파가 겹쳐서, 두께가 파장의 1/4 쯤이면 두꺼운 벽보다 더 잘 되튄다
EPS.update(hpl13=("slab", 4.8 - 0.25j, 13.0),        # 화장실 칸막이 (합성수지 판 13mm): 정면에서 65% 가 되튄다
           glass6=("slab", 6.27 - 0.09j, 6.0),        # 유리창 6mm
           wood40=("slab", 1.99 - 0.09j, 40.0),       # 나무 문 40mm
           open="open")                               # 뚫린 곳 (칸막이 위아래 틈, 침대처럼 파를 흩어 버리는 곳): 되튀지 않는다

ROOMS = {
    "화장실": dict(L=(1.6, 2.2, 2.3), wall="tile", floor="tile", ceil="concrete", clutter=(-22.0, -14.0), absorb=(0.9, 1.0),
                 patches=[dict(wall="x0", rect=(0.5, 1.2, 0.9, 1.7), mat="metal"),          # 세면대 위 거울 0.7 x 0.8
                          dict(wall="y0", rect=(0.4, 1.1, 0.0, 2.0), mat="wood")],           # 문
                 heights=((0.0, 0.6), (0.45, 0.15), (0.85, 0.25))),                          # 수신기를 놓는 높이와 비율: 바닥, 변기 뚜껑, 세면대
    "작은 방": dict(L=(2.4, 3.0, 2.3), wall="brick", floor="concrete", ceil="plaster", clutter=(-18.0, -10.0), absorb=(0.6, 1.0), patches=[],
                 heights=((0.0, 0.5), (0.45, 0.2), (0.75, 0.3))),
    "보통 방": dict(L=(3.5, 4.5, 2.4), wall="brick", floor="concrete", ceil="plaster", clutter=(-18.0, -10.0), absorb=(0.6, 1.0), patches=[],
                 heights=((0.0, 0.5), (0.45, 0.2), (0.75, 0.3))),
}
PERSON_R, PERSON_H = 0.17, 1.70


def plates():
    global _PL
    if _PL is None:
        _PL = [Plate(dist=d) for d in DISTS]
    return _PL


def fresnel_cs(x):
    """프레넬 적분 C(x), S(x). 근사식(오차 0.002 아래)이라 numpy 만 있으면 된다."""
    x = np.asarray(x, dtype=float); a = np.abs(x)
    f = (1 + 0.926 * a) / (2 + 1.792 * a + 3.104 * a * a)
    g = 1 / (2 + 4.142 * a + 3.492 * a * a + 6.670 * a ** 3)
    t = np.pi * a * a / 2
    c = 0.5 + f * np.sin(t) - g * np.cos(t); s = 0.5 - f * np.cos(t) - g * np.sin(t)
    return np.sign(x) * c, np.sign(x) * s


def strip_factor(v1, v2):
    """폭이 프레넬 변수로 v1..v2 인 띠가 길을 막을 때 남는 전기장 (막지 않을 때 = 1). 띠가 멀리 비켜 있으면 1 에 가깝다."""
    c1, s1 = fresnel_cs(v1); c2, s2 = fresnel_cs(v2)
    return 1.0 - 0.5 * (1 + 1j) * ((c2 - c1) - 1j * (s2 - s1))


def gamma(eps, cos_t, chi):
    """반사 계수. chi = 전기장이 입사면과 이루는 각 (0 이면 TM, 90도면 TE)"""
    if eps is None:
        return -1.0 + 0j
    if isinstance(eps, str):                              # "open"
        return 0j
    if isinstance(eps, tuple):                            # 얇은 판
        _, e, d = eps
        root = np.sqrt(e - (1 - cos_t * cos_t))
        ph = np.exp(-2j * 2 * np.pi * (d / 1000.0) / LAM_M * root)
        slab = lambda r: r * (1 - ph) / (1 - r * r * ph)
        return np.cos(chi) ** 2 * slab((e * cos_t - root) / (e * cos_t + root)) + np.sin(chi) ** 2 * slab((cos_t - root) / (cos_t + root))
    root = np.sqrt(eps - (1 - cos_t * cos_t))
    te = (cos_t - root) / (cos_t + root)
    tm = (eps * cos_t - root) / (eps * cos_t + root)
    return np.cos(chi) ** 2 * tm + np.sin(chi) ** 2 * te


def fold(u, L):
    u = np.mod(u, 2 * L)
    return np.where(u > L, 2 * L - u, u)


class RoomWorld:
    """한 번의 실험. montecarlo.World 와 같은 scan(yaw, tilt) 를 갖는다."""
    def __init__(self, rng, room="화장실", shape="W", person=None, env="보통", order=3, cam=None, base=None, parallax=True, nf=len(DF)):
        R = ROOMS[room]; e = mc.ENV[env]
        self.rng = rng; self.shape = shape; self.rate = e["rate"]; self.pkt = e["pkt"]; self.order = order
        self.parallax = parallax; self.df = DF if nf >= len(DF) else (np.array([0.0]) if nf <= 1 else np.linspace(DF[0], DF[-1], nf))
        self.L = np.array(R["L"]); self.R = R
        self.mat = dict(x0=R["wall"], x1=R["wall"], y0=R["wall"], y1=R["wall"], z0=R["floor"], z1=R["ceil"])
        self.mat.update(R.get("mat", {}))                                      # 벽마다 다른 재질 (있으면)
        self.absorb = {k: rng.uniform(*R["absorb"]) for k in self.mat}        # 벽마다 가구·커튼이 먹는 정도
        hs, ps = zip(*R["heights"]); zs = float(rng.choice(hs, p=ps))
        Lx, Ly, Lz = self.L
        if base is None:
            base = np.array([rng.uniform(0.3, Lx - 0.3), rng.uniform(0.3, Ly - 0.3), zs])
        self.base = np.asarray(base, dtype=float)
        self.ref = self.base + np.array([0, 0, AXIS_H + ANT_R * np.sin(np.radians(TILT3)) + ANT_H * np.cos(np.radians(TILT3))])   # 세 번 재는 동안 안테나가 도는 원의 한가운데
        self.head = rng.uniform(0, 360)                                       # 수신기 앞(받침 0도)이 방에서 향하는 쪽
        if cam is None:
            cam = self.random_cam()
        self.cam = np.asarray(cam, dtype=float)
        self.c = self.to_dev(self.cam - self.ref); self.dist = float(np.linalg.norm(self.cam - self.ref)); self.c = self.c / np.linalg.norm(self.c)
        self.back = rng.uniform(0.15, 0.55)
        self.leak = rng.uniform(0.0, e["leak"]) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        self.v = dict(D=rng.uniform(8.0, 13.0), sig=rng.uniform(18.0, 30.0), lo=rng.uniform(-38.0, -22.0), hi=rng.uniform(38.0, 52.0), w=rng.uniform(5.0, 9.0))
        # 거울상 목록 (i, j, k) 와 길마다 편파 각, 카메라 안테나가 그 쪽으로 내보내는 세기(±2dB)
        idx = [(i, j, k) for i in range(-order, order + 1) for j in range(-order, order + 1) for k in range(-order, order + 1) if abs(i) + abs(j) + abs(k) <= order]
        idx.sort(key=lambda t: abs(t[0]) + abs(t[1]) + abs(t[2]))
        self.idx = np.array(idx)
        self.chi = rng.uniform(0, np.pi / 2, len(idx)); self.gain = 10 ** (rng.normal(0, 2.0, len(idx)) / 20.0); self.gain[0] = 1.0
        par = (self.idx % 2 != 0)
        self.img = self.idx * self.L + np.where(par, self.L - self.cam, self.cam)
        # 가구 같은 물건에서 흩어져 오는 약한 파 5개
        kdb = rng.uniform(*R["clutter"]); n = 5
        amp = rng.rayleigh(1.0, n); amp *= np.sqrt(10 ** (kdb / 10) / np.sum(amp ** 2))
        self.clutter = [(a * np.exp(1j * rng.uniform(0, 2 * np.pi)), unit(rng.uniform(0, 360), np.clip(rng.normal(0, 25), -70, 70)), rng.uniform(0.5, 6.0)) for a in amp]    # (세기, 방향, 더 돌아온 거리 m)
        self.person = None if person is None else np.asarray(person, dtype=float)      # (x, y) 또는 스캔마다 다른 자리의 목록
        self.kfac = None; self.block_db = None; self._nscan = 0

    # ---- 좌표
    def to_dev(self, v):
        a = np.radians(-self.head); c, s = np.cos(a), np.sin(a)
        v = np.asarray(v, dtype=float)
        return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)

    def random_cam(self):
        rng = self.rng; Lx, Ly, Lz = self.L; off = 0.03
        for _ in range(200):
            areas = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly])
            w = int(rng.choice(5, p=areas / areas.sum()))
            z = rng.uniform(0.05, Lz - 0.05)
            p = [np.array([off, rng.uniform(0.05, Ly - 0.05), z]), np.array([Lx - off, rng.uniform(0.05, Ly - 0.05), z]),
                 np.array([rng.uniform(0.05, Lx - 0.05), off, z]), np.array([rng.uniform(0.05, Lx - 0.05), Ly - off, z]),
                 np.array([rng.uniform(0.05, Lx - 0.05), rng.uniform(0.05, Ly - 0.05), Lz - off])][w]
            d = p - self.ref; r = np.linalg.norm(d)
            if r >= 0.45 and d[2] / r > np.sin(np.radians(-80.0)):
                return p
        return p

    def antenna(self, yaw, tilt):
        if not self.parallax:
            return self.ref.copy()
        t = np.radians(tilt); x = -ANT_R * np.cos(t) + ANT_H * np.sin(t); z = AXIS_H + ANT_R * np.sin(t) + ANT_H * np.cos(t)
        a = np.radians(yaw + self.head)
        return self.base + np.array([x * np.cos(a), x * np.sin(a), z])

    # ---- 길마다의 세기 (직접파를 막지 않았을 때 = 1)
    def paths(self, ant, person=None):
        v = self.img - ant; r = np.linalg.norm(v, axis=1); d = v / r[:, None]
        r0 = r[0]
        amp = (r0 / r) * np.exp(-2j * np.pi * (r - r0) / LAM_M) * self.gain
        names = (("x0", "x1"), ("y0", "y1"), ("z0", "z1"))
        for ax in range(3):
            L = self.L[ax]; o1, o2 = [a for a in range(3) if a != ax]
            cos_t = np.abs(d[:, ax])
            for p in range(len(r)):
                i = int(self.idx[p, ax])
                if i == 0:
                    continue
                for n in (range(1, i + 1) if i > 0 else range(i + 1, 1)):
                    t = (n * L - ant[ax]) / v[p, ax]
                    q = ant + t * v[p]
                    u1 = float(fold(q[o1], self.L[o1])); u2 = float(fold(q[o2], self.L[o2]))
                    wall = names[ax][n % 2]
                    mat = self.mat[wall]; ab = self.absorb[wall]
                    for pt in self.R["patches"]:
                        a0, a1, b0, b1 = pt["rect"]
                        if pt["wall"] == wall and a0 <= u1 <= a1 and b0 <= u2 <= b1:
                            mat = pt["mat"]; ab = 1.0
                    amp[p] *= gamma(EPS[mat], cos_t[p], self.chi[p]) * ab
        blk = np.ones(len(r), dtype=complex)
        if person is not None:
            blk = self.block(ant, v, r, d, person)
        return amp * blk, d, r, blk

    def block(self, ant, v, r, d, person):
        """사람(기둥)이 길마다 남기는 비율. 접힌 길을 펴서 보면 사람도 거울상으로 여럿 보인다: 그 모두에 대해 띠 회절을 곱한다."""
        Lx, Ly, Lz = self.L; o = self.order + 1
        out = np.ones(len(r), dtype=complex)
        ii = np.arange(-o, o + 1)
        px = (ii * Lx + np.where(ii % 2 != 0, Lx - person[0], person[0]))[:, None]
        py = (ii * Ly + np.where(ii % 2 != 0, Ly - person[1], person[1]))[None, :]
        PX = np.broadcast_to(px, (len(ii), len(ii))).ravel(); PY = np.broadcast_to(py, (len(ii), len(ii))).ravel()
        for p in range(len(r)):
            h = np.hypot(d[p, 0], d[p, 1])
            if h < 1e-6:
                continue
            ux, uy = d[p, 0] / h, d[p, 1] / h
            s = ((PX - ant[0]) * ux + (PY - ant[1]) * uy) / h                 # 길을 따라 잰 거리 (3차원)
            ok = (s > 0.02) & (s < r[p] - 0.02)
            if not ok.any():
                continue
            b = -(PX - ant[0]) * uy + (PY - ant[1]) * ux                      # 옆으로 벗어난 거리
            z = fold(ant[2] + s * d[p, 2], Lz)
            ok &= (np.abs(b) < PERSON_R + 0.6) & (z < PERSON_H + 0.05)
            for q in np.nonzero(ok)[0]:
                d2 = s[q]; d1 = r[p] - s[q]
                kap = np.sqrt(2 * (d1 + d2) / (LAM_M * d1 * d2))
                out[p] *= strip_factor((b[q] - PERSON_R) * kap, (b[q] + PERSON_R) * kap)
        return out

    def look(self, P, beta, dphi):
        if self.shape == "W":
            return P.look(beta, dphi)
        v = self.v
        s = 1.0 / (1.0 + np.exp(-(beta - v["lo"]) / v["w"])) / (1.0 + np.exp((beta - v["hi"]) / v["w"]))
        dd = (np.asarray(dphi) + 180.0) % 360.0 - 180.0
        return 10 ** (-v["D"] * s * np.exp(-(dd / v["sig"]) ** 2) / 20.0) + 0j

    def scan(self, yaw, tilt):
        rng = self.rng
        person = None
        if self.person is not None:
            person = self.person if self.person.ndim == 1 else self.person[min(self._nscan, len(self.person) - 1)]
        self._nscan += 1
        ant = self.antenna(yaw, tilt)
        amp, d, r, blk = self.paths(ant, person)
        dd = self.to_dev(d)
        if self.kfac is None:
            rest = np.sum(np.abs(amp[1:]) ** 2) + sum(abs(a) ** 2 for a, _, _ in self.clutter)
            self.kfac = float(10 * np.log10(max(abs(amp[0]) ** 2, 1e-9) / max(rest, 1e-9)))
            self.block_db = float(-20 * np.log10(max(abs(blk[0]), 1e-6)))
        fr = frame(yaw + rng.normal(0, 1.0), tilt + rng.normal(0, 1.0)); zero = rng.normal(0, 2.0)
        phi = np.arange(0.0, 360.0, mc.BIN) + mc.BIN / 2
        PL = plates(); ld = np.log(np.array(DISTS)); nf = len(self.df)
        E = np.zeros((nf, len(phi)), dtype=complex)
        rot = lambda extra: np.exp(-2j * np.pi * self.df * extra / C0)[:, None]          # 주파수가 다르면 돌아온 거리만큼 위상이 달라진다
        for k in range(len(r)):
            psi, beta = to_scan(dd[k], fr)
            P = PL[int(np.argmin(np.abs(ld - np.log(r[k] * 1000.0))))]
            f = self.look(P, beta, phi + zero - psi)
            if k == 0:
                x = ((phi + zero - psi) % 360.0) - 180.0
                f = f + self.back * np.exp(-(x / 45.0) ** 2) * np.cos(np.radians(beta)) ** 2
            E += rot(r[k] - r[0]) * (amp[k] * f)[None, :]
        for a, u, extra in self.clutter:
            psi, beta = to_scan(u, fr)
            E += rot(extra) * (a * self.look(PL[3], beta, phi + zero - psi))[None, :]
        E += self.leak * amp[0]
        lvf = 20 * np.log10(np.maximum(np.abs(E), 1e-4))
        lv = np.median(lvf - np.median(lvf, axis=1, keepdims=True), axis=0)              # 서브캐리어마다 제 가운데값을 빼고, 서브캐리어끼리의 가운데값
        n = rng.poisson(self.rate * mc.REV_S * mc.BIN / 360.0, len(phi))
        lv = lv + rng.normal(0, 1, len(phi)) * self.pkt / np.sqrt(np.maximum(n, 1))
        lv[n == 0] = np.nan
        return phi, lv


def person_spot(w, rng, mode):
    """사람이 서는 자리. block = 카메라와 수신기 사이를 막고 선다, near = 수신기 옆 아무 데나, move = 스캔마다 다른 자리(한 번은 막는다)"""
    Lx, Ly, _ = w.L
    def on_line():
        t = rng.uniform(0.3, 0.7); p = w.ref + t * (w.cam - w.ref)
        n = np.array([-(w.cam - w.ref)[1], (w.cam - w.ref)[0]]); n = n / max(np.linalg.norm(n), 1e-6)
        q = p[:2] + n * rng.uniform(-0.08, 0.08)
        return np.clip(q, [PERSON_R, PERSON_R], [Lx - PERSON_R, Ly - PERSON_R])
    def near():
        for _ in range(50):
            a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.35, 0.7)
            q = w.base[:2] + rr * np.array([np.cos(a), np.sin(a)])
            if PERSON_R <= q[0] <= Lx - PERSON_R and PERSON_R <= q[1] <= Ly - PERSON_R:
                return q
        return np.clip(q, [PERSON_R, PERSON_R], [Lx - PERSON_R, Ly - PERSON_R])
    if mode == "block":
        return on_line()
    if mode == "near":
        return near()
    if mode == "move":
        spots = [near(), near(), near(), near()]; spots[int(rng.integers(3))] = on_line()
        return np.array(spots)
    return None


def run(n=600, room="화장실", shape="W", person=None, env="보통", seed=11, plan="tilt3", order=3, parallax=True, nf=len(DF), dist=None):
    fus = Fusion(); frames = [tuple(f) for f in {"tilt3": [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], "tilt4": [(-135.0, 60.0), (-45.0, 60.0), (45.0, 60.0), (135.0, 60.0)],
                                                 "up+tilt3": [(-120.0, 0.0), (-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)]}[plan]]
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        w = RoomWorld(rng, room, shape, None, env, order, parallax=parallax, nf=nf)
        if person:
            w.person = person_spot(w, rng, person)
        curves = [w.scan(y, t) for y, t in frames]
        dd = w.dist if dist == "true" else dist
        est, obs = fus.solve(curves, frames, dist=dd)
        nseen = sum(o["seen"] for o in obs)
        if est is None:
            rows.append((180.0, 0, 0, w.kfac, w.block_db, w.dist, azel(w.c)[1])); continue
        ok, _ = consistency(est, obs, frames, dist=dd)
        ang = float(np.degrees(np.arccos(np.clip(est @ w.c, -1, 1))))
        rows.append((ang, nseen, int(ok), w.kfac, w.block_db, w.dist, float(azel(w.c)[1])))
    return np.array(rows)


def summ(R):
    e = R[:, 0]; found = R[:, 1] > 0; ok = R[:, 2] > 0
    return dict(n=len(R), med=round(float(np.median(e)), 1), p75=round(float(np.percentile(e, 75)), 1), p90=round(float(np.percentile(e, 90)), 1),
                within10=round(float(np.mean(e <= 10) * 100), 1), within20=round(float(np.mean(e <= 20) * 100), 1), over30=round(float(np.mean(e > 30) * 100), 1),
                none=round(float(np.mean(~found) * 100), 1), flagged=round(float(np.mean(found & ~ok) * 100), 1),
                bad_unflagged=round(float(np.mean((e > 30) & found & ok) * 100), 1),
                med_ok=round(float(np.median(e[found & ok])), 1) if (found & ok).any() else None,
                kfac=round(float(np.median(R[:, 3])), 1), block=round(float(np.median(R[:, 4])), 1), dist=round(float(np.median(R[:, 5])), 2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = {}
    for room in ROOMS:
        for person in (None, "block", "near", "move"):
            for shape in ("W", "V"):
                t0 = time.time(); R = run(n, room, shape, person)
                key = "%s|%s|%s" % (room, person or "없음", shape); out[key] = summ(R); s = out[key]
                print("%-22s 가운데값 %5.1f도  75%% %5.1f  90%% %5.1f | 10도 안 %4.1f%%  30도 넘음 %4.1f%% | 못 찾음 %4.1f%%  경고 %4.1f%%  경고 없이 틀림 %4.1f%% | K %4.1fdB  가림 %4.1fdB  거리 %.2fm  (%.0f초)" % (
                    key, s["med"], s["p75"], s["p90"], s["within10"], s["over30"], s["none"], s["flagged"], s["bad_unflagged"], s["kfac"], s["block"], s["dist"], time.time() - t0), flush=True)
    json.dump(dict(n=n, summary=out), open(os.path.join(HERE, "room_result.json"), "w"), ensure_ascii=False, indent=1)
