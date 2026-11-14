"""전파 표(NEC2)로 '재는 과정'을 흉내 내서, 장치와 같은 계산(fusion)으로 방향을 구해 본다.

world.py 의 방 모형과 같은 조건을 쓴다: 직접파 + 반사파 4개(세기는 직접파보다 K dB 약함), 패킷 수에 따른 흔들림, 관절 각도 오차.
다른 점: 판이 가리는 정도를 수식 모형 대신 NEC 로 푼 값에서 꺼낸다. 그래서 안테나 무늬, 편파, 수신기 자기 부품의 영향이 들어간다.
"""
import os, sys, json
import numpy as np
import empaths
HERE = empaths.HERE
import device as V
from estimator import unit, azel, frame, to_scan
import fusion as F
import world as W

TILT3 = W.TILT3
PLAN3 = [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)]
PLAN6 = [(y, TILT3) for y in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0)]


class Resp:
    """표 하나: E[spin, theta, phi, 2]. look(방향(머리 좌표), 편파 벡터(머리 좌표), 판 각 배열) -> 복소 세기"""
    def __init__(self, name):
        f = os.path.join(V.CACHE, name + ".npz"); self.name = name
        if os.path.exists(f):
            d = np.load(f); self.E = d["E"]; self.spins = d["spins"]
        else:                                             # 저장소에 넣어 둔 줄인 표 (tables/): 16비트 실수·허수로 눌러 담았다 (tables.py)
            import tables as TB
            self.E, self.spins = TB.load(name)
        self.ds = float(self.spins[1] - self.spins[0])

    def at(self, dh):
        """방향 dh(머리 좌표 단위 벡터)에서의 E[spin, 2] 와 그 방향의 theta, phi 단위 벡터"""
        th = np.degrees(np.arccos(np.clip(dh[2], -1, 1))); ph = np.degrees(np.arctan2(dh[1], dh[0])) % 360.0
        ft = th / 5.0; i0 = int(min(np.floor(ft), len(V.TH) - 2)); wt = ft - i0
        fp = ph / 5.0; j0 = int(np.floor(fp)) % len(V.PH); j1 = (j0 + 1) % len(V.PH); wp = fp - np.floor(fp)
        E = self.E
        e = (1 - wt) * ((1 - wp) * E[:, i0, j0] + wp * E[:, i0, j1]) + wt * ((1 - wp) * E[:, i0 + 1, j0] + wp * E[:, i0 + 1, j1])
        t, p = np.radians(th), np.radians(ph)
        tu = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)]); pu = np.array([-np.sin(p), np.cos(p), 0.0])
        return e, tu, pu

    def look(self, dh, eh, phi):
        e, tu, pu = self.at(dh)
        v = e[:, 0] * (eh @ tu) + e[:, 1] * (eh @ pu)                         # 판 각 5도마다의 값
        x = (np.asarray(phi, float) % 360.0) / self.ds; k0 = np.floor(x).astype(int) % len(v); k1 = (k0 + 1) % len(v); w = x - np.floor(x)
        return (1 - w) * v[k0] + w * v[k1]


def rand_perp(rng, d):
    v = rng.normal(size=3); v -= (v @ d) * d
    return v / np.linalg.norm(v)


class EMWorld:
    """한 번의 실험. resp: Resp 하나(받침 각과 상관없는 표) 또는 {받침 각: Resp} (요크·받침까지 넣은 표)"""
    def __init__(self, rng, c, resp, env="보통", multipath=True):
        e = W.ENV[env]; self.rng = rng; self.c = c; self.resp = resp; self.rate = e["rate"]; self.pkt = e["pkt"]
        self.rays = [(1.0 + 0j, c, rand_perp(rng, c))]
        if multipath:
            kdb = rng.uniform(*e["k"]); n = 4
            amp = rng.rayleigh(1.0, n); amp *= np.sqrt(10 ** (-kdb / 10) / np.sum(amp ** 2))
            az = rng.uniform(0, 360, n); el = np.clip(rng.normal(0, 22, n), -70, 70); el[0] = rng.choice([-1, 1]) * rng.uniform(25, 65)
            for a, z, v in zip(amp, az, el):
                d = unit(z, v); self.rays.append((a * np.exp(1j * rng.uniform(0, 2 * np.pi)), d, rand_perp(rng, d)))

    def table_for(self, yaw):
        if isinstance(self.resp, dict):
            keys = np.array(list(self.resp.keys())); k = keys[int(np.argmin(np.abs((keys - yaw + 180.0) % 360.0 - 180.0)))]
            return self.resp[float(k)]
        return self.resp

    def scan(self, yaw, tilt, noise=True):
        rng = self.rng
        a, ex, ey = frame(yaw + (rng.normal(0, 1.0) if noise else 0.0), tilt + (rng.normal(0, 1.0) if noise else 0.0)); zero = rng.normal(0, 2.0) if noise else 0.0
        phi = np.arange(0.0, 360.0, W.BIN) + W.BIN / 2; R = self.table_for(yaw)
        E = np.zeros(len(phi), dtype=complex)
        for amp, d, pol in self.rays:
            dh = np.array([d @ ex, d @ ey, d @ a]); eh = np.array([pol @ ex, pol @ ey, pol @ a])
            E += amp * R.look(dh, eh, phi + zero)
        lv = 20 * np.log10(np.maximum(np.abs(E), 1e-9))
        if noise:
            n = rng.poisson(self.rate * W.REV_S * W.BIN / 360.0, len(phi))
            lv = lv + rng.normal(0, 1, len(phi)) * self.pkt / np.sqrt(np.maximum(n, 1)); lv[n == 0] = np.nan
        return phi, lv


def trial(rng, c, resp, plan=PLAN3, env="보통", multipath=True, fus=None, noise=True):
    w = EMWorld(rng, c, resp, env, multipath); fus = fus or F.Fusion()
    curves = [w.scan(y, t, noise) for y, t in plan]
    est, obs = fus.solve(curves, plan)
    out = dict(seen=int(sum(o["seen"] for o in obs)), err=None, scans=[])
    for (y, t), o in zip(plan, obs):
        psi, beta = to_scan(c, frame(y, t))
        out["scans"].append(dict(beta=float(beta), seen=bool(o["seen"]), depth=float(o["depth"]), dpsi=float((o["psi"] - psi + 180.0) % 360.0 - 180.0)))
    if est is not None:
        out["err"] = float(np.degrees(np.arccos(np.clip(est @ c, -1, 1))))
    return out


def run(resp, n=300, seed=11, plan=PLAN3, env="보통", multipath=True, noise=True, b_range=None):
    """n 번 돌려 통계를 낸다. b_range = (판이 가리는 범위의 아래, 위 각): 판 크기가 다르면 바꿔 준다."""
    old = (F.B_LO, F.B_HI)
    if b_range:
        F.B_LO, F.B_HI = b_range
    try:
        cams = W.cameras(n, seed); fus = F.Fusion(); res = []
        for i, c in enumerate(cams):
            res.append(trial(np.random.default_rng(seed * 1000 + i), c, resp, plan, env, multipath, fus, noise))
    finally:
        F.B_LO, F.B_HI = old
    return summarize(res)


def summarize(res):
    err = np.array([r["err"] if r["err"] is not None else 180.0 for r in res])
    sc = [s for r in res for s in r["scans"]]
    cov = [s for s in sc if F.B_LO + 8 < s["beta"] < F.B_HI - 8]              # 판이 뚜렷이 가리는 범위 안의 스캔
    dps = np.array([abs(s["dpsi"]) for s in cov if s["seen"]])
    return dict(n=len(res), med=float(np.median(err)), p80=float(np.percentile(err, 80)), over15=float(np.mean(err > 15) * 100), over30=float(np.mean(err > 30) * 100),
                none=float(np.mean([r["err"] is None for r in res]) * 100), seen_in_cov=float(np.mean([s["seen"] for s in cov]) * 100) if cov else 0.0,
                depth_med=float(np.median([s["depth"] for s in cov])) if cov else 0.0, dpsi_med=float(np.median(dps)) if len(dps) else 0.0,
                dpsi_p90=float(np.percentile(dps, 90)) if len(dps) else 0.0)


def line(name, s):
    return "%-26s 방향 오차 가운데값 %5.1f도, 80%%가 %5.1f도 안, 15도 넘음 %4.1f%%, 30도 넘음 %4.1f%% | 가리는 범위에서 골이 보인 비율 %3.0f%%, 골 깊이 %4.1f dB, 골 가운데 오차 %4.1f도 (90%%: %4.1f)" % (
        name, s["med"], s["p80"], s["over15"], s["over30"], s["seen_in_cov"], s["depth_med"], s["dpsi_med"], s["dpsi_p90"])


if __name__ == "__main__":
    names = sys.argv[1:] or ["ideal_z", "ant", "ant_conn", "head", "head_cw"]
    out = {}
    for mp, label in ((False, "반사 없음"), (True, "반사 있음(보통 방)")):
        print("==", label)
        for nm in names:
            if not os.path.exists(os.path.join(V.CACHE, nm + ".npz")):
                continue
            s = run(Resp(nm), n=300, multipath=mp); out["%s|%s" % (nm, label)] = s; print(line(nm, s), flush=True)
