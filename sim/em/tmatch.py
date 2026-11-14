"""무늬 맞추기(template matching)로 방향을 구한다.

지금 장치의 계산(fusion)은 곡선에서 '골' 하나를 찾는다. 전파를 제대로 풀어 보면 곡선은 골 하나가 아니라
W 모양, 뒤쪽 혹, 안테나 무늬가 섞인 모양이다. 그래서 곡선 '전체'를, 방향마다 미리 계산해 둔 곡선(전파 표)과 견준다.

  - 카메라 전파의 편파(흔들리는 방향)는 모른다. 그런데 받는 세기는 방향마다 네 개의 기본 곡선의 합으로 정확히 써진다:
        P(판 각) = c0·|Eθ|² + c1·|Eφ|² + c2·2Re(Eθ·Eφ*) + c3·(−2Im(Eθ·Eφ*))        (스토크스 변수 4개)
    그러니 방향 하나마다 '네 숫자를 가장 잘 고른 뒤 남는 차이'를 한 번의 선형 계산으로 구할 수 있다.
  - 잰 값의 흔들림은 dB 로 고르므로(곱하는 꼴) 세기의 '상대 오차'를 줄인다 (가중치 1/P²).
  - 자세 여러 개는 log(남는 차이)를 더한다 (자세마다 흔들림 크기를 모르는 경우의 가능도).
"""
import os, sys
import numpy as np
import empaths
HERE = empaths.HERE
import device as V
import study as S
import world as W
from estimator import frame, unit

PHI = np.arange(0.0, 360.0, W.BIN) + W.BIN / 2


def fib(n=5000, el_min=-80.0):
    i = np.arange(n) + 0.5; z = 1 - 2 * i / n; r = np.sqrt(1 - z * z); t = np.pi * (1 + 5 ** 0.5) * i
    v = np.stack([r * np.cos(t), r * np.sin(t), z], 1)
    return v[v[:, 2] > np.sin(np.radians(el_min))]


GRID = fib()


class Templates:
    """전파 표 하나를 '방향마다의 기본 곡선 4개'로 바꿔 둔다. B[theta, phi, 판 각(잰 칸), 4]"""
    def __init__(self, resp, phi=PHI):
        self.resp = resp; self.phi = np.asarray(phi, float)
        x = (self.phi % 360.0) / resp.ds; k0 = np.floor(x).astype(int) % len(resp.spins); k1 = (k0 + 1) % len(resp.spins); w = (x - np.floor(x)).astype(np.float32)
        self.k0, self.k1, self.w = k0, k1, w
        Ei = (1 - w)[:, None, None, None] * resp.E[k0] + w[:, None, None, None] * resp.E[k1]           # [n, th, ph, 2]
        self.B = self.basis(Ei[..., 0], Ei[..., 1]).transpose(1, 2, 0, 3).astype(np.float32)             # [th, ph, n, 4]

    @staticmethod
    def basis(et, ep):
        x = et * np.conj(ep)
        return np.stack([np.abs(et) ** 2, np.abs(ep) ** 2, 2 * x.real, -2 * x.imag], -1)

    def cost_map(self, lv):
        """잰 곡선 lv(dB, 못 잰 칸은 nan) -> 방향 격자(theta 37 x phi 72)마다 남는 상대 오차의 제곱 평균"""
        ok = ~np.isnan(lv); P = 10 ** (lv[ok] / 10.0); P = P / np.median(P); wgt = 1.0 / P ** 2
        B = self.B[:, :, ok, :]; BW = B * wgt[None, None, :, None]
        A = np.einsum("tpnk,tpnl->tpkl", BW, B, optimize=True); b = np.einsum("tpnk,n->tpk", BW, P, optimize=True)
        # 작은 값을 대각에 더해 풀이가 늘 되게 한다. 뒤의 1e-30 은 안테나가 전혀 못 받는 방향(표가 0)에서도 풀리게 하는 바닥값이다: 그 방향은 '전혀 안 맞음'으로 나온다
        A = A + (1e-6 * np.trace(A, axis1=-2, axis2=-1) / 4.0 + 1e-30)[..., None, None] * np.eye(4, dtype=A.dtype)
        c = np.linalg.solve(A, b[..., None])[..., 0]
        J = len(P) - np.einsum("tpk,tpk->tp", b, c)
        return np.maximum(J, 1e-9) / len(P)

    def cost_at(self, dh, lv):
        """방향 dh[m, 3](머리 좌표 단위 벡터)에서의 남는 차이. 전파 표를 그 방향으로 보간해 정확히 계산한다 (세밀하게 다듬을 때)."""
        dh = np.atleast_2d(dh); E = self.resp.E
        th = np.degrees(np.arccos(np.clip(dh[:, 2], -1, 1))); ph = np.degrees(np.arctan2(dh[:, 1], dh[:, 0])) % 360.0
        ft = th / 5.0; i0 = np.minimum(np.floor(ft).astype(int), len(V.TH) - 2); wt = (ft - i0)[None, :, None]
        fp = ph / 5.0; j0 = np.floor(fp).astype(int) % len(V.PH); j1 = (j0 + 1) % len(V.PH); wp = (fp - np.floor(fp))[None, :, None]
        e = (1 - wt) * ((1 - wp) * E[:, i0, j0] + wp * E[:, i0, j1]) + wt * ((1 - wp) * E[:, i0 + 1, j0] + wp * E[:, i0 + 1, j1])     # [spin, m, 2]
        w = self.w[:, None, None]; e = (1 - w) * e[self.k0] + w * e[self.k1]                              # [n, m, 2]
        ok = ~np.isnan(lv); P = 10 ** (lv[ok] / 10.0); P = P / np.median(P); wgt = 1.0 / P ** 2
        B = self.basis(e[ok, :, 0], e[ok, :, 1]).transpose(1, 0, 2)                                       # [m, n, 4]
        BW = B * wgt[None, :, None]; A = np.einsum("mnk,mnl->mkl", BW, B); b = np.einsum("mnk,n->mk", BW, P)
        A = A + (1e-6 * np.trace(A, axis1=-2, axis2=-1) / 4.0 + 1e-30)[..., None, None] * np.eye(4)
        c = np.linalg.solve(A, b[..., None])[..., 0]
        return np.maximum(len(P) - np.einsum("mk,mk->m", b, c), 1e-9) / len(P)


def interp_map(M, dh):
    """격자 값 M[theta, phi] 를 방향 dh[m, 3] 에서 읽는다 (쌍선형)"""
    th = np.degrees(np.arccos(np.clip(dh[:, 2], -1, 1))); ph = np.degrees(np.arctan2(dh[:, 1], dh[:, 0])) % 360.0
    ft = th / 5.0; i0 = np.minimum(np.floor(ft).astype(int), M.shape[0] - 2); wt = ft - i0
    fp = ph / 5.0; j0 = np.floor(fp).astype(int) % M.shape[1]; j1 = (j0 + 1) % M.shape[1]; wp = fp - np.floor(fp)
    return (1 - wt) * ((1 - wp) * M[i0, j0] + wp * M[i0, j1]) + wt * ((1 - wp) * M[i0 + 1, j0] + wp * M[i0 + 1, j1])


def to_head(c, yaw, tilt):
    a, ex, ey = frame(yaw, tilt); c = np.atleast_2d(c)
    return np.stack([c @ ex, c @ ey, c @ a], 1)


def solve(curves, plan, tmpl, refine=True, grid=GRID):
    """curves: [(phi, lv)], plan: [(yaw, tilt)], tmpl: Templates 하나 또는 {받침 각: Templates}.  -> (방향 단위 벡터, 정보)"""
    def T(y):
        if isinstance(tmpl, dict):
            ks = np.array(list(tmpl.keys())); return tmpl[float(ks[int(np.argmin(np.abs((ks - y + 180.0) % 360.0 - 180.0)))])]
        return tmpl
    maps = [np.log(T(y).cost_map(lv)) for (phi, lv), (y, t) in zip(curves, plan)]
    n = [int(np.sum(~np.isnan(lv))) for phi, lv in curves]
    sc = np.zeros(len(grid))
    for M, k, (y, t) in zip(maps, n, plan):
        sc += 0.5 * k * interp_map(M, to_head(grid, y, t))
    order = np.argsort(sc); best = grid[order[0]]; info = dict(score=float(sc[order[0]]))
    # 두 번째로 좋은 '다른 곳'(30도 넘게 떨어진 곳)과의 차이 = 답을 얼마나 믿을 만한가
    far = order[np.degrees(np.arccos(np.clip(grid[order] @ best, -1, 1))) > 30.0]
    info["margin"] = float(sc[far[0]] - sc[order[0]]) if len(far) else 0.0
    if not refine:
        return best, info
    cands = [best] + ([grid[far[0]]] if len(far) else [])
    out = []
    for c0 in cands:
        c = c0.copy(); step = 3.0
        for it in range(4):                                         # 둘레 5x5 점을 정확히 계산해 가장 좋은 점으로 옮기기를 되풀이 (3도 -> 0.4도)
            u = np.cross(c, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); v = np.cross(c, u)
            g = np.radians(step) * np.arange(-2, 3); pts = np.array([c + a * u + b * v for a in g for b in g]); pts /= np.linalg.norm(pts, axis=1)[:, None]
            s = np.zeros(len(pts))
            for (phi, lv), k, (y, t) in zip(curves, n, plan):
                s += 0.5 * k * np.log(T(y).cost_at(to_head(pts, y, t), lv))
            c = pts[int(np.argmin(s))]; step /= 2.0
        out.append((float(np.min(s)), c))
    out.sort(key=lambda x: x[0]); info["score"] = out[0][0]
    return out[0][1], info


def trial(rng, c, world_resp, tmpl, plan=S.PLAN3, env="보통", multipath=True, noise=True):
    w = S.EMWorld(rng, c, world_resp, env, multipath)
    curves = [w.scan(y, t, noise) for y, t in plan]
    est, info = solve(curves, plan, tmpl)
    return float(np.degrees(np.arccos(np.clip(est @ c, -1, 1)))), info


def run(world_resp, tmpl, n=200, seed=11, **kw):
    cams = W.cameras(n, seed); err = []; mar = []
    for i, c in enumerate(cams):
        e, info = trial(np.random.default_rng(seed * 1000 + i), c, world_resp, tmpl, **kw); err.append(e); mar.append(info["margin"])
    err = np.array(err)
    return dict(n=n, med=float(np.median(err)), p80=float(np.percentile(err, 80)), p95=float(np.percentile(err, 95)), over15=float(np.mean(err > 15) * 100), over30=float(np.mean(err > 30) * 100),
                err=[round(float(e), 2) for e in err], margin=[round(float(m), 2) for m in mar])


def line(name, s):
    return "%-30s 가운데값 %5.1f도 | 80%%가 %5.1f도 안 | 95%%가 %5.1f도 안 | 15도 넘음 %4.1f%% | 30도 넘음 %4.1f%%" % (name, s["med"], s["p80"], s["p95"], s["over15"], s["over30"])


if __name__ == "__main__":
    import time
    names = sys.argv[1:] or ["ideal_z", "head"]
    for nm in names:
        R = S.Resp(nm); t0 = time.time(); T = Templates(R)
        for mp, lab in ((False, "반사 없음"), (True, "보통 방")):
            s = run(R, T, n=100, multipath=mp); print(line("%s | %s" % (nm, lab), s), " (%.0f초)" % (time.time() - t0), flush=True)
