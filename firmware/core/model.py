"""판 뒤 신호의 회절 모형 (3차원). 금속판을 '막힌 구멍'으로 보고 레일리-조머펠트 적분을 판 위에서 더한다.

스칼라 근사다. 편파, 판에 흐르는 전류, 파이 기판과 모터의 반사는 들어 있지 않다. 경향을 보는 용도이고,
손으로 잰 값과 맞춰 본 결과는 validate.py 가 보여 준다.

머리 좌표: 원점 = 안테나, Z = 판 회전축. 판은 반지름 R 에서 접선 방향으로 서 있고, 폭 w(접선), 축 방향으로 z0 ~ z1.
카메라 방향은 (판이 도는 면 안에서의 각 psi, 그 면에서 벗어난 각 beta) 로 쓴다. 판 각 phi 는 +X 에서 잰다.
회전 대칭이라 결과는 (beta, phi - psi) 에만 달려 있다 -> 표로 만들어 두고 꺼내 쓴다.
"""
import os
import numpy as np

LAM = 299792458.0 / 2.437e9 * 1000.0      # 2.4GHz 6번 채널 파장 (mm) = 123.0
K = 2 * np.pi / LAM
HERE = os.path.dirname(os.path.abspath(__file__))


def field(dphi_deg, beta_deg, R=80.0, w=100.0, z0=-58.0, z1=92.0, dist=3000.0, step=2.5):
    """판이 카메라 쪽에서 dphi 만큼 돌아가 있을 때의 (판이 있을 때 전기장) / (없을 때 전기장). 복소수 배열."""
    dphi = np.radians(np.atleast_1d(np.asarray(dphi_deg, dtype=float)))
    b = np.radians(beta_deg)
    src = dist * np.array([np.cos(b), 0.0, np.sin(b)])                       # 카메라는 psi = 0 쪽
    u = np.arange(-w / 2 + step / 2, w / 2, step)
    v = np.arange(z0 + step / 2, z1, step)
    U, V = np.meshgrid(u, v, indexing="ij")
    out = np.empty(len(dphi), dtype=complex)
    for i, p in enumerate(dphi):
        n = np.array([np.cos(p), np.sin(p), 0.0]); t = np.array([-np.sin(p), np.cos(p), 0.0])
        Q = R * n + U[..., None] * t + V[..., None] * np.array([0.0, 0.0, 1.0])
        d1 = src - Q
        r1 = np.linalg.norm(d1, axis=-1); r2 = np.linalg.norm(Q, axis=-1)
        obl = np.clip(((d1 @ n) / r1 + (Q @ n) / r2) / 2, 0.0, None)          # 프레넬-키르히호프 기울기 인자
        ker = (1 / (2 * np.pi)) * (1 / r2 - 1j * K) * obl * np.exp(1j * K * (r1 + r2 - dist)) / (r1 * r2) * dist
        out[i] = 1.0 - np.sum(ker) * step * step
    return out


def table(R=80.0, w=100.0, z0=-58.0, z1=92.0, dist=3000.0, dbeta=2.0, dphi=1.0, step=2.5, cache=True):
    """복소 전기장 표 F[beta, dphi]. beta = -90..90, dphi = -180..179."""
    name = "table_R%d_w%d_z%d_%d_d%d_b%g_p%g.npz" % (R, w, z0, z1, dist, dbeta, dphi)
    path = os.path.join(HERE, "cache", name)
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["betas"], d["dphis"], d["F"]
    betas = np.arange(-90.0, 90.0 + 1e-9, dbeta)
    dphis = np.arange(-180.0, 180.0, dphi)
    F = np.stack([field(dphis, b, R, w, z0, z1, dist, step) for b in betas])
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, betas=betas, dphis=dphis, F=F)
    return betas, dphis, F


class Plate:
    """표에서 값을 꺼내는 도우미. look(beta, dphi) -> 복소 전기장 비"""
    def __init__(self, **kw):
        self.betas, self.dphis, self.F = table(**kw)
        self.db = self.betas[1] - self.betas[0]; self.dp = self.dphis[1] - self.dphis[0]

    def look(self, beta, dphi):
        beta = np.clip(np.asarray(beta, dtype=float), -90.0, 90.0)
        fb = (beta + 90.0) / self.db
        i0 = np.clip(np.floor(fb).astype(int), 0, len(self.betas) - 2); wb = fb - i0
        fp = ((np.asarray(dphi, dtype=float) + 180.0) % 360.0) / self.dp
        j0 = np.floor(fp).astype(int) % len(self.dphis); j1 = (j0 + 1) % len(self.dphis); wp = fp - np.floor(fp)
        F = self.F
        return ((1 - wb) * ((1 - wp) * F[i0, j0] + wp * F[i0, j1]) + wb * ((1 - wp) * F[i0 + 1, j0] + wp * F[i0 + 1, j1]))

    def level(self, beta, dphi):
        return 20 * np.log10(np.maximum(np.abs(self.look(beta, dphi)), 1e-6))


if __name__ == "__main__":
    import time
    t = time.time(); P = Plate(); print("표 만들기 %.1f초, 크기 %s" % (time.time() - t, P.F.shape))
    for b in (0, 15, 30, 36, 45, -15, -30, -36, -45, 60):
        lv = P.level(b, P.dphis)
        i = np.argmin(lv)
        print("면에서 %+3d도 벗어남: 가운데 %.1f dB, 가장 깊은 곳 %.1f dB (%+d도), 판이 반대편일 때 %+.1f dB" % (b, P.level(b, 0.0), lv[i], P.dphis[i], P.level(b, 180.0)))
