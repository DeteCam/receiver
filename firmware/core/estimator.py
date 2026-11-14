"""스캔 여러 번의 곡선에서 카메라 방향(방위각, 높이각)을 구한다.

스캔 하나 = 머리를 (받침 각 yaw, 끄덕임 각 tilt) 로 두고 판을 한 바퀴 돌리며 잰 세기 곡선.
방향 후보마다 '그 방향에 카메라가 있다면 각 스캔에서 보였을 곡선'을 모형 표에서 꺼내고,
잰 곡선과 가장 잘 맞는 후보를 고른다 (성긴 격자 -> 둘레를 촘촘히).
실제 측정값에도 그대로 쓸 수 있다: curves = [(판 각 배열(도), 세기 배열(dB)), ...], frames = [(yaw, tilt), ...]
"""
import numpy as np
from model import Plate


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


def azel(c):
    c = np.asarray(c)
    return np.degrees(np.arctan2(c[..., 1], c[..., 0])), np.degrees(np.arcsin(np.clip(c[..., 2], -1, 1)))


def frame(yaw, tilt):
    """(회전축, 머리 X축, 머리 Y축) 을 세계 좌표로. 판 각 phi 는 머리 X축에서 잰다."""
    y, t = np.radians(yaw), np.radians(tilt)
    a = np.array([np.sin(t) * np.cos(y), np.sin(t) * np.sin(y), np.cos(t)])
    ex = np.array([-np.sin(y), np.cos(y), 0.0])
    return a, ex, np.cross(a, ex)


def to_scan(c, fr):
    """방향 c 가 이 스캔에서 보이는 (면 안의 각 psi, 면에서 벗어난 각 beta)"""
    a, ex, ey = fr
    return np.degrees(np.arctan2(c @ ey, c @ ex)), np.degrees(np.arcsin(np.clip(c @ a, -1, 1)))


def sphere(n=6000, el_min=-88.0):
    """구 위에 고르게 뿌린 방향 (피보나치 격자)"""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n; r = np.sqrt(1 - z * z); th = np.pi * (1 + 5 ** 0.5) * i
    c = np.stack([r * np.cos(th), r * np.sin(th), z], -1)
    return c[c[:, 2] >= np.sin(np.radians(el_min))]


def huber(r, d=3.0):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


class Solver:
    def __init__(self, plate=None, n_grid=6000):
        self.P = plate or Plate()
        self.G = sphere(n_grid)
        self._cache = {}

    def templates(self, C, fr, phi):
        """방향 C (N,3) 에 대한 곡선 (N, len(phi)), 가운데값을 0 으로 맞춘 dB"""
        psi, beta = to_scan(C, fr)
        T = self.P.level(beta[:, None], phi[None, :] - psi[:, None])
        return T - np.median(T, axis=1, keepdims=True)

    def cost(self, C, curves, frames):
        tot = np.zeros(len(C))
        for (phi, lv), (yaw, tilt) in zip(curves, frames):
            ok = np.isfinite(lv)
            key = (round(yaw, 3), round(tilt, 3), len(phi), float(phi[0]), id(C))
            if C is self.G:
                if key not in self._cache:
                    self._cache[key] = self.templates(C, frame(yaw, tilt), phi)
                T = self._cache[key]
            else:
                T = self.templates(C, frame(yaw, tilt), phi)
            m = lv - np.nanmedian(lv)
            tot += huber(T[:, ok] - m[None, ok]).sum(1)
        return tot

    def solve(self, curves, frames, n_seed=4, span=7.0, step=1.0):
        """가장 잘 맞는 방향. (방향 벡터, 비용, 둘째 후보와의 비용 차이) 를 돌려준다."""
        cost = self.cost(self.G, curves, frames)
        order = np.argsort(cost)
        seeds = []
        for i in order[:60]:                                   # 서로 12도 넘게 떨어진 후보 몇 개
            if all(np.degrees(np.arccos(np.clip(self.G[i] @ s, -1, 1))) > 12 for s in seeds):
                seeds.append(self.G[i])
            if len(seeds) >= n_seed:
                break
        best = None; alts = []
        for s in seeds:
            c0 = s
            for sp, st in ((span, step), (2.0, 0.25)):         # 둘레를 촘촘히 두 번
                az0, el0 = azel(c0)
                d = np.arange(-sp, sp + 1e-9, st)
                EL = np.clip(el0 + d, -89.5, 89.5)
                AZ = az0 + d / max(np.cos(np.radians(el0)), 0.15)
                A, E = np.meshgrid(AZ, EL)
                C = unit(A.ravel(), E.ravel())
                cc = self.cost(C, curves, frames)
                j = int(np.argmin(cc)); c0 = C[j]; v = float(cc[j])
            alts.append((v, c0))
            if best is None or v < best[0]:
                best = (v, c0)
        alts.sort(key=lambda x: x[0])
        gap = alts[1][0] - alts[0][0] if len(alts) > 1 else np.inf
        return best[1], best[0], gap


def focus_axis(c, tilt=54.7356):
    """카메라 방향 c 가 판이 도는 면에 정확히 들어오게 하는 (yaw, tilt). 기울임을 될 수 있으면 그대로 둔다."""
    az, el = azel(c)
    t = max(tilt, min(abs(float(el)) + 0.5, 90.0))             # 높이각이 기울임보다 크면 그만큼 더 눕힌다
    x = -np.tan(np.radians(el)) / np.tan(np.radians(t))
    d = np.degrees(np.arccos(np.clip(x, -1, 1)))
    return [((float(az) + d + 180) % 360 - 180, t), ((float(az) - d + 180) % 360 - 180, t)]
