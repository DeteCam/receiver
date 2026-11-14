"""NEC2 (PyNEC, 모멘트법) 로 수신기의 전파 모형을 만드는 도우미. 단위는 mm 로 받아 m 로 바꿔 넣는다.

금속 면은 철망(와이어 그리드)으로 바꾼다. 칸 크기 d 인 철망의 선 반지름은 d / (2 pi) (같은 넓이 규칙).
"""
import numpy as np
from PyNEC import nec_context

C0 = 299.792458          # mm * GHz


class Model:
    def __init__(self, f_ghz=2.437, cell=10.0):
        self.f = f_ghz; self.lam = C0 / f_ghz; self.cell = cell
        self.ctx = nec_context(); self.geo = self.ctx.get_geometry()
        self.tag = 0; self.nseg = 0; self.parts = {}; self._seen = set(); self.feed = None; self.rec = []

    # ---- 선 하나 (mm). 이미 넣은 선과 똑같은 것은 건너뛴다 (맞닿은 면의 모서리)
    def wire(self, p, q, rad, nseg=None, name=None):
        p = np.asarray(p, float); q = np.asarray(q, float); L = np.linalg.norm(q - p)
        if L < 1e-6:
            return None
        key = tuple(sorted([tuple(np.round(p, 3)), tuple(np.round(q, 3))]))
        if key in self._seen:
            return None
        self._seen.add(key)
        n = nseg or max(1, int(np.ceil(L / self.cell)))
        self.tag += 1
        self.rec.append((p, q, rad, name or ""))
        self.geo.wire(self.tag, n, p[0] / 1e3, p[1] / 1e3, p[2] / 1e3, q[0] / 1e3, q[1] / 1e3, q[2] / 1e3, rad / 1e3, 1.0, 1.0)
        self.nseg += n
        if name:
            self.parts.setdefault(name, []).append(self.tag)
        return self.tag

    # ---- 네모난 금속 면: 한 귀 o, 두 변 벡터 u, v (mm)
    def sheet(self, o, u, v, name=None, cell=None):
        o = np.asarray(o, float); u = np.asarray(u, float); v = np.asarray(v, float); d = cell or self.cell
        nu = max(1, int(round(np.linalg.norm(u) / d))); nv = max(1, int(round(np.linalg.norm(v) / d)))
        rad = max(np.linalg.norm(u) / nu, np.linalg.norm(v) / nv) / (2 * np.pi)
        for i in range(nu + 1):
            for j in range(nv):
                self.wire(o + u * i / nu + v * j / nv, o + u * i / nu + v * (j + 1) / nv, rad, 1, name)
        for j in range(nv + 1):
            for i in range(nu):
                self.wire(o + u * i / nu + v * j / nv, o + u * (i + 1) / nu + v * j / nv, rad, 1, name)

    # ---- 금속 상자 (여섯 면)
    def box(self, x0, x1, y0, y1, z0, z1, name=None, cell=None, T=None):
        f = (lambda p: p) if T is None else (lambda p: T[:3, :3] @ np.asarray(p, float) + T[:3, 3])
        fv = (lambda p: np.asarray(p, float)) if T is None else (lambda p: T[:3, :3] @ np.asarray(p, float))
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        for o, u, v in (((x0, y0, z0), (dx, 0, 0), (0, dy, 0)), ((x0, y0, z1), (dx, 0, 0), (0, dy, 0)), ((x0, y0, z0), (dx, 0, 0), (0, 0, dz)),
                        ((x0, y1, z0), (dx, 0, 0), (0, 0, dz)), ((x0, y0, z0), (0, dy, 0), (0, 0, dz)), ((x1, y0, z0), (0, dy, 0), (0, 0, dz))):
            self.sheet(f(o), fv(u), fv(v), name, cell)

    # ---- 금속 원통 (옆면 세로줄 + 고리, 위아래 면은 바퀴살). 축은 c0 -> c1
    def cylinder(self, c0, c1, r, name=None, nphi=8, cell=None):
        c0 = np.asarray(c0, float); c1 = np.asarray(c1, float); ax = c1 - c0; L = np.linalg.norm(ax); ax = ax / L
        a = np.cross(ax, [1, 0, 0]); a = a if np.linalg.norm(a) > 0.3 else np.cross(ax, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(ax, a)
        d = cell or self.cell; nl = max(1, int(round(L / d))); ph = np.arange(nphi) * 2 * np.pi / nphi
        ring = [r * (np.cos(t) * a + np.sin(t) * b) for t in ph]; rad = min(2 * np.pi * r / nphi, L / nl) / (2 * np.pi)
        for k in range(nl + 1):
            c = c0 + ax * L * k / nl
            for i in range(nphi):
                self.wire(c + ring[i], c + ring[(i + 1) % nphi], rad, 1, name)
                if k < nl:
                    self.wire(c + ring[i], c + ax * L / nl + ring[i], rad, 1, name)
        for c in (c0, c1):
            for i in range(nphi):
                self.wire(c, c + ring[i], rad, max(1, int(np.ceil(r / d))), name)

    def polyline(self, pts, rad=0.4, name=None):
        for p, q in zip(pts[:-1], pts[1:]):
            self.wire(p, q, rad, None, name)

    # ---- 점검: 끝점을 같이 쓰지 않는데도 서로 닿는 선 (NEC 가 받아 주지 않는다)
    def clashes(self, margin=0.0, limit=12):
        P = np.array([r[0] for r in self.rec]); Q = np.array([r[1] for r in self.rec]); rad = np.array([r[2] for r in self.rec]); out = []
        for i in range(len(P)):
            d1 = Q[i] - P[i]; d2 = Q[i + 1:] - P[i + 1:]; r = P[i] - P[i + 1:]
            a = d1 @ d1; e = np.einsum("ij,ij->i", d2, d2); f = np.einsum("ij,ij->i", d2, r); c = r @ d1; b = d2 @ d1
            den = a * e - b * b
            s = np.where(den > 1e-9, np.clip((b * f - c * e) / np.where(den > 1e-9, den, 1), 0, 1), 0.0)
            t = np.clip((b * s + f) / e, 0, 1); s = np.clip((b * t - c) / a, 0, 1)
            dist = np.linalg.norm((P[i] + s[:, None] * d1) - (P[i + 1:] + t[:, None] * d2), axis=1)
            share = (np.linalg.norm(P[i] - P[i + 1:], axis=1) < 1e-3) | (np.linalg.norm(P[i] - Q[i + 1:], axis=1) < 1e-3) | (np.linalg.norm(Q[i] - P[i + 1:], axis=1) < 1e-3) | (np.linalg.norm(Q[i] - Q[i + 1:], axis=1) < 1e-3)
            for j in np.where((dist < rad[i] + rad[i + 1:] + margin) & ~share)[0]:
                out.append((float(dist[j]), self.rec[i][3], self.rec[i + 1 + j][3], np.round(self.rec[i][0], 1), np.round(self.rec[i][1], 1), np.round(self.rec[i + 1 + j][0], 1), np.round(self.rec[i + 1 + j][1], 1)))
        return sorted(out, key=lambda x: x[0])[:limit]

    # ---- 풀기: 먹임 선(tag, seg)에 1V 를 걸고 먼 곳의 전기장을 구한다
    def solve(self, th, ph):
        """th, ph: 고른 간격의 각도 배열(도). 돌려주는 것: E_theta, E_phi (복소수, [th, ph]), 입력 임피던스"""
        ctx = self.ctx
        ctx.geometry_complete(0)
        ctx.set_extended_thin_wire_kernel(True)
        ctx.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
        for tag, seg in (self.feed if isinstance(self.feed, list) else [self.feed]):          # 먹임이 여러 개면 모두 같은 위상으로 건다
            ctx.ex_card(0, tag, seg, 0, 1.0, 0, 0, 0, 0, 0)
        ctx.fr_card(0, 1, self.f * 1000.0, 0)
        dth = th[1] - th[0] if len(th) > 1 else 0.0; dph = ph[1] - ph[0] if len(ph) > 1 else 0.0
        ctx.rp_card(0, len(th), len(ph), 0, 0, 0, 0, float(th[0]), float(ph[0]), float(dth), float(dph), 0.0, 0.0)
        rp = ctx.get_radiation_pattern(0)
        # NEC 는 phi 를 바깥 고리로 돈다: 받은 배열은 [phi][theta] 순서다 (gain 표와 견줘 확인했다)
        et = np.asarray(rp.get_e_theta()).reshape(len(ph), len(th)).T; ep = np.asarray(rp.get_e_phi()).reshape(len(ph), len(th)).T
        z = ctx.get_input_parameters(0).get_impedance()[0]
        return et, ep, z
