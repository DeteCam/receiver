"""Small z-buffer renderer (numpy) for the DeteCam assembly animation and the part pictures.
Units are millimetres, z is up.  Triangles are flat shaded, lines are depth tested."""
import numpy as np
from PIL import Image


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-12)


class Camera:
    def __init__(self, eye, target, fov_deg=28.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.fov = float(fov_deg)

    def basis(self):
        f = _norm(self.target - self.eye)
        r = _norm(np.cross(f, np.array([0.0, 0.0, 1.0])))
        u = np.cross(r, f)
        return r, u, f


def orbit(target, dist, az_deg, el_deg, fov_deg=28.0):
    a, e = np.radians(az_deg), np.radians(el_deg)
    eye = np.asarray(target, dtype=np.float64) + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return Camera(eye, target, fov_deg)


class Mesh:
    """Triangles with one colour per triangle, plus optional outline segments."""
    def __init__(self):
        self.V, self.F, self.C, self.E = [], [], [], []
        self.n = 0

    def add(self, verts, faces, color, edges=()):
        verts = np.asarray(verts, dtype=np.float64)
        self.V.append(verts)
        f = np.asarray(faces, dtype=np.int64) + self.n
        self.F.append(f)
        self.C.append(np.tile(np.asarray(color, dtype=np.float32), (len(f), 1)))
        for a, b in edges:
            self.E.append((self.n + a, self.n + b))
        self.n += len(verts)
        return self

    def arrays(self):
        if not self.V:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)
        return (np.vstack(self.V), np.vstack(self.F), np.vstack(self.C),
                np.asarray(self.E, dtype=np.int64).reshape(-1, 2))


_BOX_F = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
_BOX_E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box(mesh, lo, hi, color, edges=True):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return mesh.add(v, _BOX_F, color, _BOX_E if edges else ())


def cyl(mesh, p0, p1, rad, color, n=14, edges=False):
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    ax = _norm(p1 - p0)
    a = _norm(np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0]))
    b = np.cross(ax, a)
    t = 2.0 * np.pi * np.arange(n) / n
    ring = rad * (np.cos(t)[:, None] * a[None] + np.sin(t)[:, None] * b[None])
    v = np.vstack([p0 + ring, p1 + ring, p0[None], p1[None]])
    f, e = [], []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
        if edges:
            e += [(i, j), (n + i, n + j)]
    return mesh.add(v, f, color, e)


def transform(V, rot_y_deg=0.0, pivot=(0.0, 0.0, 0.0), shift=(0.0, 0.0, 0.0)):
    V = np.asarray(V, dtype=np.float64)
    if rot_y_deg:
        c, s = np.cos(np.radians(rot_y_deg)), np.sin(np.radians(rot_y_deg))
        p = np.asarray(pivot, dtype=np.float64)
        q = V - p
        V = np.stack([c * q[:, 0] + s * q[:, 2], q[:, 1], -s * q[:, 0] + c * q[:, 2]], axis=1) + p
    return V + np.asarray(shift, dtype=np.float64)


class Renderer:
    def __init__(self, width, height, ss=2, bg_top=(0.96, 0.97, 0.99), bg_bottom=(0.86, 0.89, 0.93)):
        self.w, self.h, self.ss = width * ss, height * ss, ss
        g = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None, None]
        self.bg = (np.asarray(bg_top, dtype=np.float32)[None, None] * (1 - g) + np.asarray(bg_bottom, dtype=np.float32)[None, None] * g)
        self.bg = np.broadcast_to(self.bg, (self.h, self.w, 3)).copy()
        self.light = _norm(np.array([0.35, -0.45, 0.82]))

    def begin(self, cam):
        self.cam = cam
        self.col = self.bg.copy()
        self.z = np.full((self.h, self.w), np.inf, dtype=np.float32)
        self.r, self.u, self.f = cam.basis()
        self.focal = 0.5 * self.h / np.tan(np.radians(cam.fov) / 2.0)

    def project(self, P):
        q = np.asarray(P, dtype=np.float64) - self.cam.eye
        zc = q @ self.f
        zc = np.maximum(zc, 1e-3)
        x = self.w / 2.0 + self.focal * (q @ self.r) / zc
        y = self.h / 2.0 - self.focal * (q @ self.u) / zc
        return x, y, zc

    def screen(self, P):
        """Screen position in output pixels (for 2-D labels)."""
        x, y, z = self.project(np.atleast_2d(P))
        return x / self.ss, y / self.ss

    def draw(self, V, F, C, E=None, outline=(0.10, 0.11, 0.13), shade=True):
        if len(F) == 0:
            return
        x, y, z = self.project(V)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1) + 1e-12
        if shade:
            inten = 0.50 + 0.50 * np.abs((nrm / ln[:, None]) @ self.light)
        else:
            inten = np.ones(len(F))
        colf = (C * inten[:, None]).astype(np.float32)
        X = np.stack([x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]], axis=1)
        Y = np.stack([y[F[:, 0]], y[F[:, 1]], y[F[:, 2]]], axis=1)
        Z = np.stack([z[F[:, 0]], z[F[:, 1]], z[F[:, 2]]], axis=1).astype(np.float32)
        minx = np.clip(np.floor(X.min(1)).astype(int), 0, self.w - 1)
        maxx = np.clip(np.ceil(X.max(1)).astype(int), 0, self.w - 1)
        miny = np.clip(np.floor(Y.min(1)).astype(int), 0, self.h - 1)
        maxy = np.clip(np.ceil(Y.max(1)).astype(int), 0, self.h - 1)
        area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
        zb, cb = self.z, self.col
        for i in np.flatnonzero((np.abs(area) > 1e-6) & (maxx >= minx) & (maxy >= miny)):
            x0, x1, y0, y1 = minx[i], maxx[i], miny[i], maxy[i]
            px = np.arange(x0, x1 + 1, dtype=np.float32)[None, :] + 0.5
            py = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
            ax_, ay_, bx_, by_, cx_, cy_ = X[i, 0], Y[i, 0], X[i, 1], Y[i, 1], X[i, 2], Y[i, 2]
            w0 = (bx_ - px) * (cy_ - py) - (cx_ - px) * (by_ - py)
            w1 = (cx_ - px) * (ay_ - py) - (ax_ - px) * (cy_ - py)
            w2 = (ax_ - px) * (by_ - py) - (bx_ - px) * (ay_ - py)
            ar = area[i]
            if ar < 0:
                m = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            else:
                m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            zz = (w0 * Z[i, 0] + w1 * Z[i, 1] + w2 * Z[i, 2]) / ar
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            m &= zz < sub
            sub[m] = zz[m]
            cb[y0:y1 + 1, x0:x1 + 1][m] = colf[i]
        if E is not None and len(E):
            self.lines(V[E[:, 0]], V[E[:, 1]], outline, width=1.0, bias=0.9)

    def lines(self, P0, P1, color, width=2.0, bias=1.5):
        """Depth-tested segments; width in output pixels."""
        P0, P1 = np.atleast_2d(P0), np.atleast_2d(P1)
        x0, y0, z0 = self.project(P0)
        x1, y1, z1 = self.project(P1)
        rad = max(0.5, 0.5 * width * self.ss)
        k = int(np.ceil(rad))
        oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
        disc = (ox ** 2 + oy ** 2) <= rad * rad + 0.3
        ox, oy = ox[disc], oy[disc]
        col = np.asarray(color, dtype=np.float32)
        for i in range(len(P0)):
            n = int(max(abs(x1[i] - x0[i]), abs(y1[i] - y0[i]))) + 2
            t = np.linspace(0.0, 1.0, n)
            xs = x0[i] + t * (x1[i] - x0[i])
            ys = y0[i] + t * (y1[i] - y0[i])
            zs = z0[i] + t * (z1[i] - z0[i]) - bias
            X = (np.round(xs)[:, None] + ox[None]).astype(int).ravel()
            Y = (np.round(ys)[:, None] + oy[None]).astype(int).ravel()
            Zs = np.repeat(zs, len(ox)).astype(np.float32)
            ok = (X >= 0) & (X < self.w) & (Y >= 0) & (Y < self.h)
            X, Y, Zs = X[ok], Y[ok], Zs[ok]
            vis = Zs <= self.z[Y, X]
            self.col[Y[vis], X[vis]] = col
            self.z[Y[vis], X[vis]] = np.minimum(self.z[Y[vis], X[vis]], Zs[vis])

    def image(self):
        a = np.clip(self.col, 0.0, 1.0)
        if self.ss > 1:
            a = a.reshape(self.h // self.ss, self.ss, self.w // self.ss, self.ss, 3).mean(axis=(1, 3))
        return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "RGB")
