"""Palm-size receiver study: compact arrays / rotating synthetic apertures synthesised from single-point ray-traced path data
(plane-wave model around the node centre; checked against per-element ray tracing for <= 10 cm apertures).

Conventions: global frame, z up. Path data per (camera r, node n): aV, aH (theta/phi-polarised complex gains), tau, th, ph (departure
angles at the node).  An element is a short dipole along unit vector `axis` displaced by `d` from the node centre."""
import numpy as np

C0 = 299792458.0

def udir(th, ph): return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
def that(th, ph): return np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
def phat(th, ph): return np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)

def ring(radius, n, normal=(0, 0, 1.0), phase0=0.0):
    """n positions on a circle of given radius in the plane with the given normal. Returns (pos (n,3), axis (n,3) = normal)."""
    nz = np.asarray(normal, float); nz = nz / np.linalg.norm(nz)
    e1 = np.cross(nz, [0, 0, 1.0]);  e1 = np.array([1.0, 0, 0]) if np.linalg.norm(e1) < 1e-6 else e1 / np.linalg.norm(e1)
    e2 = np.cross(nz, e1)
    a = phase0 + 2 * np.pi * np.arange(n) / n
    pos = radius * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    return pos, np.tile(nz, (n, 1))

def tilt_normal(bore_h, tilt_deg):
    """Ring axis tilted back from vertical by tilt_deg, away from the horizontal viewing direction bore_h."""
    b = np.array([bore_h[0], bore_h[1], 0.0]); b = b / np.linalg.norm(b); t = np.radians(tilt_deg)
    return np.cos(t) * np.array([0, 0, 1.0]) - np.sin(t) * b

class Band:
    def __init__(self, band, bw=17.5e6, k=16):
        self.fc = 2.437e9 if band == "2g4" else 5.22e9; self.k0 = 2 * np.pi * self.fc / C0; self.K = k
        self.df = np.linspace(-bw / 2, bw / 2, k); self.band = band

def element_gain(aV, aH, th, ph, axis):
    """Response of a short dipole along `axis` (3,) or (M,3) to each path. Returns (P,) or (M,P)."""
    tv, pv = that(th, ph), phat(th, ph)
    axis = np.atleast_2d(axis)
    g = aV[None] * (axis @ tv.T) + aH[None] * (axis @ pv.T)
    return -g                                                    # sign: theta_hat . z = -sin(theta)

def synth(paths, r, n, pos, axis, B, block=None):
    """CSI (M, K) of the array (pos, axis) at node n for camera r."""
    aV, aH = paths["a"][r, n, 0], paths["a"][r, n, 1]; th, ph, tau = paths["th"][r, n], paths["ph"][r, n], paths["tau"][r, n]
    u = udir(th, ph); g = element_gain(aV, aH, th, ph, axis)                           # (M, P)
    st = np.exp(1j * B.k0 * (pos @ u.T)) * g
    return st @ np.exp(-2j * np.pi * tau[:, None] * B.df[None])

def observe(H, rng, cal_deg=4.5, cal_db=0.4, rel_noise_db=-30.0):
    """Per-position gain/phase residual, thermal + relative noise, random common phase and delay (packet timing is unknown)."""
    M, K = H.shape
    c = 10 ** (rng.normal(0, cal_db, M) / 20) * np.exp(1j * np.radians(rng.normal(0, cal_deg, M)))
    p = np.mean(np.abs(H) ** 2); sig2 = max(10 ** (-109 / 10), p * 10 ** (rel_noise_db / 10))
    return c[:, None] * H + np.sqrt(sig2 / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))

def los_info(paths, r, n, cam, node_pos):
    """(exists, power fraction of the direct path, V-pol)."""
    tau = paths["tau"][r, n]; a = paths["a"][r, n]; pw = (np.abs(a) ** 2).sum(0)
    tg = np.linalg.norm(cam - node_pos) / C0
    m = (np.abs(tau - tg) < 0.06e-9) & (pw > 0)
    tot = pw.sum()
    return bool(m.any()), float(pw[m].sum() / tot) if tot > 0 else 0.0

# ----------------------------------------------------------------------------- direction grids
def dir_grid(az_step=2.0, el_lo=-75, el_hi=45, el_step=5.0):
    az = np.radians(np.arange(0, 360, az_step)); el = np.radians(np.arange(el_lo, el_hi + 1e-6, el_step))
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    U = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    return U, AZ.ravel(), EL.ravel()

def steer(U, pos, axis, B):
    """Far-field steering (G, M) for theta-polarised unit field (vertical-ish emitters). Not normalised."""
    th = np.arccos(np.clip(U[:, 2], -1, 1)); ph = np.arctan2(U[:, 1], U[:, 0])
    g = -(that(th, ph) @ np.atleast_2d(axis).T)                                         # (G, M)
    return np.exp(1j * B.k0 * (U @ pos.T)) * g

def az_err(u_est, u_true):
    d = np.degrees(np.arctan2(u_est[1], u_est[0]) - np.arctan2(u_true[1], u_true[0]))
    return abs((d + 180) % 360 - 180)
def ang_err(u_est, u_true): return float(np.degrees(np.arccos(np.clip(u_est @ u_true, -1, 1))))

# ----------------------------------------------------------------------------- single-node direction estimators
def bartlett_dir(X, A):
    """X (M,K), A (G,M). Non-coherent over subcarriers."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return np.sum(np.abs(An.conj() @ X) ** 2, axis=1)

def capon_dir(X, A, load=0.05):
    M = X.shape[0]; R = X @ X.conj().T / X.shape[1]; Ri = np.linalg.inv(R + load * np.real(np.trace(R)) / M * np.eye(M))
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return 1.0 / np.maximum(np.real(np.einsum("gm,mk,gk->g", An.conj(), Ri, An)), 1e-12)

class Nomp:
    """Joint direction-delay path extraction (Newtonised-OMP style: greedy pick, local refinement, joint amplitude re-fit, cyclic refinement)."""
    def __init__(self, pos, axis, B, U, taus=np.arange(-40e-9, 100e-9 + 1e-12, 4e-9)):
        self.pos, self.axis, self.B, self.U = pos, axis, B, U
        self.A = steer(U, pos, axis, B); self.An2 = np.sum(np.abs(self.A) ** 2, axis=1) + 1e-12
        self.taus = taus; self.F = np.exp(-2j * np.pi * B.df[:, None] * taus[None])     # (K, T)

    def atom(self, u, tau):
        a = steer(u[None], self.pos, self.axis, self.B)[0]; f = np.exp(-2j * np.pi * self.B.df * tau)
        return a[:, None] * f[None]                                                     # (M, K)

    def _refine(self, R, u, tau, rounds=2):
        best = (np.abs(np.vdot(self.atom(u, tau), R)) ** 2 / (np.sum(np.abs(self.atom(u, tau)) ** 2) + 1e-12), u, tau)
        saz, sel, st = np.radians(1.0), np.radians(2.5), 2e-9
        for _ in range(rounds):
            az0 = np.arctan2(best[1][1], best[1][0]); el0 = np.arcsin(np.clip(best[1][2], -1, 1))
            for daz in (-saz, 0, saz):
                for de in (-sel, 0, sel):
                    el = np.clip(el0 + de, -1.45, 1.45); az = az0 + daz
                    uu = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                    a = steer(uu[None], self.pos, self.axis, self.B)[0]; y = a.conj() @ R            # (K,)
                    for dt in (-st, 0, st):
                        tt = best[2] + dt; c = np.abs(np.sum(y * np.exp(2j * np.pi * self.B.df * tt))) ** 2 / ((np.sum(np.abs(a) ** 2) + 1e-12) * len(self.B.df))
                        if c > best[0]: best = (c, uu, tt)
            saz, sel, st = saz / 2, sel / 2, st / 2
        return best[1], best[2]

    def run(self, X, L=3, stop_db=-17.0, cyc=2):
        R = X.copy(); paths = []; K = X.shape[1]; e0 = np.sum(np.abs(X) ** 2)
        for l in range(L):
            C = (self.A.conj() @ R) @ self.F.conj()                                     # (G, T)
            P = np.abs(C) ** 2 / (self.An2[:, None] * K)
            g, t = np.unravel_index(int(np.argmax(P)), P.shape)
            u, tau = self._refine(R, self.U[g], self.taus[t])
            paths.append([u, tau])
            for _ in range(cyc if l > 0 else 0):                                        # cyclic refinement
                for i in range(len(paths)):
                    S = np.stack([self.atom(p[0], p[1]).ravel() for p in paths], 1); gam = np.linalg.lstsq(S, X.ravel(), rcond=None)[0]
                    Ri = X - sum(gam[j] * self.atom(paths[j][0], paths[j][1]) for j in range(len(paths)) if j != i)
                    paths[i] = list(self._refine(Ri, paths[i][0], paths[i][1], rounds=1))
            S = np.stack([self.atom(p[0], p[1]).ravel() for p in paths], 1); gam = np.linalg.lstsq(S, X.ravel(), rcond=None)[0]
            R = X - (S @ gam).reshape(X.shape)
            pw = np.abs(gam) ** 2 * np.sum(np.abs(S) ** 2, axis=0)
            if l > 0 and 10 * np.log10(pw[-1] / (pw.max() + 1e-30) + 1e-30) < stop_db:
                paths.pop(); S = S[:, :-1]; gam = np.linalg.lstsq(S, X.ravel(), rcond=None)[0]; R = X - (S @ gam).reshape(X.shape); pw = np.abs(gam) ** 2 * np.sum(np.abs(S) ** 2, axis=0); break
        return [dict(u=p[0], tau=p[1], gam=g, pw=w) for p, g, w in zip(paths, gam, pw)], float(np.sum(np.abs(R) ** 2) / e0), S, gam

def pick_direct(paths, rel_db=-10.0, tie_ns=5.0, tie_db=6.0):
    """Earliest path among those within rel_db of the strongest; if a clearly stronger path follows within tie_ns, prefer it."""
    pw = np.array([p["pw"] for p in paths]); ok = [i for i in range(len(paths)) if 10 * np.log10(pw[i] / pw.max()) >= rel_db]
    i0 = min(ok, key=lambda i: paths[i]["tau"])
    for i in ok:
        if i != i0 and 0 <= (paths[i]["tau"] - paths[i0]["tau"]) * 1e9 <= tie_ns and 10 * np.log10(pw[i] / pw[i0]) >= tie_db: i0 = i
    return i0

# ----------------------------------------------------------------------------- position maps for multi-node fusion
def sph_steer_el(grid, node_pos, pos, axis, B):
    """Steering towards every grid point (C, M) (far field around the node centre; dipole pattern for a vertical emitter)."""
    v = grid - node_pos; d = np.linalg.norm(v, axis=1, keepdims=True); U = v / d
    return steer(U, pos, axis, B)

def capon_ll(X, A, load=0.08):
    p = capon_dir(X, A, load); return np.log(p / p.max() + 1e-6)
def bartlett_ll(X, A):
    p = bartlett_dir(X, A); return np.log(p / p.max() + 1e-6)

class MapModel:
    """Frequency-coherent map-aware likelihood: direct path + images on the room planes (first order, optionally second order on
    perpendicular plane pairs), free complex gains ridge-regularised by nominal path strength, unknown common delay, mixture with a
    'direct path blocked' hypothesis.  refl = scalar or dict {(axis, value): amplitude reflectivity} (e.g. from in-situ calibration)."""
    def __init__(self, grid, node_pos, pos, axis, B, dims, refl=0.5, second=False, lam=0.08, tc=np.arange(-45e-9, 45e-9 + 1e-12, 3e-9)):
        L, W, H = dims; self.B = B; self.tc = tc; C = len(grid)
        planes = [(0, 0.0), (0, L), (1, 0.0), (1, W), (2, H), (2, 0.0)]
        rho1 = [refl[p] if isinstance(refl, dict) else refl for p in planes]
        imgs = [grid] + [self._mirror(grid, ax, val) for ax, val in planes]; rho = [1.0] + rho1
        if second:
            for i in range(6):
                for j in range(i + 1, 6):
                    if planes[i][0] == planes[j][0]: continue
                    imgs.append(self._mirror(self._mirror(grid, *planes[i]), *planes[j])); rho.append(rho1[i] * rho1[j])
        R = len(imgs); M = len(pos)
        self.S = np.zeros((C, R, M), np.complex64); self.dt = np.zeros((C, R)); self.w = np.zeros((C, R))
        d0 = np.linalg.norm(grid - node_pos, axis=1)
        for i, g in enumerate(imgs):
            v = g - node_pos; d = np.linalg.norm(v, axis=1); U = v / d[:, None]
            self.S[:, i] = steer(U, pos, axis, B); self.dt[:, i] = (d - d0) / C0; self.w[:, i] = (rho[i] * d0 / d) ** 2
        Fq = np.exp(-2j * np.pi * B.df[None, None, :] * self.dt[:, :, None])                              # (C, R, K)
        Gs = np.einsum("crm,csm->crs", self.S.conj(), self.S); Gf = np.einsum("crk,csk->crs", Fq.conj(), Fq)
        self.G = Gs * Gf; self.Fq = Fq; self.lam = lam
        self.Ftc = np.exp(2j * np.pi * B.df[:, None] * tc[None])                                         # (K, T)

    @staticmethod
    def _mirror(g, ax, val):
        o = g.copy(); o[:, ax] = 2 * val - o[:, ax]; return o

    def score(self, X, use=None, return_gain=False):
        """Fraction of ||X||^2 explained by the model at each candidate (maximised over the common delay)."""
        idx = np.arange(self.S.shape[1]) if use is None else np.asarray(use); n = len(idx)
        S = self.S[:, idx]; Fq = self.Fq[:, idx]; G = self.G[:, idx][:, :, idx]; w = self.w[:, idx]
        b = (np.einsum("crm,mk->crk", S.conj(), X) * Fq.conj()) @ self.Ftc                                # (C, R, T)
        tr = np.real(np.einsum("crr->c", G)) / n
        Greg = G + ((self.lam * tr)[:, None] / np.maximum(w, 1e-4))[:, :, None] * np.eye(n)[None]
        sol = np.linalg.solve(Greg, b)
        q = np.real(np.sum(b.conj() * sol, axis=1))                                                      # (C, T)
        t = np.argmax(q, axis=1); sc = np.clip(q[np.arange(len(t)), t] / (np.sum(np.abs(X) ** 2) + 1e-30), 0, 0.999)
        if return_gain: return sc, sol[np.arange(len(t)), :, t]
        return sc

    def ll(self, X, n_eff=10.0, w_vis=0.7):
        s1 = self.score(X); s0 = self.score(X, use=np.arange(1, self.S.shape[1]))
        l1 = -n_eff * np.log(1 - s1 + 2e-3); l0 = -n_eff * np.log(1 - s0 + 2e-3)
        return np.logaddexp(np.log(w_vis) + l1, np.log(1 - w_vis) + l0)

    def ll_direct_only(self, X, n_eff=10.0):
        return -n_eff * np.log(1 - self.score(X, use=[0]) + 2e-3)


def room_mask(U, node_pos, dims, reach=0.5):
    """Directions that stay inside the room for at least `reach` metres (the emitter cannot be behind a wall)."""
    p = node_pos[None] + reach * U; L, W, H = dims
    return (p[:, 0] > 0) & (p[:, 0] < L) & (p[:, 1] > 0) & (p[:, 1] < W) & (p[:, 2] > 0) & (p[:, 2] < H)

def az_spread(ll, grid, node_pos, temp=1.0):
    """Circular spread (deg) of the azimuth posterior implied by a log-likelihood map: small = confident."""
    p = np.exp((ll - ll.max()) / temp); p /= p.sum(); v = grid - node_pos; az = np.arctan2(v[:, 1], v[:, 0])
    Rm = np.abs(np.sum(p * np.exp(1j * az))); return float(np.degrees(np.sqrt(max(-2 * np.log(max(Rm, 1e-9)), 0.0))))
