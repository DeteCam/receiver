"""Direction & suspect-zone estimators for DeteCam V1 switch-ring evaluation.
STRICT SEPARATION (AGENTS.md 5.1.1, receiver AGENTS.md #3):
  This module NEVER imports gen_scenes.py or channel_sim.py.
  It ONLY consumes:
    - Measured CSI matrix X (M, K)
    - Noisy user-provided room box dims_est = (L, W, H), node_pos_est, heading_est
    - Optional noisy user-visible major reflector planes (blackboard / large wall mirror)
    - 1-time calibrated free-space/box NEC2 steering function (turntable calibration table)
"""
import numpy as np

C0 = 299792458.0

# SPEC-002+ Rev.2 settings (docs/Rev2_검증_보고서.md). Defaults of the functions below keep the
# pre-registered V1 behaviour so that RESULTS_20261125.md stays reproducible; pass these to use Rev.2.
REV2_ESTIMATOR_KWARGS = dict(exact_manifold=True, tc_step_ns="auto")
REV2_FUSION_KWARGS = dict(confidence_weighted=False)  # equal-weight L0 + L1


def build_calibrated_manifold(U, cal_pat, heading_rad=0.0, include_ref=True, exact=False):
    """Evaluate 1-time calibrated ring manifold (G, M) for unit direction vectors U (G, 3).
    Note: For a switched receiver measuring H_m / H_ref (or weighted [H_1/H_ref, ..., H_N/H_ref, 1.0]),
    the exact spatial steering vector at direction U is proportional to [a_1(u), ..., a_N(u), a_ref(u)]
    (or divided by a_ref(u) when include_ref=False).
    exact=True (Rev.2) returns [a_1(u), ..., a_N(u), a_ref(u)] itself. The legacy form [et/rt, 1] rescales each
    direction by 1/a_ref(u), which is harmless for one path but distorts the relative amplitude between the direct
    path and its mirror images when several images are combined in one Gram matrix (RingMapEstimator).
    """
    th = np.arccos(np.clip(U[:, 2], -1.0, 1.0))
    ph = np.arctan2(U[:, 1], U[:, 0]) - heading_rad
    et, rt = cal_pat.element_responses(th, ph, aV=None, aH=None)  # et (N, G), rt (G,)
    if include_ref and exact:
        A = np.vstack([et, rt[None, :]])
    elif include_ref:
        # Stack [et / rt, 1.0] * |rt| -> proportional to [et, rt] while matching ratio observation scaling
        A = np.vstack([et / (rt[None, :] + 1e-12), np.ones((1, len(U)), dtype=np.complex128)])
    else:
        A = et / (rt[None, :] + 1e-12)
    return A.T.astype(np.complex64)  # (G, M)


def room_interior_mask(U, node_pos, dims, reach=0.45):
    L, W, H = dims
    p = node_pos[None, :] + reach * U
    return (p[:, 0] > 0.0) & (p[:, 0] < L) & (p[:, 1] > 0.0) & (p[:, 1] < W) & (p[:, 2] > 0.0) & (p[:, 2] < H)


def dir_grid(az_step=2.0, el_lo=-65.0, el_hi=35.0, el_step=5.0):
    az = np.radians(np.arange(0.0, 360.0, az_step))
    el = np.radians(np.arange(el_lo, el_hi + 1e-6, el_step))
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    U = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], axis=-1).reshape(-1, 3)
    return U, AZ.ravel(), EL.ravel()


def bartlett_azimuth(X, node_pos_est, dims_est, cal_pat, heading_est=0.0, include_ref=False):
    """Baseline non-coherent Bartlett beamforming (no map, no learning)."""
    U, AZ, _ = dir_grid()
    mk = room_interior_mask(U, np.clip(node_pos_est, 0.05, np.asarray(dims_est) - 0.05), dims_est)
    if not np.any(mk):
        mk[:] = True
    Um, AZm = U[mk], AZ[mk]
    A = build_calibrated_manifold(Um, cal_pat, heading_rad=heading_est, include_ref=include_ref)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    p = np.sum(np.abs(An.conj() @ X) ** 2, axis=1)
    return float(AZm[int(np.argmax(p))])


def make_room_grid(dims_est, max_xy_pts=260):
    """3-height-layer 3D candidate grid with fine horizontal angular resolution (~3-4 deg)
    tailored to horizontal circular arrays (low vertical resolution per ai SPEC-002 section 5)."""
    L, W, H = dims_est
    step_xy = max(0.30, np.sqrt((L * W) / float(max_xy_pts)))
    xs = np.arange(0.15, L - 0.10, step_xy)
    ys = np.arange(0.15, W - 0.10, step_xy)
    zs = np.array([0.25 * H, 0.50 * H, 0.80 * H], dtype=np.float64)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float64)


class RingMapEstimator:
    """Frequency-coherent map-aware likelihood estimator for switch-ring arrays.
    Supports:
      - Standard SPEC-002 mode: 6 room walls (1st+2nd order images), Gaussian ridge likelihood, 8 elements
      - Upgraded SPEC-002+ mode: 9 elements (8+center ref), full-wave calibrated NEC2 manifold,
        user-visible major metal reflectors (blackboard/mirror/locker face), Huber/Student-t robust
        subcarrier weighting against dense multipath clutter (DMC), and confidence-weighted 2-node fusion.
    """
    def __init__(
        self,
        grid,
        node_pos_est,
        dims_est,
        cal_pat,
        df_hz,
        heading_est=0.0,
        include_ref=True,
        reflector_faces=(),
        refl_wall=0.50,
        second_order=True,
        lam=0.08,
        robust_huber=True,
        huber_delta=0.35,
        exact_manifold=False,
        tc_step_ns=8.0,
    ):
        self.grid = grid
        self.node_pos = np.asarray(node_pos_est, dtype=np.float64)
        self.df = np.asarray(df_hz, dtype=np.float64)
        self.lam = float(lam)
        self.robust_huber = bool(robust_huber)
        self.huber_delta = float(huber_delta)

        L, W, H = dims_est
        planes = [(0, 0.0), (0, L), (1, 0.0), (1, W), (2, H), (2, 0.0)]
        imgs = [grid] + [self._mirror(grid, ax, val) for ax, val in planes]
        rhos = [1.0] + [refl_wall] * 6
        valid = [np.ones(len(grid), dtype=bool)] * len(imgs)

        if second_order:
            for i in range(6):
                for j in range(i + 1, 6):
                    if planes[i][0] == planes[j][0]:
                        continue
                    imgs.append(self._mirror(self._mirror(grid, *planes[i]), *planes[j]))
                    rhos.append(refl_wall * refl_wall)
                    valid.append(np.ones(len(grid), dtype=bool))

        # Add 1-2 user-visible major metal reflector faces (blackboard / mirror / locker wall)
        for ax, val, lo, hi, rf in reflector_faces:
            g = self._mirror(grid, ax, val)
            den = g[:, ax] - self.node_pos[ax]
            t = (val - self.node_pos[ax]) / np.where(np.abs(den) < 1e-9, 1e-9, den)
            p = self.node_pos[None, :] + t[:, None] * (g - self.node_pos[None, :])
            ok = (t > 0.0) & (t < 1.0) & (np.sign(grid[:, ax] - val) == np.sign(self.node_pos[ax] - val))
            for j in range(3):
                if j != ax:
                    ok &= (p[:, j] >= lo[j] - 0.10) & (p[:, j] <= hi[j] + 0.10)
            imgs.append(g)
            rhos.append(rf)
            valid.append(ok)

        C = len(grid)
        R = len(imgs)
        M = cal_pat.n_elem + (1 if include_ref else 0)
        self.S = np.zeros((C, R, M), dtype=np.complex64)
        self.dt = np.zeros((C, R), dtype=np.float64)
        self.w = np.zeros((C, R), dtype=np.float64)

        d0 = np.maximum(np.linalg.norm(grid - self.node_pos[None, :], axis=1), 0.15)
        for i, g in enumerate(imgs):
            v = g - self.node_pos[None, :]
            d = np.maximum(np.linalg.norm(v, axis=1), 0.15)
            U = v / d[:, None]
            man = build_calibrated_manifold(
                U, cal_pat, heading_rad=heading_est, include_ref=include_ref, exact=exact_manifold
            )
            self.S[:, i, :] = man * valid[i][:, None]
            self.dt[:, i] = (d - d0) / C0
            self.w[:, i] = (rhos[i] * d0 / d) ** 2

        self.Fq = np.exp(-2j * np.pi * self.df[None, None, :] * self.dt[:, :, None]).astype(np.complex64)
        Gs = np.einsum("crm,csm->crs", self.S.conj(), self.S)
        Gf = np.einsum("crk,csk->crs", self.Fq.conj(), self.Fq)
        self.G = (Gs * Gf).astype(np.complex64)
        if tc_step_ns == "auto":
            # Rev.2: step = min(8 ns, 0.16 / bandwidth) so 80 MHz channels do not alias the delay search
            bw = float(self.df.max() - self.df.min()) if self.df.size > 1 else 20e6
            tc_step = min(8e-9, 0.16 / max(bw, 1.0))
        else:
            tc_step = float(tc_step_ns) * 1e-9
        tc = np.arange(-40e-9, 40e-9 + 1e-12, tc_step)
        self.Ftc = np.exp(2j * np.pi * self.df[:, None] * tc[None, :]).astype(np.complex64)

        # Precompute regularized inverse Gram matrices once per node (exact same math, ~20x faster)
        self._inv_all = self._build_inv(np.arange(R))
        self._inv_nlos = self._build_inv(np.arange(1, R))
        # Precompute S_conj and Fq_conj for cache-friendly scoring
        self._S_conj = np.ascontiguousarray(self.S.conj())
        self._Fq_conj = np.ascontiguousarray(self.Fq.conj())

    def _build_inv(self, idx):
        n = len(idx)
        G = self.G[:, idx][:, :, idx]
        w = self.w[:, idx]
        tr = np.real(np.einsum("crr->c", G)) / max(n, 1)
        Greg = G + ((self.lam * tr)[:, None] / np.maximum(w, 1e-4))[:, :, None] * np.eye(n, dtype=np.complex64)[None, :, :]
        return np.linalg.inv(Greg).astype(np.complex64)

    @staticmethod
    def _mirror(g, ax, val):
        o = g.copy()
        o[:, ax] = 2.0 * val - o[:, ax]
        return o

    def _score_precomputed(self, b_all, inv_mat, idx, x_norm2):
        b = b_all[:, idx, :]
        sol = inv_mat @ b
        q = np.real(np.sum(b.conj() * sol, axis=1))
        sc = np.clip(np.max(q, axis=1) / (x_norm2 + 1e-30), 0.0, 0.999)
        return sc

    def log_likelihood(self, X, n_eff=10.0, w_vis=0.72):
        """Returns log-likelihood map over candidate grid (C,) and circular spread (deg)."""
        X_use = X.astype(np.complex64)
        if self.robust_huber:
            # Huber-clip extreme subcarrier energy spikes (from deep H_ref fades or DMC resonances)
            col_norm = np.linalg.norm(X_use, axis=0)
            med = float(np.median(col_norm) + 1e-12)
            thr = med * (1.0 + self.huber_delta * 3.0)
            scale = np.where(col_norm > thr, thr / (col_norm + 1e-12), 1.0)
            X_use = X_use * scale[None, :].astype(np.float32)

        x_norm2 = float(np.sum(np.abs(X_use) ** 2))
        # Direct BLAS matmul (C, R, M) @ (M, K) -> (C, R, K)
        b_all = ((self._S_conj @ X_use) * self._Fq_conj) @ self.Ftc  # (C, R, T)
        R = self.S.shape[1]
        s1 = self._score_precomputed(b_all, self._inv_all, np.arange(R), x_norm2)
        s0 = self._score_precomputed(b_all, self._inv_nlos, np.arange(1, R), x_norm2)

        if self.robust_huber:
            # Student-t / soft-bounded residual log-likelihood prevents single-candidate overfitting under DMC
            nu = 6.0
            l1 = -0.5 * (nu + 1.0) * np.log(1.0 + (2.0 * n_eff / nu) * np.maximum(1.0 - s1, 5e-3))
            l0 = -0.5 * (nu + 1.0) * np.log(1.0 + (2.0 * n_eff / nu) * np.maximum(1.0 - s0, 5e-3))
        else:
            l1 = -n_eff * np.log(1.0 - s1 + 2e-3)
            l0 = -n_eff * np.log(1.0 - s0 + 2e-3)

        ll = np.logaddexp(np.log(w_vis) + l1, np.log(1.0 - w_vis) + l0)
        spread = self.azimuth_spread(ll)
        return ll, spread

    def azimuth_spread(self, ll, temp=1.0):
        p = np.exp((ll - ll.max()) / temp)
        p /= (p.sum() + 1e-30)
        v = self.grid - self.node_pos[None, :]
        az = np.arctan2(v[:, 1], v[:, 0])
        Rm = float(np.abs(np.sum(p * np.exp(1j * az))))
        return float(np.degrees(np.sqrt(max(-2.0 * np.log(max(Rm, 1e-9)), 0.0))))

    def estimate_azimuth(self, ll):
        c = int(np.argmax(ll))
        v = self.grid[c] - self.node_pos
        return float(np.arctan2(v[1], v[0])), self.grid[c]


def fuse_two_nodes(ll_list, spreads, grid, node_positions, confidence_weighted=True, sp_floor=10.0):
    """Asynchronous 2-position confidence-weighted log-likelihood fusion (no ns sync needed, R9 compliant).
    Returns: estimated 3D point, azimuth error list seen from each fused node, combined spread.
    """
    if confidence_weighted:
        weights = [1.0 / max(sp, sp_floor) for sp in spreads]
        w_sum = sum(weights) + 1e-12
        weights = [w / w_sum * len(ll_list) for w in weights]
    else:
        weights = [1.0] * len(ll_list)

    ll_tot = sum(w * (ll - ll.max()) for w, ll in zip(weights, ll_list))
    c_best = int(np.argmax(ll_tot))
    pt_est = grid[c_best]
    az_ests = [float(np.arctan2(pt_est[1] - np_[1], pt_est[0] - np_[0])) for np_ in node_positions]
    return pt_est, az_ests, ll_tot


def az_err_deg(az_est, az_true):
    d = np.degrees(az_est - az_true)
    return float(abs((d + 180.0) % 360.0 - 180.0))
