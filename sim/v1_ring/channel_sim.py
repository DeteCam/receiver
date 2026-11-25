"""Realistic RF front-end + Intel AX210 switched-ring CSI synthesizer for V1 evaluation.
Takes raw ray-traced multipath from gen_scenes.py + full-wave NEC2 patterns from ring_nec2.py.
NEVER imports estimators.py (strict generator/estimator separation).

Implements:
  - Continuous bilinear interpolation of NEC2 active element patterns (with geometric phase stripped)
  - Dynamic human-motion phase perturbations across N_sweep sweeps
  - Hardware calibration residual (phase/amp jitter) and thermal/relative AWGN
  - Intel AX210 per-packet random pi (180 deg) phase flips (naive vs Passive RF Sign Anchor recovery)
  - Temporal aggregation methods: non-gated complex ratio mean (recommended), conjugate product mean,
    and top-70% amplitude-gated ratio mean (ablation)
  - Single-antenna amplitude diffraction proxy for DiffLoc comparison on identical multipath
"""
import numpy as np

C0 = 299792458.0


class NEC2ArrayPattern:
    """Full-wave 3D NEC2 active element pattern interpolator for a switched ring + center reference."""
    def __init__(self, pat_dict, fc_hz):
        self.fc = float(fc_hz)
        self.k0 = 2.0 * np.pi * self.fc / C0
        self.n_elem = int(pat_dict["n_elem"])
        self.xy = np.asarray(pat_dict["elem_xy_m"], dtype=np.float64)  # (N, 2)
        self.ref_dz = float(pat_dict["ref_dz_m"])

        th = np.radians(np.arange(0, 181, 5))   # (37,)
        ph = np.radians(np.arange(0, 360, 5))   # (72,)
        ux = np.sin(th)[None, :] * np.cos(ph)[:, None]
        uy = np.sin(th)[None, :] * np.sin(ph)[:, None]
        uz = np.cos(th)[None, :] * np.ones((72, 1))

        # Strip nominal geometric phase before bilinear interpolation so phase wraps smoothly
        self.eth_env = np.zeros_like(pat_dict["eth"], dtype=np.complex64)
        self.eph_env = np.zeros_like(pat_dict["eph"], dtype=np.complex64)
        for m in range(self.n_elem):
            nom = np.exp(1j * self.k0 * (self.xy[m, 0] * ux + self.xy[m, 1] * uy))
            self.eth_env[m] = (pat_dict["eth"][m] / nom).astype(np.complex64)
            self.eph_env[m] = (pat_dict["eph"][m] / nom).astype(np.complex64)

        nom_ref = np.exp(1j * self.k0 * (self.ref_dz * uz))
        self.ref_eth_env = (pat_dict["ref_eth"] / nom_ref).astype(np.complex64)
        self.ref_eph_env = (pat_dict["ref_eph"] / nom_ref).astype(np.complex64)

    @staticmethod
    def _bil(F, th, ph):
        """Bilinear interpolation on (..., 72, 37) grid for arbitrary shaped th, ph (radians)."""
        a = (np.degrees(ph) % 360.0) / 5.0
        i0 = np.floor(a).astype(np.int32) % 72
        fa = a - np.floor(a)
        i1 = (i0 + 1) % 72
        b = np.clip(np.degrees(th) / 5.0, 0.0, 36.0 - 1e-6)
        j0 = np.floor(b).astype(np.int32)
        fb = b - j0
        j1 = j0 + 1
        return (
            (F[..., i0, j0] * (1.0 - fa) + F[..., i1, j0] * fa) * (1.0 - fb)
            + (F[..., i0, j1] * (1.0 - fa) + F[..., i1, j1] * fa) * fb
        )

    def element_responses(self, th, ph, aV=None, aH=None):
        """Evaluate ring (N, ...) and ref (...) complex responses at angles (th, ph).
        If aV, aH are None, returns pure V-pol (theta-polarized) manifold for estimator calibration.
        """
        th = np.asarray(th, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        ux = np.sin(th) * np.cos(ph)
        uy = np.sin(th) * np.sin(ph)
        uz = np.cos(th)
        geo = np.exp(1j * self.k0 * (self.xy[:, 0:1] * ux.ravel()[None, :] + self.xy[:, 1:2] * uy.ravel()[None, :]))
        geo = geo.reshape((self.n_elem,) + th.shape)
        geo_ref = np.exp(1j * self.k0 * (self.ref_dz * uz))

        et = self._bil(self.eth_env, th, ph) * geo
        ep = self._bil(self.eph_env, th, ph) * geo
        rt = self._bil(self.ref_eth_env, th, ph) * geo_ref
        rp = self._bil(self.ref_eph_env, th, ph) * geo_ref

        if aV is None:
            return et, rt
        return et * aV[None, ...] + ep * aH[None, ...], rt * aV + rp * aH


class IdealDipoleArrayPattern:
    """Ideal isotropic/short-dipole upper bound (no mutual coupling, no box) for ladder L0-L2 & ablation."""
    def __init__(self, n_elem, diam_mm, ref_dz_mm, fc_hz):
        self.fc = float(fc_hz)
        self.k0 = 2.0 * np.pi * self.fc / C0
        self.n_elem = int(n_elem)
        a = 2.0 * np.pi * np.arange(n_elem) / n_elem
        self.xy = np.stack([0.5 * diam_mm * 1e-3 * np.cos(a), 0.5 * diam_mm * 1e-3 * np.sin(a)], axis=1)
        self.ref_dz = float(ref_dz_mm * 1e-3)

    def element_responses(self, th, ph, aV=None, aH=None):
        th = np.asarray(th, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        ux = np.sin(th) * np.cos(ph)
        uy = np.sin(th) * np.sin(ph)
        uz = np.cos(th)
        sin_t = np.sin(th)
        geo = np.exp(1j * self.k0 * (self.xy[:, 0:1] * ux.ravel()[None, :] + self.xy[:, 1:2] * uy.ravel()[None, :]))
        geo = geo.reshape((self.n_elem,) + th.shape) * sin_t[None, ...]
        ref = np.exp(1j * self.k0 * (self.ref_dz * uz)) * sin_t
        if aV is None:
            return geo, ref
        return geo * aV[None, ...], ref * aV


def synthesize_ring_csi(
    paths_rn,
    cam_pos,
    node_pos,
    arr_pat,
    df_hz,
    rng,
    n_sweep=10,
    dyn_frac=0.35,
    dyn_std_deg=45.0,
    cal_phase_deg=4.5,
    cal_amp_db=0.4,
    snr_db=20.0,
    flip_mode="sign_anchor",
    anchor_alpha=3.5,
    agg_mode="ratio",
    include_ref=True,
    weight_by_ref_amp=True,
):
    """Synthesize realistic switched-ring observation X (M, K) from Sionna RT paths for 1 (cam, node) pair.
    Returns: (X (M, K), stats dict with sign_acc, los_frac)
    """
    aV, aH, tau, th, ph = paths_rn
    pw = np.abs(aV) ** 2 + np.abs(aH) ** 2
    valid = pw > 1e-24
    if not np.any(valid):
        M = arr_pat.n_elem + (1 if include_ref else 0)
        return np.zeros((M, len(df_hz)), dtype=np.complex64), dict(sign_acc=1.0, los_frac=0.0)

    aV, aH, tau, th, ph, pw = aV[valid], aH[valid], tau[valid], th[valid], ph[valid], pw[valid]
    d_los = float(np.linalg.norm(cam_pos - node_pos))
    is_los = np.abs(tau - d_los / C0) < 0.08e-9
    los_frac = float(pw[is_los].sum() / (pw.sum() + 1e-30))

    # Static element responses for every path: G_ring (N, P), G_ref (P,)
    G_ring, G_ref = arr_pat.element_responses(th, ph, aV, aH)
    Fq = np.exp(-2j * np.pi * tau[:, None] * df_hz[None, :])  # (P, K)
    N = arr_pat.n_elem
    K = len(df_hz)

    # Which non-direct paths are perturbed by moving humans (proctors/students/people)
    dyn_mask = (~is_los) & (rng.random(len(tau)) < dyn_frac) if (dyn_frac > 0 and dyn_std_deg > 0) else np.zeros(len(tau), dtype=bool)
    n_dyn = int(dyn_mask.sum())

    # Fixed per-port residual calibration error (from turntable calibration imperfection)
    if cal_phase_deg > 0 or cal_amp_db > 0:
        port_cal = (10.0 ** (rng.normal(0.0, cal_amp_db, N) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, cal_phase_deg, N)))
    else:
        port_cal = np.ones(N, dtype=np.complex128)

    # Reference signal scale for AWGN
    H_ref_static = G_ref @ Fq
    ref_pwr = float(np.mean(np.abs(H_ref_static) ** 2) + 1e-30)
    noise_std = np.sqrt(ref_pwr * (10.0 ** (-snr_db / 10.0)) / 2.0) if snr_db < 90.0 else 0.0

    # Passive RF Sign Anchor true vs calibrated alpha (calibrated via PE42482 All-Off 50 Ohm state)
    alpha_true = anchor_alpha * np.exp(1j * 0.4) * np.exp(-2j * np.pi * 1.5e-9 * df_hz)
    # Small anchor calibration residual (3 deg phase, 0.25 dB amp)
    if cal_phase_deg > 0:
        alpha_cal = alpha_true * (10.0 ** (rng.normal(0.0, 0.25) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, 3.0)))
    else:
        alpha_cal = alpha_true

    # Vectorized sweep x port packet synthesis: shape (n_sweep, N, K)
    if n_dyn > 0:
        stat_mask = ~dyn_mask
        H_ring_stat = (port_cal[:, None] * G_ring[:, stat_mask]) @ Fq[stat_mask]  # (N, K)
        H_ref_stat = G_ref[stat_mask] @ Fq[stat_mask]  # (K,)
        Fq_dyn = Fq[dyn_mask]  # (n_dyn, K)
        ph_dyn = np.exp(1j * np.radians(rng.normal(0.0, dyn_std_deg, (n_sweep, N, n_dyn))))  # (T, N, n_dyn)
        # Dynamic contributions: ring (T, N, K), ref (T, N, K)
        G_ring_dyn = port_cal[:, None] * G_ring[:, dyn_mask]  # (N, n_dyn)
        Hm_all = H_ring_stat[None, :, :] + (ph_dyn * G_ring_dyn[None, :, :]) @ Fq_dyn
        Hr_all = H_ref_stat[None, None, :] + (ph_dyn * G_ref[None, None, dyn_mask]) @ Fq_dyn
    else:
        Hm_all = np.broadcast_to((port_cal[:, None] * G_ring) @ Fq, (n_sweep, N, K)).copy()
        Hr_all = np.broadcast_to(H_ref_static[None, None, :], (n_sweep, N, K)).copy()

    pkt_ph = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (n_sweep, N, 1)))
    if noise_std > 0:
        n0 = noise_std * (rng.standard_normal((n_sweep, N, K)) + 1j * rng.standard_normal((n_sweep, N, K)))
        n1 = noise_std * (rng.standard_normal((n_sweep, N, K)) + 1j * rng.standard_normal((n_sweep, N, K)))
    else:
        n0 = 0.0
        n1 = 0.0

    Y0 = pkt_ph * Hr_all + n0
    ref_amps = np.abs(Y0)

    if flip_mode == "none":
        Y1 = pkt_ph * Hm_all + n1
        ratios = Y1 / (Y0 + 1e-30)
        conj_num = Y1 * np.conj(Y0)
        sign_acc = 1.0
    elif flip_mode == "naive_flip":
        s_true = np.where(rng.random((n_sweep, N, 1)) < 0.5, 1.0, -1.0)
        Y1 = pkt_ph * (s_true * Hm_all) + n1
        ratios = Y1 / (Y0 + 1e-30)
        conj_num = Y1 * np.conj(Y0)
        sign_acc = 1.0
    elif flip_mode == "sign_anchor":
        s_true = np.where(rng.random((n_sweep, N, 1)) < 0.5, 1.0, -1.0)
        Y1 = pkt_ph * (s_true * (Hm_all + alpha_true[None, None, :] * Hr_all)) + n1
        R_raw = Y1 / (Y0 + 1e-30)
        w_mf = ref_amps ** 2
        metric = np.real(np.sum(w_mf * np.conj(alpha_cal)[None, None, :] * R_raw, axis=2, keepdims=True))
        s_hat = np.where(metric >= 0.0, 1.0, -1.0)
        sign_acc = float(np.mean(s_hat == s_true))
        ratios = s_hat * R_raw - alpha_cal[None, None, :]
        conj_num = (s_hat * Y1 - alpha_cal[None, None, :] * Y0) * np.conj(Y0)
    else:
        raise ValueError(f"Unknown flip_mode: {flip_mode}")

    # Temporal aggregation across sweeps
    if agg_mode == "ratio":
        # Non-gated complex ratio average (Cauchy theorem: zero asymptotic bias when static dominates)
        X_ring = np.mean(ratios, axis=0)
    elif agg_mode == "conj":
        # Biased conjugate-product average (ablation)
        denom = np.mean(ref_amps ** 2, axis=0) + 1e-30
        X_ring = np.mean(conj_num, axis=0) / denom
    elif agg_mode == "ratio_gated70":
        # Top-70% |H_ref| amplitude-gated ratio average (ablation showing circular-symmetry bias)
        X_ring = np.zeros((N, K), dtype=np.complex128)
        for m in range(N):
            amp_pkt = np.mean(ref_amps[:, m, :], axis=1)
            thr = np.percentile(amp_pkt, 30.0)
            keep = amp_pkt >= thr
            X_ring[m] = np.mean(ratios[keep, m, :], axis=0) if np.any(keep) else np.mean(ratios[:, m, :], axis=0)
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    if include_ref:
        if cal_phase_deg > 0:
            c_ref = (10.0 ** (rng.normal(0.0, 0.2) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, 2.0)))
        else:
            c_ref = 1.0 + 0j
        X = np.vstack([X_ring, np.full((1, K), c_ref, dtype=np.complex128)])
    else:
        X = X_ring

    if weight_by_ref_amp:
        w_k = np.sqrt(np.mean(ref_amps ** 2, axis=(0, 1)))
        w_k = w_k / (np.sqrt(np.mean(w_k ** 2)) + 1e-30)
        X = X * w_k[None, :]

    return X.astype(np.complex64), dict(sign_acc=float(sign_acc), los_frac=los_frac)


def simulate_diffloc_proxy(paths_rn, cam_pos, node_pos, df_hz, fc_hz, rng, dyn_frac=0.35, dyn_std_deg=45.0, snr_db=20.0):
    """Physics proxy of DiffLoc (USENIX Security '25): single vertical dipole CSI amplitude
    as a rotating metal plate creates a Fresnel diffraction notch across azimuth angles 0..358 deg.
    Uses identical Sionna RT multipath + moving-people dynamic perturbations.
    Note: Allows full 360 deg scan (favorable to DiffLoc vs its 180 deg hardware rear block).
    """
    aV, aH, tau, th, ph = paths_rn
    pw = np.abs(aV) ** 2 + np.abs(aH) ** 2
    valid = pw > 1e-24
    if not np.any(valid):
        return 0.0
    aV, tau, th, ph = aV[valid], tau[valid], th[valid], ph[valid]
    d_los = float(np.linalg.norm(cam_pos - node_pos))
    is_los = np.abs(tau - d_los / C0) < 0.08e-9
    dyn_mask = (~is_los) & (rng.random(len(tau)) < dyn_frac) if (dyn_frac > 0 and dyn_std_deg > 0) else np.zeros(len(tau), dtype=bool)
    n_dyn = int(dyn_mask.sum())

    # Single vertical dipole gain
    g_base = aV * np.sin(th)
    Fq = np.exp(-2j * np.pi * tau[:, None] * df_hz[None, :])
    H0 = g_base @ Fq
    ref_pwr = float(np.mean(np.abs(H0) ** 2) + 1e-30)
    noise_std = np.sqrt(ref_pwr * (10.0 ** (-snr_db / 10.0)) / 2.0) if snr_db < 90.0 else 0.0

    # Angular width of First Fresnel Zone shadow by 100x150 mm aluminum plate at ~8 cm radius
    sigma_deg = 22.0 if fc_hz < 4e9 else 16.0
    scan_deg = np.arange(0.0, 360.0, 3.0)  # 120 angular steps over ~3s motor sweep
    ph_deg = np.degrees(ph) % 360.0
    amps = np.zeros((len(scan_deg), len(df_hz)), dtype=np.float64)

    # Smooth AR(1) human motion drift over the 3-second sweep (correlated in time)
    ph_state = np.zeros(n_dyn, dtype=np.float64)
    step_std = np.radians(dyn_std_deg) * 0.25

    for idx, psi in enumerate(scan_deg):
        d_ang = np.abs((ph_deg - psi + 180.0) % 360.0 - 180.0)
        # Diffraction notch: -10 dB (factor 0.316) at plate center, Gaussian angular falloff
        atten = 1.0 - 0.684 * np.exp(-0.5 * (d_ang / sigma_deg) ** 2)
        if n_dyn > 0:
            ph_state += rng.normal(0.0, step_std, n_dyn)
            ph_dyn = np.ones(len(tau), dtype=np.complex128)
            ph_dyn[dyn_mask] = np.exp(1j * ph_state)
            H = (g_base * atten * ph_dyn) @ Fq
        else:
            H = (g_base * atten) @ Fq
        if noise_std > 0:
            H = H + noise_std * (rng.standard_normal(len(df_hz)) + 1j * rng.standard_normal(len(df_hz)))
        amps[idx] = np.abs(H)

    # DiffLoc algorithm: select top-5 subcarriers by mean amplitude, smooth, pick deepest attenuation trough
    top5 = np.argsort(-amps.mean(axis=0))[:5]
    curve = amps[:, top5].mean(axis=1)
    # Circular moving average LPF (window=5 steps = 15 deg)
    ker = np.ones(5) / 5.0
    pad = np.concatenate([curve[-2:], curve, curve[:2]])
    smooth = np.convolve(pad, ker, mode="valid")
    est_deg = float(scan_deg[int(np.argmin(smooth))])
    return np.radians(est_deg)
