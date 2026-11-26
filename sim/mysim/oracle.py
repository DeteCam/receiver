"""Upper-bound check for E6: what if the receiver knew every packet's absolute phase?

The real receiver only has the per-packet ratio ring/reference, so the estimator must assume that the phase of the
reference channel is (nearly) linear in frequency.  Here the same paths / calibration errors / human motion / noise
are used, but the per-packet random phase is removed by an oracle and ring and reference channels are averaged
directly.  With the exact manifold [et, rt] this observation fits the estimator's linear model with no
approximation, so (ratio result - oracle result) bounds what any 'self-consistent reference phase' trick could gain.
"""
import numpy as np

C0 = 299792458.0


def synth_oracle(paths_rn, cam_pos, node_pos, arr_pat, df_hz, rng, n_sweep=10, dyn_frac=0.35, dyn_std_deg=45.0,
                 cal_phase_deg=4.5, cal_amp_db=0.4, snr_db=20.0, **_):
    aV, aH, tau, th, ph = paths_rn
    pw = np.abs(aV) ** 2 + np.abs(aH) ** 2
    valid = pw > 1e-24
    N, K = arr_pat.n_elem, len(df_hz)
    if not np.any(valid):
        return np.zeros((N + 1, K), dtype=np.complex64), dict(sign_acc=1.0, los_frac=0.0)
    aV, aH, tau, th, ph, pw = aV[valid], aH[valid], tau[valid], th[valid], ph[valid], pw[valid]
    d_los = float(np.linalg.norm(cam_pos - node_pos))
    is_los = np.abs(tau - d_los / C0) < 0.08e-9
    los_frac = float(pw[is_los].sum() / (pw.sum() + 1e-30))
    G_ring, G_ref = arr_pat.element_responses(th, ph, aV, aH)
    Fq = np.exp(-2j * np.pi * tau[:, None] * df_hz[None, :])
    dyn_mask = (~is_los) & (rng.random(len(tau)) < dyn_frac)
    port_cal = (10.0 ** (rng.normal(0.0, cal_amp_db, N) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, cal_phase_deg, N)))
    H_ref_static = G_ref @ Fq
    ref_pwr = float(np.mean(np.abs(H_ref_static) ** 2) + 1e-30)
    noise_std = np.sqrt(ref_pwr * (10.0 ** (-snr_db / 10.0)) / 2.0)
    stat = ~dyn_mask
    Hm = np.broadcast_to(((port_cal[:, None] * G_ring[:, stat]) @ Fq[stat])[None], (n_sweep, N, K)).copy()
    Hr = np.broadcast_to((G_ref[stat] @ Fq[stat])[None, None], (n_sweep, N, K)).copy()
    if dyn_mask.any():
        phd = np.exp(1j * np.radians(rng.normal(0.0, dyn_std_deg, (n_sweep, N, int(dyn_mask.sum())))))
        Hm = Hm + (phd * (port_cal[:, None] * G_ring[:, dyn_mask])[None]) @ Fq[dyn_mask]
        Hr = Hr + (phd * G_ref[None, None, dyn_mask]) @ Fq[dyn_mask]
    n0 = noise_std * (rng.standard_normal((n_sweep, N, K)) + 1j * rng.standard_normal((n_sweep, N, K)))
    n1 = noise_std * (rng.standard_normal((n_sweep, N, K)) + 1j * rng.standard_normal((n_sweep, N, K)))
    X_ring = np.mean(Hm + n1, axis=0)
    X_ref = np.mean(Hr + n0, axis=(0, 1))[None, :]
    X = np.vstack([X_ring, X_ref]) * np.exp(2j * np.pi * (d_los / C0) * df_hz)[None, :]
    X = X / (np.sqrt(np.mean(np.abs(X) ** 2)) + 1e-30)
    return X.astype(np.complex64), dict(sign_acc=1.0, los_frac=los_frac)
