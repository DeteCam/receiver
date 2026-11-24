import numpy as np
from em_manifold import *
res = np.load("em_results.npy", allow_pickle=True).item()
sel = slice(0, 12)
Us, _, _ = fov_grid(75, 60, 1.5); Ut, azt, elt = fov_grid(60, 40, 4)
rng = np.random.default_rng(1)
def noisy(H, snr_db=20):
    p = np.mean(np.abs(H) ** 2, axis=0, keepdims=True)
    return H + np.sqrt(p * 10 ** (-snr_db / 10) / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
for tag, f in (("sh_2g4", 2440), ("sh_5g", 5220)):
    m = Manifold(res[tag][f], f * 1e6); A_id = m.ideal(Us)[sel]
    # (a) repo plan: one coupling matrix C from a single turntable rotation (azimuth cut, elevation 0, one polarisation)
    az = np.radians(np.arange(-80, 81, 2.0)); Ucal = np.stack([np.sin(az), np.cos(az), np.zeros_like(az)], 1)
    out = []
    for cal_name, cal_psi in (("V-pol source", 0.0), ("45deg source", np.pi / 4)):
        Hc = m.resp(Ucal, pol_vec(Ucal, cal_psi))[sel]; Ac = m.ideal(Ucal)[sel]
        Chat = Hc @ np.linalg.pinv(Ac)                                   # least squares  H ~ C A
        A_c = Chat @ A_id
        row = []
        for psi in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
            H = noisy(m.resp(Ut, pol_vec(Ut, psi))[sel])
            e = doa_err(H, A_c, Us, Ut); row.append(f"{np.median(e):4.1f}/{np.percentile(e,90):4.1f}")
        out.append(f"   C from turntable ({cal_name}): med/P90 deg for V, H, +45, -45 pol = " + ", ".join(row))
    # (b) ideal manifold without calibration, incl. -45
    row = []
    for psi in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
        H = noisy(m.resp(Ut, pol_vec(Ut, psi))[sel]); e = doa_err(H, A_id, Us, Ut); row.append(f"{np.median(e):4.1f}/{np.percentile(e,90):4.1f}")
    # (c) full 2-D dual-polarised calibration table + joint polarisation search (best achievable for this hardware)
    Bt = m.resp(Us, pol_vec(Us, 0.0))[sel]; Bp = m.resp(Us, pol_vec(Us, np.pi / 2))[sel]            # (12, G)
    def dualpol_err(H):
        G = Bt.shape[1]; spec = np.zeros((G, H.shape[1]))
        a = np.sum(np.abs(Bt) ** 2, 0); b = np.sum(np.conj(Bt) * Bp, 0); d = np.sum(np.abs(Bp) ** 2, 0)
        det = np.maximum(a * d - np.abs(b) ** 2, 1e-12 * a * d + 1e-30)
        y1 = Bt.conj().T @ H; y2 = Bp.conj().T @ H                                                    # (G, N)
        spec = np.real((d[:, None] * np.abs(y1) ** 2 - 2 * np.real(b[:, None] * np.conj(y1) * y2) + a[:, None] * np.abs(y2) ** 2) / det[:, None])
        best = np.argmax(spec, 0); cos = np.clip(np.sum(Us[best] * Ut, 1), -1, 1); return np.degrees(np.arccos(cos))
    row2 = []
    for psi in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
        e = dualpol_err(noisy(m.resp(Ut, pol_vec(Ut, psi))[sel])); row2.append(f"{np.median(e):4.1f}/{np.percentile(e,90):4.1f}")
    # (d) reference: an ideal co-polarised 12-element array with the same positions (what the repo simulation assumes)
    e = doa_err(noisy(m.ideal(Ut)[sel]), A_id, Us, Ut)
    print(f"== {tag} {f} MHz, single path, SNR 20 dB")
    print("   repo assumption (ideal elements, same positions): med/P90 = " + f"{np.median(e):4.1f}/{np.percentile(e,90):4.1f} deg")
    print("   full-wave board + ideal steering (no calibration):   V, H, +45, -45 = " + ", ".join(row))
    print("\n".join(out))
    print("   full 2-D dual-pol chamber calibration + pol search:  V, H, +45, -45 = " + ", ".join(row2))
