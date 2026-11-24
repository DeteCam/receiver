import numpy as np
C0 = 299792458.0; TH = np.arange(0, 181, 5); PH = np.arange(0, 360, 5)

def interp_alpha(col, alpha_deg):
    a = (np.asarray(alpha_deg) % 360) / 5.0; i0 = np.floor(a).astype(int) % 72; f = a - np.floor(a)
    return col[i0] * (1 - f) + col[(i0 + 1) % 72] * f

def analyse(r, f, R, label, eps_list=(5, 15, 30), seed=5):
    """r[9001]/r[9002]: reference / moving antenna patterns eth, eph (72 phi, 37 theta). Prints deviation and azimuth errors."""
    rng = np.random.default_rng(seed); k = 2 * np.pi * f * 1e6 / C0; phm = np.arange(0, 360, 10.0)
    def meas(psi, it, az):
        pt, pp = np.cos(np.radians(psi)), np.sin(np.radians(psi))
        mov = interp_alpha(r[9002]["eth"][:, it], az - phm) * pt + interp_alpha(r[9002]["eph"][:, it], az - phm) * pp
        ref = interp_alpha(r[9001]["eth"][:, it], az - phm) * pt + interp_alpha(r[9001]["eph"][:, it], az - phm) * pp
        x = mov * np.exp(-1j * np.angle(ref))                                   # phase-referenced to the fixed antenna
        return x + (np.abs(x).mean() * 10 ** (-20 / 20) / np.sqrt(2)) * (rng.standard_normal(36) + 1j * rng.standard_normal(36))
    print(f"\n=== {f} MHz, {label}")
    for eps in (5, 15, 30, 45):
        it = int((90 - eps) / 5); ideal = np.exp(1j * k * R * np.sin(np.radians(90 - eps)) * np.cos(np.radians(PH)))
        c = r[9002]["eth"][:, it] * np.exp(-1j * np.angle(r[9001]["eth"][:, it])) / ideal; c = c / np.exp(1j * np.angle(np.mean(c)))
        xp = 20 * np.log10(np.abs(r[9002]["eph"][:, it]).max() / np.abs(r[9002]["eth"][:, it]).mean() + 1e-12)
        xr = 20 * np.log10(np.abs(r[9001]["eph"][:, it]).max() / np.abs(r[9001]["eth"][:, it]).mean() + 1e-12)
        print(f"  elevation {eps:2d} deg: amplitude ripple {20*np.log10(np.abs(c).max()/np.abs(c).min()):4.1f} dB, phase deviation rms {np.degrees(np.std(np.angle(c))):5.1f} deg, cross-pol moving {xp:6.1f} dB / reference {xr:6.1f} dB")
    azs = np.arange(0, 360, 0.5); tl = [int(t / 5) for t in np.arange(40, 90, 5)]; d = azs[:, None] - phm[None]
    Aid, Ac1, Ac2, Ap = [], [], [], []
    itc = int(80 / 5); cc = r[9002]["eth"][:, itc] * np.exp(-1j * np.angle(r[9001]["eth"][:, itc])) / np.exp(1j * k * R * np.sin(np.radians(80)) * np.cos(np.radians(PH)))
    for it in tl:
        ideal = np.exp(1j * k * R * np.sin(np.radians(TH[it])) * np.cos(np.radians(d)))
        ref_ph = np.exp(-1j * np.angle(interp_alpha(r[9001]["eth"][:, it], d)))
        at = interp_alpha(r[9002]["eth"][:, it], d) * ref_ph; ap = interp_alpha(r[9002]["eph"][:, it], d) * ref_ph
        Aid.append(ideal); Ac1.append(ideal * interp_alpha(cc, d)); Ac2.append(at); Ap.append(ap)
    nrm = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)
    Aid, Ac1, Ac2 = nrm(np.stack(Aid)), nrm(np.stack(Ac1)), nrm(np.stack(Ac2)); At = np.stack([a for a in Ac2]); Ap = np.stack(Ap)
    Ap_n = Ap / (np.linalg.norm(Ap, axis=-1, keepdims=True) + 1e-12)
    for psi in (0, 45, 70):
        errs = {"ideal ring model": [], "1 rotation (V) calibration": [], "V table at each elevation": [], "V+H tables (2-D subspace)": []}
        for eps in eps_list:
            it = int((90 - eps) / 5)
            for az in np.arange(0, 360, 7.0):
                x = meas(psi, it, az)
                for nm, A in (("ideal ring model", Aid), ("1 rotation (V) calibration", Ac1), ("V table at each elevation", Ac2)):
                    P = np.abs(A.conj() @ x) ** 2; g = np.unravel_index(int(np.argmax(P)), P.shape)[1]; errs[nm].append(abs((azs[g] - az + 180) % 360 - 180))
                y1 = At.conj() @ x; y2 = Ap_n.conj() @ x; b = np.sum(At.conj() * Ap_n, axis=-1); det = np.maximum(1 - np.abs(b) ** 2, 1e-6)
                P = (np.abs(y1) ** 2 - 2 * np.real(b * y1.conj() * y2) + np.abs(y2) ** 2) / det
                g = np.unravel_index(int(np.argmax(P)), P.shape)[1]; errs["V+H tables (2-D subspace)"].append(abs((azs[g] - az + 180) % 360 - 180))
        print(f"  azimuth error, emitter polarisation tilted {psi:2d} deg: " + " | ".join(f"{nm}: {np.median(v):4.1f} / {np.percentile(v, 90):4.1f}" for nm, v in errs.items()))
