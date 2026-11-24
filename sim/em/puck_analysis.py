"""How far is the rotating-puck response from the ideal ring model, and how much does a one-rotation (1-D) calibration fix?"""
import numpy as np
C0 = 299792458.0
res = np.load("puck_results.npy", allow_pickle=True).item()
rng = np.random.default_rng(5)
TH = np.arange(0, 181, 5); PH = np.arange(0, 360, 5)

def q_table(r, psi_deg):
    """ratio moving/reference for a linearly polarised emitter tilted psi from vertical; (72 phi, 37 theta)"""
    pt, pp = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    return (r[9002]["eth"] * pt + r[9002]["eph"] * pp) / (r[9001]["eth"] * pt + r[9001]["eph"] * pp)

def interp_alpha(col, alpha_deg):
    """periodic linear interpolation of a (72,) complex column at angles alpha (deg)"""
    a = (np.asarray(alpha_deg) % 360) / 5.0; i0 = np.floor(a).astype(int) % 72; f = a - np.floor(a)
    return col[i0] * (1 - f) + col[(i0 + 1) % 72] * f

for (f, box), r in res.items():
    k = 2 * np.pi * f * 1e6 / C0; R = 0.044; phm = np.arange(0, 360, 10.0)
    qV = q_table(r, 0.0)
    print(f"\n=== {f} MHz {'with metal body behind the disc' if box else 'disc only'}")
    # deviation from the ideal ring model
    for eps in (5, 15, 30, 45):
        it = int((90 - eps) / 5); ideal = np.exp(1j * k * R * np.sin(np.radians(90 - eps)) * np.cos(np.radians(PH)))
        c = qV[:, it] / ideal; c = c / np.exp(1j * np.angle(np.mean(c)))
        xp = 20 * np.log10(np.abs(r[9002]["eph"][:, it]).max() / np.abs(r[9002]["eth"][:, it]).mean() + 1e-12)
        print(f"  elevation {eps:2d} deg: amplitude ripple {20*np.log10(np.abs(c).max()/np.abs(c).min()):4.1f} dB, phase deviation from ideal rms {np.degrees(np.std(np.angle(c))):5.1f} deg (max {np.degrees(np.abs(np.angle(c)).max()):5.1f}), cross-pol {xp:5.1f} dB")
    # azimuth error with different manifolds
    azs = np.arange(0, 360, 0.5); ths = np.arange(40, 90, 5)
    def manifold(kind, th_idx_list):
        out = []
        for it in th_idx_list:
            th = np.radians(TH[it]); ideal = np.exp(1j * k * R * np.sin(th) * np.cos(np.radians(azs[:, None] - phm[None])))
            if kind == "ideal": A = ideal
            elif kind == "cal1d":
                itc = int(80 / 5); cc = qV[:, itc] / np.exp(1j * k * R * np.sin(np.radians(80)) * np.cos(np.radians(PH)))
                A = ideal * interp_alpha(cc, azs[:, None] - phm[None])
            else: A = interp_alpha(qV[:, it], azs[:, None] - phm[None])
            out.append(A / np.linalg.norm(A, axis=1, keepdims=True))
        return np.stack(out, 0)                                              # (T, G, M)
    tl = [int(t / 5) for t in ths]
    for psi in (0, 45, 70):
        q = q_table(r, psi); line = []
        for kind in ("ideal", "cal1d", "cal2d"):
            A = manifold(kind, tl); errs = []
            for eps in (5, 15, 30):
                it = int((90 - eps) / 5)
                for az in np.arange(0, 360, 7.0):
                    x = interp_alpha(q[:, it], az - phm); x = x + (np.abs(x).mean() * 10 ** (-20 / 20) / np.sqrt(2)) * (rng.standard_normal(36) + 1j * rng.standard_normal(36))
                    P = np.abs(A.conj() @ x) ** 2; g = np.unravel_index(int(np.argmax(P)), P.shape)[1]
                    d = abs((azs[g] - az + 180) % 360 - 180); errs.append(d)
            errs = np.array(errs); line.append(f"{kind}: median {np.median(errs):4.1f}, P90 {np.percentile(errs, 90):4.1f}")
        print(f"  azimuth error, emitter polarisation tilted {psi:2d} deg from vertical ->  " + "  |  ".join(line))
