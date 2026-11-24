"""How far is the full-wave array manifold from the ideal point-source steering vector the software assumes?"""
import numpy as np, sys
import necmodel as nm
C0 = 299792458.0
POS = np.array([[nm.elem_xy(e, c)[0], 0.0, nm.elem_xy(e, c)[1]] for e, c in nm.ELEMS]) * 1e-3     # (13,3) m
EDGE = [e for e, c in nm.ELEMS]

class Manifold:
    """Embedded complex patterns G_theta/G_phi of each element, bilinear in (theta, phi) after removing ideal phase."""
    def __init__(self, d, f_hz, pos=POS):
        self.k = 2 * np.pi * f_hz / C0; self.pos = pos
        nth, nph = 37, 72
        th = np.radians(d["th"].reshape(nph, nth)[0]); ph = np.radians(d["ph"].reshape(nph, nth)[:, 0])
        self.th, self.ph = th, ph
        TH, PHg = np.meshgrid(th, ph)                                    # (nph, nth)
        U = np.stack([np.sin(TH) * np.cos(PHg), np.sin(TH) * np.sin(PHg), np.cos(TH)], -1)
        ideal = np.exp(1j * self.k * np.einsum("pti,ni->npt", U, pos))   # (n, nph, nth)
        n = pos.shape[0]
        # NEC far field ~ exp(-jkr) convention with e^{+jwt}: transmit pattern phase = +k r.u  -> demodulate
        self.Gt = d["Eth"].reshape(n, nph, nth) * np.conj(ideal); self.Gp = d["Eph"].reshape(n, nph, nth) * np.conj(ideal)
    def _interp(self, G, th, ph):
        ti = np.clip(th / np.radians(5), 0, 35.999); pi_ = (ph % (2 * np.pi)) / np.radians(5)
        t0 = np.floor(ti).astype(int); p0 = np.floor(pi_).astype(int) % 72; p1 = (p0 + 1) % 72
        ft = ti - t0; fp = pi_ - np.floor(pi_)
        return (G[:, p0, t0] * (1 - fp) * (1 - ft) + G[:, p1, t0] * fp * (1 - ft) + G[:, p0, t0 + 1] * (1 - fp) * ft + G[:, p1, t0 + 1] * fp * ft)
    def resp(self, U, pol):
        """U (N,3) unit vectors, pol (N,2) complex (theta, phi) components of the incident field -> (n, N)."""
        th = np.arccos(np.clip(U[:, 2], -1, 1)); ph = np.arctan2(U[:, 1], U[:, 0])
        ideal = np.exp(1j * self.k * (self.pos @ U.T))
        return (self._interp(self.Gt, th, ph) * pol[:, 0] + self._interp(self.Gp, th, ph) * pol[:, 1]) * ideal
    def ideal(self, U):
        return np.exp(1j * self.k * (self.pos @ U.T))

def fov_grid(az_max=60, el_max=40, step=4):
    az = np.radians(np.arange(-az_max, az_max + 1e-6, step)); el = np.radians(np.arange(-el_max, el_max + 1e-6, step))
    AZ, EL = np.meshgrid(az, el)
    U = np.stack([np.cos(EL) * np.sin(AZ), np.cos(EL) * np.cos(AZ), np.sin(EL)], -1).reshape(-1, 3)
    return U, AZ.ravel(), EL.ravel()

def pol_vec(U, psi):
    """Linear polarisation at angle psi from 'vertical' (theta_hat) in the transverse plane -> (theta, phi) components."""
    return np.stack([np.full(len(U), np.cos(psi)), np.full(len(U), np.sin(psi))], 1).astype(complex)

def doa_err(H, A_search, U_search, U_true):
    """Bartlett peak with manifold A_search (n, G) for each column of H (n, N). Returns angular error (deg)."""
    An = A_search / np.linalg.norm(A_search, axis=0, keepdims=True)
    spec = np.abs(An.conj().T @ H) ** 2                                   # (G, N)
    best = np.argmax(spec, axis=0)
    cos = np.clip(np.sum(U_search[best] * U_true, axis=1), -1, 1)
    return np.degrees(np.arccos(cos))

if __name__ == "__main__":
    res = np.load("em_results.npy", allow_pickle=True).item()
    sel = slice(0, 12)                                                    # the 12 switched elements
    Us, _, _ = fov_grid(75, 60, 1.5)                                      # search grid
    Ut, azt, elt = fov_grid(60, 40, 4)                                    # truth directions (field of view)
    for tag, f in (("sh_2g4", 2440), ("sh_5g", 5220), ("sh_5g", 5500), ("ns_2g4", 2440), ("ns_5g", 5220)):
        m = Manifold(res[tag][f], f * 1e6)
        A_id = m.ideal(Us)[sel]
        ub = np.array([[0.0, 1.0, 0.0]])
        gv = np.abs(m.resp(ub, pol_vec(ub, 0.0))[sel, 0]); gh = np.abs(m.resp(ub, pol_vec(ub, np.pi / 2))[sel, 0])
        xpd = 20 * np.log10(gh / gv)
        print(f"\n== {tag} {f} MHz  (boresight H/V gain ratio dB by element; + = horizontally polarised)")
        print("   top:", np.round(xpd[0:3], 1), "bottom:", np.round(xpd[3:6], 1), "left:", np.round(xpd[6:9], 1), "right:", np.round(xpd[9:12], 1))
        for name, psi in (("V-pol", 0.0), ("H-pol", np.pi / 2), ("45deg", np.pi / 4)):
            H = m.resp(Ut, pol_vec(Ut, psi))[sel]
            ideal_t = m.ideal(Ut)[sel]
            rel = H * np.conj(ideal_t)                                    # residual after removing ideal steering
            amp_db = 20 * np.log10(np.abs(rel) / np.abs(rel).mean(0, keepdims=True) + 1e-9)
            ph = np.angle(rel * np.conj(np.sum(rel, 0, keepdims=True) / np.abs(np.sum(rel, 0, keepdims=True))))
            corr = np.abs(np.sum(np.conj(ideal_t) * H, 0)) / (np.linalg.norm(ideal_t, axis=0) * np.linalg.norm(H, axis=0))
            e = doa_err(H, A_id, Us, Ut)
            print(f"   {name}: element amp spread (std) {amp_db.std():4.1f} dB | phase dev from ideal (rms) {np.degrees(np.sqrt((ph**2).mean())):5.1f} deg | "
                  f"corr with ideal {np.median(corr):.2f} | DoA err with ideal manifold: median {np.median(e):5.1f} deg, P90 {np.percentile(e,90):5.1f} deg, >15deg {100*(e>15).mean():3.0f}%")
