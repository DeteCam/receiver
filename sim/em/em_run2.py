"""Co-tuned (iteratively conjugate-matched with all neighbours terminated) NEC2 characterisation."""
import numpy as np, time
import necmodel as nm
TH = np.arange(0, 181, 5.0); PH = np.arange(0, 360, 5.0); RP = (len(TH), len(PH), 0, 0, 5, 5)

def cotune(gw, n, f, iters=6):
    w = 2 * np.pi * f * 1e6
    r0 = nm.run(gw, n, f, [(0, 0, 0)] * n, list(range(n)), name="tune")
    Zl = np.conj(np.array([r0["zin"][k] for k in range(n)]))                 # load = conj(Z_ant)
    for it in range(iters):
        loads = [(float(max(Zl[k].real, 0.2)), 0.0, float(-1.0 / (w * Zl[k].imag))) for k in range(n)]
        r = nm.run(gw, n, f, loads, list(range(n)), name="tune")
        Zact = np.array([r["zin"][k] for k in range(n)]) - Zl                  # active antenna impedance (others terminated)
        Zl = 0.5 * Zl + 0.5 * np.conj(Zact)
    return [(float(max(Zl[k].real, 0.2)), 0.0, float(-1.0 / (w * Zl[k].imag))) for k in range(n)]

def characterise(shield, f_tune, f_eval):
    gw, n = nm.build(shield=shield)
    loads = cotune(gw, n, f_tune)
    R = np.array([l[0] for l in loads]); out = {}
    for f in f_eval:
        r = nm.run(gw, n, f, loads, list(range(n)), rp=RP, name="b")
        I = np.array([r["iport"][k] for k in range(n)])
        T = 2 * np.sqrt(np.outer(R, R)) * np.abs(I)                           # |S_jk| for j != k
        refl = np.abs(1 - 2 * R * np.diag(I))
        Pav = 1.0 / (8 * R)
        th, ph = r["pat"][0][0], r["pat"][0][1]
        Eth = np.array([r["pat"][k][2] for k in range(n)]) / np.sqrt(Pav)[:, None]
        Eph = np.array([r["pat"][k][3] for k in range(n)]) / np.sqrt(Pav)[:, None]
        dOm = np.sin(np.radians(th)) * np.radians(5) ** 2
        eff = np.sum((np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / (2 * 376.73) * dOm[None], axis=1)
        out[f] = dict(th=th, ph=ph, Eth=Eth, Eph=Eph, T=T, refl=refl, eff=eff, R=R)
    return out

if __name__ == "__main__":
    res = {}; t0 = time.time()
    for shield in (True, False):
        for band, ft, fe in (("2g4", 2440, [2412, 2440, 2472]), ("5g", 5500, [5220, 5500, 5785])):
            tag = f"{'sh' if shield else 'ns'}_{band}"; res[tag] = characterise(shield, ft, fe)
            for f in fe:
                d = res[tag][f]; off = d["T"][~np.eye(13, dtype=bool)]
                print(f"{tag} {f} | R {d['R'].min():.1f}-{d['R'].max():.1f} | eff {100*d['eff'].min():.0f}-{100*d['eff'].max():.0f}% mean {100*d['eff'].mean():.0f}% | "
                      f"refl worst {20*np.log10(d['refl'].max()+1e-9):.1f} dB | coupling max {20*np.log10(off.max()):.1f} mean {20*np.log10(np.sqrt((off**2).mean())):.1f} dB | {time.time()-t0:.0f}s", flush=True)
    np.save("em_results.npy", res, allow_pickle=True)
