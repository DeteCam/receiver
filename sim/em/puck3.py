"""Where to put the fixed reference dipole and the metal body of the rotating receiver (NEC2).
 dz  = height of the reference dipole (on the rotation axis) above the moving dipole, mm (negative = below)
 gap = distance between the lower end of the moving dipole and the metal body (56 x 56 x 28 mm box), mm
Saves puck3_results.npy {(MHz, body present, dz, gap): patterns} for the 44 mm arm and puck3_r90_results.npy for the 90 mm arm, and prints how
far the phase-referenced response (moving / reference) is from the ideal ring model, for waves from above and from below the horizon."""
import numpy as np, puck2
from puck_analysis_lib import analyse
C0 = 299792458.0

def build3(f_mhz, rmov=44.0, box=False, gap=15.0, dz=0.0):
    lam = C0 / (f_mhz * 1e6) * 1e3; hl = 0.235 * lam; gw = puck2_build(f_mhz, rmov=rmov, box=box, gap=gap)
    return [g if not g.startswith("GW 9001 ") else f"GW 9001 9 0 0 {dz - hl:.3f} 0 0 {dz + hl:.3f} 0.5" for g in gw]
puck2_build = puck2.build; puck2.build = build3

def closeness(r, f, its=range(8, 19), R=0.044):
    """(worst, mean) correlation of moving / reference with the ideal ring model, and the worst cross-polar level (dB), over the polar
    angles 5 * its degrees (8..18 = 50 deg above the horizon ... horizon, 18..28 = horizon ... 50 deg below)."""
    k = 2 * np.pi * f * 1e6 / C0; ph = np.radians(np.arange(0, 360, 5)); out = []; xp = []
    for it in its:
        th = np.radians(5 * it); x = r[9002]["eth"][:, it] / r[9001]["eth"][:, it]; a = np.exp(1j * k * R * np.sin(th) * np.cos(ph))
        out.append(abs(np.vdot(a, x)) / np.linalg.norm(a) / np.linalg.norm(x))
        xp.append(20 * np.log10(np.abs(r[9002]["eph"][:, it]).max() / np.abs(r[9002]["eth"][:, it]).mean() + 1e-9))
    return min(out), float(np.mean(out)), max(xp)

if __name__ == "__main__":
    res = {}
    for f in (2440, 5220):
        for box, dz, gap in [(False, d, 15.0) for d in (0.0, 40.0, 65.0, 90.0)] + [(True, d, 15.0) for d in (0.0, 40.0, 65.0, 90.0)] + [(True, -65.0, 75.0), (True, 65.0, 60.0), (True, 65.0, 100.0), (True, -65.0, 115.0)]:
            r = puck2.run(f, f"p3_{f}_{int(dz)}_{int(gap)}{'_box' if box else ''}", box=box, dz=dz, gap=gap); res[(f, box, dz, gap)] = r
            up, dn = closeness(r, f), closeness(r, f, range(18, 29))
            print(f"{f} MHz {'metal body' if box else 'free      '} reference {dz:+4.0f} mm, body {gap:4.0f} mm below: from above match min {up[0]:.3f} mean {up[1]:.3f} x-pol {up[2]:6.1f} dB"
                  f" | from below min {dn[0]:.3f} mean {dn[1]:.3f} x-pol {dn[2]:6.1f} dB", flush=True)
    np.save("puck3_results.npy", res, allow_pickle=True)
    r90 = {}
    for f in (2440, 5220):
        r = puck2.run(f, f"p3_{f}_r90", box=True, dz=65.0, gap=100.0, rmov=90.0); r90[(f, True, 65.0, 100.0)] = r; c = closeness(r, f, range(8, 29), R=0.090)
        print(f"{f} MHz 90 mm arm, reference +65 mm, body 100 mm below: match min {c[0]:.3f} mean {c[1]:.3f} x-pol {c[2]:.1f} dB", flush=True)
    np.save("puck3_r90_results.npy", r90, allow_pickle=True)
    for key in ((2440, True, 0.0, 15.0), (2440, True, 65.0, 100.0), (5220, True, 0.0, 15.0), (5220, True, 65.0, 100.0)):
        analyse(res[key], key[0], 0.044, f"two dipoles, reference {key[2]:+.0f} mm, metal body {key[3]:.0f} mm below")
