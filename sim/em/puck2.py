"""Variant without a ground disc: two vertical half-wave (sleeve-type) dipoles on a non-metallic arm, optional metal body nearby.
Compares cross-polarisation and azimuth bias with the monopole-on-disc version (puck.py). Units mm."""
import numpy as np, subprocess, re
from puck_analysis_lib import analyse
C0 = 299792458.0

def build(f_mhz, rmov=44.0, box=False, gap=15.0):
    lam = C0 / (f_mhz * 1e6) * 1e3; hl = 0.235 * lam; gw = []; tag = 1
    if box:                                                   # floating metal body (host board + battery + motor), 56 x 56 x 28 mm, below the dipoles
        b = 28.0; z1 = -(hl + gap); z2 = z1 - 28.0; pts = [(-b, -b), (b, -b), (b, b), (-b, b)]
        for z in (z1, z2):
            for k in range(4):
                (x0, y0), (x1, y1) = pts[k], pts[(k + 1) % 4]; gw.append(f"GW {tag} 7 {x0} {y0} {z} {x1} {y1} {z} 0.8"); tag += 1
            for t in (-14.0, 0.0, 14.0):                        # a few cross wires to approximate the plates
                gw.append(f"GW {tag} 7 {-b} {t} {z} {b} {t} {z} 0.8"); tag += 1; gw.append(f"GW {tag} 7 {t} {-b} {z} {t} {b} {z} 0.8"); tag += 1
        for (x0, y0) in pts: gw.append(f"GW {tag} 4 {x0} {y0} {z1} {x0} {y0} {z2} 0.8"); tag += 1
    gw.append(f"GW 9001 9 0 0 {-hl:.3f} 0 0 {hl:.3f} 0.5"); gw.append(f"GW 9002 9 {rmov:.3f} 0 {-hl:.3f} {rmov:.3f} 0 {hl:.3f} 0.5")
    return gw

def run(f_mhz, name, **kw):
    gw = build(f_mhz, **kw); out = {}
    for drive, other in ((9001, 9002), (9002, 9001)):
        L = ["CM puck dipoles", "CE"] + gw + ["GS 0 0 0.001", "GE 0", "EK", f"FR 0 1 0 0 {f_mhz} 0", f"LD 0 {other} 5 5 50 0 0", "LD 5 0 0 0 5.8e7",
                                             f"EX 0 {drive} 5 0 1.0 0.0", "RP 0 37 72 1000 0 0 5 5", "EN"]
        fin = f"{name}_{drive}.nec"; fout = f"{name}_{drive}.out"; open(fin, "w").write("\n".join(L) + "\n"); subprocess.run(["nec2c", "-i", fin, "-o", fout], check=True)
        txt = open(fout).read(); zin = None
        for ln in txt.split("ANTENNA INPUT PARAMETERS")[1].split("\n")[1:8]:
            p = ln.split()
            if len(p) >= 11 and p[0] == str(drive): zin = complex(float(p[6]), float(p[7])); break
        m2 = re.search(r"VOLTS/M\s+DEGREES\s+VOLTS/M\s+DEGREES\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S); rows = []
        for r in m2.group(1).split("\n"):
            if not r.strip(): continue
            q = r.replace("LINEAR", " ").replace("RIGHT", " ").replace("LEFT", " ").split(); rows.append([float(q[0]), float(q[1])] + [float(v) for v in q[-4:]])
        rows = np.array(rows)
        out[drive] = dict(zin=zin, eth=(rows[:, 2] * np.exp(1j * np.radians(rows[:, 3]))).reshape(72, 37), eph=(rows[:, 4] * np.exp(1j * np.radians(rows[:, 5]))).reshape(72, 37))
    return out

if __name__ == "__main__":
    res = {}
    for f in (2440, 5220):
        for box in (False, True):
            r = run(f, f"dip_{f}{'_box' if box else ''}", box=box); res[(f, box)] = r
            print(f, "box" if box else "   ", "Zin ref", np.round(r[9001]["zin"], 1), "mov", np.round(r[9002]["zin"], 1), flush=True)
    np.save("puck2_results.npy", res, allow_pickle=True)
    for key, r in res.items(): analyse(r, key[0], 0.044, f"two dipoles, {'metal body 15 mm below' if key[1] else 'free standing'}")
