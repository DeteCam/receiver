"""NEC2 model of the 'rotating puck' front end: circular ground disc, reference monopole at the centre, moving monopole near the rim.
The whole puck rotates, so the complex ratio (moving / reference) seen from a fixed emitter depends only on the relative azimuth alpha
and the elevation -> a single periodic function instead of a per-element 2-D dual-polarised manifold.
Units mm. Disc in the X-Y plane, antennas along +Z (the antenna side faces the room)."""
import numpy as np, subprocess, re, sys
C0 = 299792458.0; P = 4.0

def build(f_mhz, rg=48.0, rmov=44.0, box=False):
    lam = C0 / (f_mhz * 1e6) * 1e3; h = 0.235 * lam; nseg = 5
    n = int(rg // P); nodes = {(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1) if (i * P) ** 2 + (j * P) ** 2 <= rg ** 2 + 1e-6}
    gw = []; tag = 1
    for (i, j) in sorted(nodes):
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in nodes:
                gw.append(f"GW {tag} 1 {i*P:.3f} {j*P:.3f} 0 {(i+di)*P:.3f} {(j+dj)*P:.3f} 0 0.64"); tag += 1
    if box:                                                   # crude metal body (host board + battery) behind the disc: 56 x 56 x 28 mm wire box
        b = 28.0; z1 = -4.0; z2 = -32.0; pts = [(-b, -b), (b, -b), (b, b), (-b, b)]
        for z in (z1, z2):
            for k in range(4):
                (x0, y0), (x1, y1) = pts[k], pts[(k + 1) % 4]
                gw.append(f"GW {tag} 7 {x0} {y0} {z} {x1} {y1} {z} 0.8"); tag += 1
        for (x0, y0) in pts:
            gw.append(f"GW {tag} 4 {x0} {y0} {z1} {x0} {y0} {z2} 0.8"); tag += 1
            gw.append(f"GW {tag} 1 {x0} {y0} 0 {x0} {y0} {z1} 0.8"); tag += 1      # bonded to the disc at 4 points
    gw.append(f"GW 9001 {nseg} 0 0 0 0 0 {h:.3f} 0.5")
    gw.append(f"GW 9002 {nseg} {rmov:.3f} 0 0 {rmov:.3f} 0 {h:.3f} 0.5")
    return gw, h

def run(f_mhz, name, **kw):
    gw, h = build(f_mhz, **kw); out = {}
    for drive, other in ((9001, 9002), (9002, 9001)):
        L = ["CM puck", "CE"] + gw + ["GS 0 0 0.001", "GE 0", "EK", f"FR 0 1 0 0 {f_mhz} 0", f"LD 0 {other} 1 1 50 0 0", "LD 5 0 0 0 5.8e7",
                                     f"EX 0 {drive} 1 0 1.0 0.0", "RP 0 37 72 1000 0 0 5 5", "EN"]
        fin = f"{name}_{drive}.nec"; fout = f"{name}_{drive}.out"; open(fin, "w").write("\n".join(L) + "\n")
        subprocess.run(["nec2c", "-i", fin, "-o", fout], check=True)
        txt = open(fout).read()
        zin = None
        for ln in txt.split("ANTENNA INPUT PARAMETERS")[1].split("\n")[1:8]:
            p = ln.split()
            if len(p) >= 11 and p[0] == str(drive): zin = complex(float(p[6]), float(p[7])); break
        m2 = re.search(r"VOLTS/M\s+DEGREES\s+VOLTS/M\s+DEGREES\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S); rows = []
        for r in m2.group(1).split("\n"):
            if not r.strip(): continue
            q = r.replace("LINEAR", " ").replace("RIGHT", " ").replace("LEFT", " ").split(); rows.append([float(q[0]), float(q[1])] + [float(v) for v in q[-4:]])
        rows = np.array(rows)
        eth = (rows[:, 2] * np.exp(1j * np.radians(rows[:, 3]))).reshape(72, 37); eph = (rows[:, 4] * np.exp(1j * np.radians(rows[:, 5]))).reshape(72, 37)
        mref = re.search(r"RADIATED POWER\s*=\s*([-\d.E+]+)", txt); minp = re.search(r"INPUT POWER\s*=\s*([-\d.E+]+)", txt)
        out[drive] = dict(zin=zin, eth=eth, eph=eph, eff=float(mref.group(1)) / float(minp.group(1)) if mref and minp else np.nan)
    return out

if __name__ == "__main__":
    res = {}
    for f in (2440, 5220):
        for box in (False, True):
            r = run(f, f"puck_{f}{'_box' if box else ''}", box=box); res[(f, box)] = r
            print(f, "box" if box else "   ", "Zin ref", np.round(r[9001]["zin"], 1), "mov", np.round(r[9002]["zin"], 1), "eff", round(r[9001]["eff"], 2), round(r[9002]["eff"], 2), flush=True)
    np.save("puck_results.npy", res, allow_pickle=True)
