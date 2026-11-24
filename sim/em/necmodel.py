"""NEC2 (MoM) model of the DeteCam Nano-2 RF board: 84x84 mm ground plane with 13 edge 'loop-notch' elements
(8x4 mm ground clearance bridged by a fed strip = generic model of an edge-mount loop chip antenna such as
AANI-CH-0078) and, optionally, the floating 82x82 mm Ni-Cu shield sheet 8.35 mm behind the board.
Board lies in the X-Z plane, boresight = +Y (same convention as the repository code). Units: mm."""
import numpy as np, subprocess, re, os

P = 4.0; HALF = 42.0
GL = np.round(np.arange(-HALF, HALF + 1e-6, P), 3)          # 22 grid lines
PORT_TAG = 9000
# (edge, coordinate along edge) snapped to the 4 mm grid; order = NANO12_EDGE_M order, then reference
ELEMS = [("top", -26.0), ("top", -6.0), ("top", 22.0), ("bot", -22.0), ("bot", 2.0), ("bot", 22.0),
         ("left", -26.0), ("left", -6.0), ("left", 22.0), ("right", -26.0), ("right", -2.0), ("right", 26.0),
         ("top", 34.0)]                                      # 13th = reference (moved to an edge)
REPO_POS = np.array([[-25.2, 39.0], [-4.5, 39.0], [23.8, 39.0], [-22.5, -39.0], [0.8, -39.0], [21.8, -39.0],
                     [-39.0, -26.5], [-39.0, -5.8], [-39.0, 21.2], [39.0, -27.2], [39.0, -3.0], [39.0, 26.8], [32.0, 32.0]])


def elem_xy(edge, c):
    """Phase-centre guess of the notch element (x, z) in mm: centre of the 8x4 mm clearance."""
    return {"top": (c, HALF - P / 2), "bot": (c, -HALF + P / 2), "left": (-HALF + P / 2, c), "right": (HALF - P / 2, c)}[edge]


def build(shield=True, shield_gap=8.35, elems=ELEMS, board_half=HALF, grid_r=0.64, port_r=0.3):
    """Return (cards list of GW strings, n_ports)."""
    gl = np.round(np.arange(-board_half, board_half + 1e-6, P), 3)
    removed = set(); ports = []
    for edge, c in elems:
        if edge in ("top", "bot"):
            zed = board_half if edge == "top" else -board_half
            zin = zed - P if edge == "top" else zed + P
            removed.add(("x", c - P, zed)); removed.add(("x", c, zed))            # edge segments -> replaced by port wire
            removed.add(("z", c, min(zed, zin)))                                    # interior wire of the clearance
            ports.append(((c - P, 0.0, zed), (c + P, 0.0, zed)))
        else:
            xed = board_half if edge == "right" else -board_half
            xin = xed - P if edge == "right" else xed + P
            removed.add(("z", xed, c - P)); removed.add(("z", xed, c))
            removed.add(("x", min(xed, xin), c))
            ports.append(((xed, 0.0, c - P), (xed, 0.0, c + P)))
    gw = []; tag = 1
    for z in gl:                                                                    # x-directed segments
        for x in gl[:-1]:
            if ("x", x, z) in removed: continue
            gw.append(f"GW {tag} 1 {x:.3f} 0 {z:.3f} {x+P:.3f} 0 {z:.3f} {grid_r}"); tag += 1
    for x in gl:                                                                    # z-directed segments
        for z in gl[:-1]:
            if ("z", x, z) in removed: continue
            gw.append(f"GW {tag} 1 {x:.3f} 0 {z:.3f} {x:.3f} 0 {z+P:.3f} {grid_r}"); tag += 1
    if shield:
        sh = 41.0; ps = 82.0 / 20
        sl = np.round(np.arange(-sh, sh + 1e-6, ps), 3)
        for z in sl:
            for x in sl[:-1]:
                gw.append(f"GW {tag} 1 {x:.3f} {-shield_gap} {z:.3f} {x+ps:.3f} {-shield_gap} {z:.3f} {grid_r}"); tag += 1
        for x in sl:
            for z in sl[:-1]:
                gw.append(f"GW {tag} 1 {x:.3f} {-shield_gap} {z:.3f} {x:.3f} {-shield_gap} {z+ps:.3f} {grid_r}"); tag += 1
    for (a, b) in ports:
        gw.append(f"GW {PORT_TAG} 3 {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} {b[0]:.3f} {b[1]:.3f} {b[2]:.3f} {port_r}")
    return gw, len(ports)


def run(gw, nports, f_mhz, loads, excite, rp=None, name="run", workdir="."):
    """loads: list of (R, L, C) series per port (C=0 -> none). excite: list of port indices to drive one by one.
    rp: (nth, nph, th0, ph0, dth, dph) or None. Returns dict with zin[k], iport[k][j], pattern[k]."""
    L = ["CM DeteCam Nano-2 board", "CE"] + gw + ["GS 0 0 0.001", "GE 0", "EK", f"FR 0 1 0 0 {f_mhz} 0"]
    for k, (R, Lh, C) in enumerate(loads):
        m = 3 * k + 2
        L.append(f"LD 0 {PORT_TAG} {m} {m} {R} {Lh} {C}")
    L.append("LD 5 0 0 0 5.8e7")                       # copper everywhere
    L.append(f"PT 0 {PORT_TAG} 1 {3*nports}")
    for k in excite:
        L.append(f"EX 0 {PORT_TAG} {3*k+2} 0 1.0 0.0")
        if rp is None:
            L.append("XQ 0")
        else:
            nth, nph, th0, ph0, dth, dph = rp
            L.append(f"RP 0 {nth} {nph} 1000 {th0} {ph0} {dth} {dph}")
    L.append("EN")
    fin = os.path.join(workdir, name + ".nec"); fout = os.path.join(workdir, name + ".out")
    open(fin, "w").write("\n".join(L) + "\n")
    subprocess.run(["nec2c", "-i", fin, "-o", fout], check=True)
    return parse(fout, nports, excite, rp)


def parse(fout, nports, excite, rp):
    txt = open(fout).read()
    res = dict(zin={}, iport={}, pat={})
    blocks = re.split(r"-+ ANTENNA INPUT PARAMETERS -+", txt)[1:]
    for k, blk in zip(excite, blocks):
        for ln in blk.split("\n")[1:8]:
            p = ln.split()
            if len(p) >= 11 and p[0] == str(PORT_TAG):
                res["zin"][k] = complex(float(p[6]), float(p[7])); break
        rows = [r.split() for r in blk.split("\n") if re.match(r"\s*\d+\s+%d\s" % PORT_TAG, r)]
        rows = [r for r in rows if len(r) == 10]
        cur = np.zeros(nports, complex)
        if len(rows) >= 3 * nports:
            for j in range(nports):
                r = rows[3 * j + 1]
                cur[j] = complex(float(r[6]), float(r[7]))
        res["iport"][k] = cur
        if rp is not None:
            m2 = re.search(r"VOLTS/M\s+DEGREES\s+VOLTS/M\s+DEGREES\s*\n(.*?)(?:\n\s*\n|\Z)", blk, re.S)
            rows = []
            for r in m2.group(1).split("\n"):
                if not r.strip(): continue
                q = r.replace("LINEAR", " ").replace("RIGHT", " ").replace("LEFT", " ").split()
                rows.append([float(q[0]), float(q[1])] + [float(v) for v in q[-4:]])
            rows = np.array(rows)
            eth = rows[:, 2] * np.exp(1j * np.radians(rows[:, 3])); eph = rows[:, 4] * np.exp(1j * np.radians(rows[:, 5]))
            res["pat"][k] = (rows[:, 0], rows[:, 1], eth, eph)
    return res
