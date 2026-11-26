"""POST-HOC cable layouts (plan sim/plan_20261126_molex.md section 3.2; added after the first G*C numbers).  [simulation]
The 'inward then straight down' cables of molex_heads.py stand as 16 vertical wires just inside the ring: parallel to the
vertical elements and a cage around the centre reference.  Two layouts keep the cables horizontal near the antennas:
  L1 (spokes + trunk, reference above): each cable runs horizontally inward at mid height to the ring axis, the 16
     cables go down the axis as one bundle (one wire, radius 2 mm) to the box top; the reference antenna is raised
     45 mm above the ring plane and its own cable runs 10 mm off the axis down to the hub.
  L2 (spokes + trunk, reference under the box): same spokes and trunk; the reference hangs under the box (its top end
     20 mm below the box bottom) and its cable runs 34 mm sideways and up to the box bottom.
Box top 58.9 mm below the element centre (as G30).  Same NEC model as ring_pynec.simulate_ring(kind="dipole").
Run: python3 mysim/molex_layouts.py mysim/nec_molex2.npy [keys]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402

LAYOUT = None
_orig_box = R._box_wires
RAD = 0.565          # 1.13 mm coax shield
SEG = 6.0
HL = None            # element half length, set per band


def _n(p, q):
    return max(1, int(round(float(np.linalg.norm(np.subtract(q, p))) / SEG)))


def _w(p, q, rad=RAD):
    return (_n(p, q), tuple(map(float, p)), tuple(map(float, q)), rad)


# L5 (POST-HOC, plan 3.4): 17 places on the D120 circle, place 16 is the reference antenna (wired to chain A), the
# other 16 are switched.  The NEC origin sits on the reference (the pattern classes assume the reference on the z axis);
# the box stays centred under the ring centre C.
PHI17 = 2.0 * np.pi * np.arange(17) / 17
C5 = -60.0 * np.array([np.cos(PHI17[16]), np.sin(PHI17[16])])
P17 = C5[None, :] + 60.0 * np.stack([np.cos(PHI17), np.sin(PHI17)], axis=1)      # P17[16] = (0, 0)


def _with_layout(z_top_mm, half_w=45.0, height=35.0, mesh_mm=11.25, sides=True, rad_mm=None):
    w = _orig_box(z_top_mm, half_w=half_w, height=height, mesh_mm=mesh_mm, sides=sides, rad_mm=rad_mm)
    if LAYOUT is None:
        return w
    if LAYOUT == "L5":
        cx, cy = float(C5[0]), float(C5[1])
        w = [(n, (p[0] + cx, p[1] + cy, p[2]), (q[0] + cx, q[1] + cy, q[2]), r) for n, p, q, r in w]
        hub = (cx, cy, 0.0)
        for k in range(17):
            u = np.array([np.cos(PHI17[k]), np.sin(PHI17[k])])
            p0 = C5 + 54.0 * u
            w.append(_w((p0[0], p0[1], 0.0), hub))
        w.append(_w(hub, (cx, cy, z_top_mm), rad=2.0))
        return w
    hub = (0.0, 0.0, 0.0)
    for k in range(16):                                   # 16 horizontal spokes at mid height
        a = 2.0 * np.pi * k / 16
        w.append(_w((54.0 * np.cos(a), 54.0 * np.sin(a), 0.0), hub))
    w.append(_w(hub, (0.0, 0.0, z_top_mm), rad=2.0))      # the bundle (one wire, radius 2 mm) down the axis to the box-top centre node
    b = np.radians(11.25)                                 # reference cable between two spokes
    if LAYOUT == "L1":
        zc = 45.0
        p0 = (6.0 * np.cos(b), 6.0 * np.sin(b), zc)
        p1 = (10.0 * np.cos(b), 10.0 * np.sin(b), zc)
        p2 = (10.0 * np.cos(b), 10.0 * np.sin(b), 0.0)
        w += [_w(p0, p1), _w(p1, p2), _w(p2, hub)]
    elif LAYOUT == "L2":
        z_bot = z_top_mm - height
        zc = z_bot - 20.0 - HL
        x_node = 33.75                                    # a node of the box-bottom grid
        w += [_w((6.0, 0.0, zc), (x_node, 0.0, zc)), _w((x_node, 0.0, zc), (x_node, 0.0, z_bot))]
    return w


R._box_wires = _with_layout

BASE = R.RING_SPECS["d120_16"]
GAP = 30.0


def spec(layout, f_mhz=2437.0):
    hl = 0.235 * (R.C0 / (f_mhz * 1e6) * 1e3)
    if layout in ("L1", "R45", "L3"):
        ref_dz = 45.0
    elif layout == "I30":
        ref_dz = 0.0
    else:
        ref_dz = -(hl + GAP) - 35.0 - 20.0 - hl
    cfg = dict(BASE, box_gap_mm=GAP, ref_dz_mm=ref_dz)
    if layout in ("L5", "L6"):
        cfg = dict(n_elem=16, diam_mm=0.0, ref_dz_mm=0.0, box_gap_mm=GAP, kind="custom",
                   elem_xyz_mm=[[float(x), float(y), 0.0] for x, y in P17[:16]])
    if layout in ("L3", "I30", "L6"):
        cfg["invert"] = True            # mirror z: box above, L1 cables go up (ref_dz 45 becomes 45 mm below the ring)
    return cfg, hl


SPECS = {}
for lay in ("L1", "L2", "R45"):          # R45: reference raised 45 mm, no cables (separates the two changes of L1)
    for t in (0, 20, -20):
        SPECS[f"d120_16{lay.lower()}" + ("" if t == 0 else f"_ty{t:+d}")] = (lay, float(t))
# POST-HOC (plan 3.3): the same heads turned upside down (box above the ring): L3 = L1 inverted (reference 45 mm below
# the ring, spokes + trunk going up to the box), I30 = G30 inverted (box above, reference at the centre, no cables)
for lay, key in (("L3", "d120_16l3"), ("I30", "d120_16i30"), ("L5", "d120_16l5"), ("L6", "d120_16l6")):   # L6 = L5 upside down
    for t in (0, 20, -20):
        SPECS[key + ("" if t == 0 else f"_ty{t:+d}")] = (lay, float(t))


def build(keys, out_path, f_mhz=2437.0, band="2g4"):
    global LAYOUT, HL
    out = np.load(out_path, allow_pickle=True).item() if os.path.exists(out_path) else {}
    for key in keys:
        if (key, band) in out:
            continue
        lay, t = SPECS[key]
        cfg, HL = spec(lay, f_mhz)
        if t:
            cfg = dict(cfg, tilt_deg=t, tilt_axis="y")
        LAYOUT = {"R45": None, "I30": None, "L3": "L1", "L6": "L5"}.get(lay, lay)
        res = R.simulate_ring(f_mhz=f_mhz, **cfg)
        LAYOUT = None
        if cfg.get("invert"):
            res["ref_dz_m"] = -res["ref_dz_m"]     # simulate_ring reports the un-mirrored height
        out[(key, band)] = res
        print(f"[NEC2/PyNEC] {key:22s} {band} | S11 ring mean={res['s11_db'].mean():6.1f} dB  ref={res['ref_s11_db']:6.1f} dB"
              f" | Zin ring={res['zin'].mean():.1f} | scale={res['scale']:.3f}", flush=True)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, out, allow_pickle=True)
        os.replace(tmp, out_path)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_molex2.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SPECS)
    build(keys, out)
    print("MOLEX2_NEC_DONE", keys, flush=True)
