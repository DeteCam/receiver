"""Bottom-fed rod antennas and film-cable variants (plan sim/plan_20261126_rod.md).  [simulation]
The film antenna (Molex 146153) has its cable leaving the middle of the long side, so the 16 cables meet on the ring
axis and go down as a trunk through the middle of the ring.  A sleeve-dipole rod is fed from its lower end: the rod
stands on a plate, every cable stays below the plate and the reference antenna keeps its place in the ring plane.
  F1  film, spokes + trunk (as L1 of molex_layouts.py), reference 45 mm above, box top 88.9 mm below the feed height
  F2  F1 with a series loss on every trunk segment (100 ohm per 6 mm): absorber / ferrite upper bound, an assumption
  R1  rods, feed point 45 mm above the plate; under each element a vertical wire (connector + coax shield) starts
      3 mm below the element tip (open: the sleeve choke), passes the plate and ends 12 mm below it, runs inward to
      radius ~30 mm and drops to the box top (bonded).  Reference rod on the axis, its cable straight down.  Box -75 mm
  R2  R1 with the vertical wire joined to the element tip through 150 ohm (a leaking choke)
  R3  R1 for a longer rod: feed point 60 mm above the plate, box top -90 mm
Same NEC model as ring_pynec.simulate_ring(kind="dipole"); all lengths in mm, z = 0 at the feed height.
Run: python3 mysim/rod_heads.py mysim/nec_rod.npy [keys] [band]"""
import os
import sys
import numpy as np
from PyNEC import nec_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402

PHI17 = 2.0 * np.pi * np.arange(17) / 17      # "l5": 17 places on the D120 circle, place 16 = reference at the NEC origin
C5 = -60.0 * np.array([np.cos(PHI17[16]), np.sin(PHI17[16])])
P17 = C5[None, :] + 60.0 * np.stack([np.cos(PHI17), np.sin(PHI17)], axis=1)
LAY = None            # layout dict of the head being built
SEG = 6.0             # cable segment length [mm]
HL = None             # element half length [mm]
EXTRA_LOADS = []      # (tag, first segment, last segment, ohm), filled while the box wires are built
OFF_Z = None          # impedance seen by a switched-off ring element (None -> term_r of simulate_ring, 50 ohm)
_orig_box = R._box_wires
GRID = np.linspace(-45.0, 45.0, 9)          # nodes of the 2.4 GHz box-top grid (also nodes of the 5 GHz grid)


THIN_FIX = False      # True: segments of thick wires are at least 2.5 radii long (NEC thin-wire rule; 5 GHz runs)


def _n(p, q, rad=0.0):
    seg = max(SEG, 2.5 * rad) if THIN_FIX else SEG
    return max(1, int(round(float(np.linalg.norm(np.subtract(q, p))) / seg)))


def _w(p, q, rad):
    return (_n(p, q, rad), tuple(map(float, p)), tuple(map(float, q)), rad)


def _seg_dist0(p, q):
    """Distance from the origin to the segment p-q (2-D)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    t = float(np.clip(-np.dot(p, q - p) / np.dot(q - p, q - p), 0.0, 1.0))
    return float(np.linalg.norm(p + t * (q - p)))


def _node(x, y):
    return float(GRID[int(np.argmin(np.abs(GRID - x)))]), float(GRID[int(np.argmin(np.abs(GRID - y)))])


def _rect_box(z_top, x0, x1, hy, height, mesh):
    """Wire-grid box with a rectangular footprint x0..x1 by -hy..hy (all nodes on the common mesh)."""
    nx = int(round((x1 - x0) / mesh))
    ny = int(round(2 * hy / mesh))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(-hy, hy, ny + 1)
    rad = mesh / (2.0 * np.pi)
    z1, z2 = z_top, z_top - height
    w = []
    for z in (z1, z2):
        for y in ys:
            for i in range(nx):
                w.append((1, (xs[i], y, z), (xs[i + 1], y, z), rad))
        for x in xs:
            for j in range(ny):
                w.append((1, (x, ys[j], z), (x, ys[j + 1], z), rad))
    nz = max(1, int(round(height / mesh)))
    zs = np.linspace(z2, z1, nz + 1)
    rim = [(x, -hy) for x in xs] + [(x, hy) for x in xs] + [(x0, y) for y in ys[1:-1]] + [(x1, y) for y in ys[1:-1]]
    for (x, y) in rim:
        for k in range(nz):
            w.append((1, (x, y, zs[k]), (x, y, zs[k + 1]), rad))
    for z in zs[1:-1]:
        for i in range(nx):
            w.append((1, (xs[i], -hy, z), (xs[i + 1], -hy, z), rad))
            w.append((1, (xs[i], hy, z), (xs[i + 1], hy, z), rad))
        for j in range(ny):
            w.append((1, (x0, ys[j], z), (x0, ys[j + 1], z), rad))
            w.append((1, (x1, ys[j], z), (x1, ys[j + 1], z), rad))
    return w


def _with_layout(z_top_mm, half_w=45.0, height=35.0, mesh_mm=11.25, sides=True, rad_mm=None):
    if LAY is not None and LAY.get("box"):
        bx0, bx1, bhy, bh = LAY["box"]
        w = _rect_box(z_top_mm, bx0, bx1, bhy, bh, mesh_mm)
    else:
        w = _orig_box(z_top_mm, half_w=half_w, height=height, mesh_mm=mesh_mm, sides=sides, rad_mm=rad_mm)
    del EXTRA_LOADS[:]
    if LAY is None:
        return w
    if LAY.get("mirror_x"):          # the same head turned by 180 deg about the vertical axis (built mirrored in x and y)
        LAY2 = dict(LAY); LAY2.pop("mirror_x")
        return _mirrored(z_top_mm, half_w, height, mesh_mm, sides, rad_mm, LAY2)
    if LAY["kind"] == "film":
        hub = (0.0, 0.0, 0.0)
        for k in range(16):
            a = 2.0 * np.pi * k / 16
            w.append(_w((54.0 * np.cos(a), 54.0 * np.sin(a), 0.0), hub, 0.565))
        w.append(_w(hub, (0.0, 0.0, z_top_mm), float(LAY.get("trunk_rad", 2.0))))   # the bundle of 16 cables as one wire
        if LAY.get("trunk_ohm"):
            EXTRA_LOADS.append((len(w), 1, w[-1][0], float(LAY["trunk_ohm"])))
        b = np.radians(11.25)
        zc = float(LAY["ref_dz"])
        rc = LAY.get("ref_cable", "side")
        if rc == "side":            # film reference: cable drops beside the element, ref_cable_r mm from the axis
            r1 = float(LAY.get("ref_cable_r", 10.0))
            p0 = (6.0 * np.cos(b), 6.0 * np.sin(b), zc)
            p1 = (r1 * np.cos(b), r1 * np.sin(b), zc)
            p2 = (r1 * np.cos(b), r1 * np.sin(b), 0.0)
            w += [_w(p0, p1, 0.565), _w(p1, p2, 0.565), _w(p2, hub, 0.565)]
        elif rc == "axis_gap":      # rod reference fed from below: cable on the axis, 3 mm under the element tip
            w.append(_w((0.0, 0.0, zc - HL - 3.0), hub, 1.0))
        elif rc == "axis_ohm":      # the same with a leaking choke: joined to the tip through a series resistor
            w.append(_w((0.0, 0.0, zc - HL), hub, 1.0))
            EXTRA_LOADS.append((len(w), 1, 1, float(LAY.get("ref_choke_ohm", 150.0))))
        elif rc != "none":
            raise ValueError(rc)
        return w
    if LAY["kind"] == "bare":
        return w
    if LAY["kind"] == "cone":
        # spokes leave the ring plane and meet on the axis 3 mm under the lower tip of the in-plane centre reference
        z_hub = -(HL + 3.0)
        hub = (0.0, 0.0, z_hub)
        for k in range(16):
            a = 2.0 * np.pi * k / 16
            w.append(_w((54.0 * np.cos(a), 54.0 * np.sin(a), 0.0), hub, 0.565))
        w.append(_w(hub, (0.0, 0.0, z_top_mm), float(LAY.get("trunk_rad", 2.0))))
        if LAY.get("ref_cable") == "side":      # film reference: its cable drops beside the lower half of the element
            b = np.radians(11.25)
            p0 = (6.0 * np.cos(b), 6.0 * np.sin(b), 0.0)
            p1 = (10.0 * np.cos(b), 10.0 * np.sin(b), 0.0)
            p2 = (10.0 * np.cos(b), 10.0 * np.sin(b), z_hub)
            w += [_w(p0, p1, 0.565), _w(p1, p2, 0.565), _w(p2, hub, 0.565)]
        return w
    if LAY["kind"] == "l5":
        cx, cy = float(C5[0]), float(C5[1])
        w = [(n, (p[0] + cx, p[1] + cy, p[2]), (q[0] + cx, q[1] + cy, q[2]), r) for n, p, q, r in w]
        hub = (cx, cy, 0.0)
        for k in range(17):
            u = np.array([np.cos(PHI17[k]), np.sin(PHI17[k])])
            p0 = C5 + 54.0 * u
            w.append(_w((p0[0], p0[1], 0.0), hub, 0.565))
        w.append(_w(hub, (cx, cy, z_top_mm), 2.0))
        return w
    if LAY["kind"] == "back":
        # herringbone in the ring plane: a spine 11.25 mm beside the axis collects the 16 cables and the reference
        # cable, one trunk goes down at the back rim node (-45, -11.25) of the box top
        ys = -11.25
        S = [(54.0 * np.cos(2.0 * np.pi * k / 16), 54.0 * np.sin(2.0 * np.pi * k / 16), 0.0) for k in range(16)]
        node = {0: 50.0, 1: 45.0, 15: 45.0, 2: 33.75, 14: 33.75, 3: 22.5, 13: 22.5, 4: 11.25, 12: 11.25,
                5: -11.25, 11: -11.25, 6: -33.75, 10: -33.75, 7: -45.0, 8: -45.0, 9: -45.0}
        xs = [50.0, 45.0, 33.75, 22.5, 11.25, 0.0, -11.25, -33.75, -45.0]
        if LAY.get("chevron"):        # ribs slanted towards the back (shorter cables): spine node = x - slope * |y - ys|,
            sl = float(LAY["chevron"])   # snapped to 5.625 mm so that no spine piece is shorter than 2.5 radii
            node = {k: float(np.clip(5.625 * round((S[k][0] - sl * abs(S[k][1] - ys)) / 5.625), -45.0, 45.0))
                    for k in range(16)}
            for k in range(16):       # a rib must pass the reference film at the centre by 10 mm or more
                while _seg_dist0(S[k][:2], (node[k], ys)) < 10.0 and node[k] > -45.0:
                    node[k] -= 5.625
            xs = sorted(set(node.values()) | {0.0, -45.0}, reverse=True)
        for k in range(16):
            w.append(_w(S[k], (node[k], ys, 0.0), 0.565))
        for a, b2 in zip(xs[:-1], xs[1:]):
            w.append(_w((a, ys, 0.0), (b2, ys, 0.0), 1.5))
        w.append(_w((0.0, -6.0, 0.0), (0.0, ys, 0.0), 0.565))                       # reference cable (film at the centre)
        tr = float(LAY.get("trunk_rad", 2.0))
        if LAY.get("trunk_out"):      # bundle leaves the ring between two back elements and drops outside it
            xo = -float(LAY["trunk_out"])
            w.append(_w((-45.0, ys, 0.0), (xo, ys, 0.0), 1.5))
            w.append(_w((xo, ys, 0.0), (xo, ys, z_top_mm), tr))
            if LAY.get("trunk_ohm"):
                EXTRA_LOADS.append((len(w), 1, w[-1][0], float(LAY["trunk_ohm"])))
            if not (LAY.get("box") and LAY["box"][0] <= xo <= LAY["box"][1]):   # box top does not reach the bundle:
                w.append(_w((xo, ys, z_top_mm), (-45.0, ys, z_top_mm), tr))      # run back to the box edge at its top
        elif not LAY.get("no_trunk"):
            w.append(_w((-45.0, ys, 0.0), (-45.0, ys, z_top_mm), tr))
            if LAY.get("trunk_ohm"):
                EXTRA_LOADS.append((len(w), 1, w[-1][0], float(LAY["trunk_ohm"])))
        return w
    # rods on a plate
    z_plate = -float(LAY["d_base"])
    z_c = z_plate - 12.0                                          # horizontal cable run under the plate
    choke = LAY.get("choke_ohm")
    z_start = -HL if choke else -(HL + 3.0)
    used = set()
    for k in range(16):
        a = 2.0 * np.pi * k / 16
        x, y = 60.0 * np.cos(a), 60.0 * np.sin(a)
        gx, gy = _node(30.0 * np.cos(a), 30.0 * np.sin(a))
        if (gx, gy) in used or (gx, gy) == (0.0, 0.0):
            raise ValueError(f"two cables on one box node {(gx, gy)}")
        used.add((gx, gy))
        w.append(_w((x, y, z_start), (x, y, z_c), 1.0))
        if choke:
            EXTRA_LOADS.append((len(w), 1, 1, float(choke)))
        w.append(_w((x, y, z_c), (gx, gy, z_c), 1.0))
        w.append(_w((gx, gy, z_c), (gx, gy, z_top_mm), 1.0))
    w.append(_w((0.0, 0.0, z_start), (0.0, 0.0, z_top_mm), 1.0))     # reference rod: cable straight down the axis
    if choke:
        EXTRA_LOADS.append((len(w), 1, 1, float(choke)))
    return w


def _mirrored(z_top_mm, half_w, height, mesh_mm, sides, rad_mm, lay2):
    global LAY
    keep = LAY
    LAY = lay2
    try:
        w = _with_layout(z_top_mm, half_w=half_w, height=height, mesh_mm=mesh_mm, sides=sides, rad_mm=rad_mm)
    finally:
        LAY = keep
    return [(n, (-p[0], -p[1], p[2]), (-q[0], -q[1], q[2]), r) for n, p, q, r in w]


def _solve(wires, ports, drive_idx, f_mhz, term_r, ref_load=50.0, ref_idx=0):
    """ring_pynec._solve with the extra series loads of the layout (cable chokes, lossy trunk)."""
    ctx = nec_context()
    geo = ctx.get_geometry()
    for tag, nseg, p0, p1, rad in wires:
        geo.wire(int(tag), int(nseg), p0[0] * R.MM, p0[1] * R.MM, p0[2] * R.MM, p1[0] * R.MM, p1[1] * R.MM,
                 p1[2] * R.MM, rad * R.MM, 1.0, 1.0)
    ctx.geometry_complete(0)
    ctx.set_extended_thin_wire_kernel(True)
    for i, (tag, seg) in enumerate(ports):
        if i == drive_idx:
            continue
        if i != ref_idx and OFF_Z is not None:      # switched-off ring ports that are not 50 ohm (reflective switch)
            ctx.ld_card(4, int(tag), int(seg), int(seg), float(OFF_Z.real), float(OFF_Z.imag), 0.0)
            continue
        r = ref_load if i == ref_idx else term_r
        ctx.ld_card(0, int(tag), int(seg), int(seg), float(r), 0.0, 0.0)
    for tag, s0, s1, ohm in EXTRA_LOADS:
        ctx.ld_card(0, int(tag), int(s0), int(s1), float(ohm), 0.0, 0.0)
    ctx.ld_card(5, 0, 0, 0, 5.8e7, 0.0, 0.0)
    ctx.fr_card(0, 1, float(f_mhz), 0.0)
    dtag, dseg = ports[drive_idx]
    ctx.ex_card(0, int(dtag), int(dseg), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ctx.rp_card(0, 37, 72, 1, 0, 0, 0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0)
    rp = ctx.get_radiation_pattern(0)
    eth = np.asarray(rp.get_e_theta()).reshape(72, 37)
    eph = np.asarray(rp.get_e_phi()).reshape(72, 37)
    zin = complex(ctx.get_input_parameters(0).get_impedance()[0])
    return zin, eth.astype(np.complex128), eph.astype(np.complex128)


R._box_wires = _with_layout
R._solve = _solve

BASE = R.RING_SPECS["d120_16"]
LAYOUTS = {
    "f1": dict(kind="film", z_top=-88.9, ref_dz=45.0),
    "f2": dict(kind="film", z_top=-88.9, ref_dz=45.0, trunk_ohm=100.0),
    "r1": dict(kind="rod", d_base=45.0, z_top=-75.0, ref_dz=0.0),
    "r2": dict(kind="rod", d_base=45.0, z_top=-75.0, ref_dz=0.0, choke_ohm=150.0),
    "r3": dict(kind="rod", d_base=60.0, z_top=-90.0, ref_dz=0.0),
    # POST-HOC (plan section 6)
    "r4": dict(kind="rod", d_base=45.0, z_top=-90.0, ref_dz=0.0),
    "f3": dict(kind="film", z_top=-75.0, ref_dz=45.0),
    "f4": dict(kind="film", z_top=-105.0, ref_dz=45.0),
    "f5": dict(kind="film", z_top=-120.0, ref_dz=45.0),
    "f6": dict(kind="film", z_top=-88.9, ref_dz=30.0),
    "f7": dict(kind="film", z_top=-88.9, ref_dz=60.0),
    "f8": dict(kind="film", z_top=-88.9, ref_dz=45.0, trunk_rad=3.0),
    # POST-HOC 2 (plan section 7.3)
    "x1": dict(kind="film", z_top=-88.9, ref_dz=45.0, ref_cable="none"),
    "x3": dict(kind="bare", z_top=-88.9, ref_dz=45.0),
    "x4": dict(kind="film", z_top=-88.9, ref_dz=45.0, ref_cable="axis_gap"),
    "x5": dict(kind="film", z_top=-88.9, ref_dz=45.0, ref_cable="axis_ohm"),
    "x6": dict(kind="film", z_top=-88.9, ref_dz=45.0, ref_cable_r=25.0),
    # POST-HOC 3 (plan section 7.5): reference antenna back in the ring plane
    "k1": dict(kind="back", z_top=-88.9, ref_dz=0.0),
    "k2": dict(kind="cone", z_top=-88.9, ref_dz=0.0),                       # K2r: rod reference fed from below
    "k3": dict(kind="cone", z_top=-88.9, ref_dz=0.0, ref_cable="side"),     # K2f: film reference
    # POST-HOC 4 (plan section 7.6)
    "k1m": dict(kind="back", z_top=-88.9, ref_dz=0.0, mirror_x=True),
    "k1n": dict(kind="back", z_top=-75.0, ref_dz=0.0),
    "k1w": dict(kind="back", z_top=-105.0, ref_dz=0.0),
    "k1o": dict(kind="back", z_top=-88.9, ref_dz=0.0, trunk_out=75.0),
    "b0": dict(kind="bare", z_top=-88.9, ref_dz=0.0, box=(-67.5, 67.5, 78.75, 56.25)),
    "bk1": dict(kind="back", z_top=-88.9, ref_dz=0.0, box=(-67.5, 67.5, 78.75, 56.25)),
    "bk1s": dict(kind="back", z_top=-88.9, ref_dz=0.0, box=(-101.25, 33.75, 78.75, 56.25)),
    # POST-HOC 5 (plan section 7.7)
    "bk1w": dict(kind="back", z_top=-105.0, ref_dz=0.0, box=(-67.5, 67.5, 78.75, 56.25)),
    "bk1x": dict(kind="back", z_top=-120.0, ref_dz=0.0, box=(-67.5, 67.5, 78.75, 56.25)),
    "bkv": dict(kind="back", z_top=-88.9, ref_dz=0.0, box=(-56.25, 0.0, 78.75, 135.0)),
    # POST-HOC 6 (plan section 7.8): 5 GHz diagnostics, thick wires with segments >= 2.5 radii
    "k1a": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True),
    "k1b": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, no_trunk=True),
    "k1l": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, trunk_ohm=100.0),
    "f1a": dict(kind="film", z_top=-88.9, ref_dz=45.0, thin_fix=True),
    "k1oa": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, trunk_out=75.0),
    # plan section 7.9: slanted ribs (shorter cables) + bundle outside the ring
    "k2": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0, trunk_out=78.75),
    "k2n": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0, trunk_out=67.5),
    "k2i": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0),
    "bk2": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0, trunk_out=78.75,
                box=(-101.25, 33.75, 78.75, 56.25)),
    # plan section 7.10: perturbations (longer bundle, ribs at 30 deg; 60 deg ribs would cross the reference cable)
    "k2iw": dict(kind="back", z_top=-105.0, ref_dz=0.0, thin_fix=True, chevron=1.0),
    "k2i3": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=0.577),
    "k2nw": dict(kind="back", z_top=-105.0, ref_dz=0.0, thin_fix=True, chevron=1.0, trunk_out=67.5),
    "k2n3": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=0.577, trunk_out=67.5),
    "bk2i": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0,
                 box=(-67.5, 67.5, 78.75, 56.25)),
    "bk2iw": dict(kind="back", z_top=-105.0, ref_dz=0.0, thin_fix=True, chevron=1.0,
                  box=(-67.5, 67.5, 78.75, 56.25)),
    "bk2n": dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0, trunk_out=67.5,
                 box=(-78.75, 56.25, 78.75, 56.25)),
    # plan_20261126_risk3.md section 3: off ports of a reflective switch seen through the cable (|gamma| 0.78)
    "t1": dict(kind="back", z_top=-88.9, ref_dz=0.0, off_z=(404.5, 0.0)),        # reflection phase 0 deg
    "t2": dict(kind="back", z_top=-88.9, ref_dz=0.0, off_z=(12.2, 48.5)),        # reflection phase 90 deg
    "k5": dict(kind="l5", z_top=-88.9, ref_dz=0.0,
               cfg=dict(n_elem=16, diam_mm=0.0, kind="custom", elem_xyz_mm=[[float(x), float(y), 0.0] for x, y in P17[:16]])),
}
SPECS = {}
for _name in LAYOUTS:
    for _t in (0, 20, -20):
        SPECS[f"d120_16{_name}" + ("" if _t == 0 else f"_ty{_t:+d}")] = (_name, float(_t))
BANDS = {"2g4": 2437.0, "5g": 5220.0}


def build(keys, out_path, band="2g4"):
    global LAY, SEG, HL, OFF_Z, THIN_FIX
    f_mhz = BANDS[band]
    out = np.load(out_path, allow_pickle=True).item() if os.path.exists(out_path) else {}
    for key in keys:
        if (key, band) in out:
            continue
        name, t = SPECS[key]
        lay = LAYOUTS[name]
        HL = 0.235 * (R.C0 / (f_mhz * 1e6) * 1e3)
        SEG = 6.0 if f_mhz < 4000 else 3.0
        cfg = dict(BASE, box_gap_mm=abs(lay["z_top"]) - HL, ref_dz_mm=lay["ref_dz"])
        cfg.update(lay.get("cfg", {}))
        if t:
            cfg = dict(cfg, tilt_deg=t, tilt_axis="y")
        LAY = lay
        THIN_FIX = bool(lay.get("thin_fix"))
        OFF_Z = complex(*lay["off_z"]) if lay.get("off_z") else None
        res = R.simulate_ring(f_mhz=f_mhz, **cfg)
        LAY = None
        OFF_Z = None
        out[(key, band)] = res
        print(f"[NEC2/PyNEC] {key:22s} {band} | S11 ring mean={res['s11_db'].mean():6.1f} dB  ref={res['ref_s11_db']:6.1f} dB"
              f" | Zin ring={res['zin'].mean():.1f} ref={res['ref_zin']:.1f} | scale={res['scale']:.3f}", flush=True)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, out, allow_pickle=True)
        os.replace(tmp, out_path)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_rod.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(SPECS)
    band = sys.argv[3] if len(sys.argv) > 3 else "2g4"
    build(keys, out, band)
    print("ROD_NEC_DONE", keys, band, flush=True)
