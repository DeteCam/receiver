"""NEC2 active-element patterns for the DeteCam switch ring, computed with PyNEC (nec2++ engine).

Same geometry, loads and output format as receiver/sim/em/ring_nec2.py (which shells out to the
`nec2c` binary).  Used for an independent re-computation, and extended with extra geometries
(ground-plane monopole ring) that the original script does not have.

Output: dict keyed by (pattern_key, band) with eth/eph (N,72,37), ref_eth/ref_eph (72,37), zin, s11.
"""
import os
import sys
import numpy as np
from PyNEC import nec_context

C0 = 299792458.0
MM = 1e-3


def _uca_xy(n_elem, diam_mm):
    a = 2.0 * np.pi * np.arange(n_elem) / n_elem
    return np.stack([0.5 * diam_mm * np.cos(a), 0.5 * diam_mm * np.sin(a)], axis=1)


def _box_wires(z_top_mm, half_w=45.0, height=35.0, mesh_mm=11.25, sides=True, rad_mm=None):
    """Connected wire-grid enclosure (default 90x90x35 mm).

    The original ring_nec2._make_metal_box draws 3 interior lines per face that cross each other and
    touch the rim in the *middle* of segments (7 segments over 90 mm), so in NEC they are not
    electrically joined, and nec2++ rejects the geometry.  Here every wire runs node-to-node on a
    common grid, so all crossings are real junctions.
      mesh_mm=22.5, sides=False, rad_mm=0.8  -> the cage the original script intended (coarse)
      mesh_mm<=0.1 lambda, sides=True        -> NEC wire-grid guideline for a solid box
    Returns list of (nseg, p0, p1, rad_mm).
    """
    n = max(2, int(round(2 * half_w / mesh_mm)))
    xs = np.linspace(-half_w, half_w, n + 1)
    mesh = 2 * half_w / n
    rad = rad_mm if rad_mm is not None else mesh / (2.0 * np.pi)
    z1, z2 = z_top_mm, z_top_mm - height
    w = []
    for z in (z1, z2):
        for a in xs:
            for i in range(n):
                w.append((1, (xs[i], a, z), (xs[i + 1], a, z), rad))
                w.append((1, (a, xs[i], z), (a, xs[i + 1], z), rad))
    if sides:
        nz = max(1, int(round(height / mesh)))
        zs = np.linspace(z2, z1, nz + 1)
        rim = [(x, -half_w) for x in xs] + [(x, half_w) for x in xs] + [(-half_w, y) for y in xs[1:-1]] + [(half_w, y) for y in xs[1:-1]]
        for (x, y) in rim:
            for k in range(nz):
                w.append((1, (x, y, zs[k]), (x, y, zs[k + 1]), rad))
        for z in zs[1:-1]:
            for i in range(n):
                w.append((1, (xs[i], -half_w, z), (xs[i + 1], -half_w, z), rad))
                w.append((1, (xs[i], half_w, z), (xs[i + 1], half_w, z), rad))
                w.append((1, (-half_w, xs[i], z), (-half_w, xs[i + 1], z), rad))
                w.append((1, (half_w, xs[i], z), (half_w, xs[i + 1], z), rad))
    else:
        for sx in (-1, 1):
            for sy in (-1, 1):
                w.append((2, (sx * half_w, sy * half_w, z2), (sx * half_w, sy * half_w, z1), rad))
    return w


def _plate_wires(z_mm, half_w, mesh_mm):
    """Square wire-grid plate (ground plane) centred on the z axis. Grid lines every mesh_mm."""
    n = int(round(2 * half_w / mesh_mm))
    xs = np.linspace(-half_w, half_w, n + 1)
    rad = mesh_mm / (2.0 * np.pi)  # equal-area rule for wire-grid surfaces
    w = []
    for a in xs:
        for i in range(n):
            w.append((1, (xs[i], a, z_mm), (xs[i + 1], a, z_mm), rad))
            w.append((1, (a, xs[i], z_mm), (a, xs[i + 1], z_mm), rad))
    return w, xs


def _vplate_wires(x_mm, w_mm, h_mm, mesh_mm, zc_mm=0.0):
    """Vertical wire-grid plate in the plane x = x_mm, |y| <= w/2, |z - zc| <= h/2, grid lines about every mesh_mm."""
    ny = max(1, int(round(w_mm / mesh_mm)))
    nz = max(1, int(round(h_mm / mesh_mm)))
    ys = np.linspace(-w_mm / 2, w_mm / 2, ny + 1)
    zs = np.linspace(zc_mm - h_mm / 2, zc_mm + h_mm / 2, nz + 1)
    rad = mesh_mm / (2.0 * np.pi)
    w = []
    for z in zs:
        for i in range(ny):
            w.append((1, (x_mm, ys[i], z), (x_mm, ys[i + 1], z), rad))
    for y in ys:
        for i in range(nz):
            w.append((1, (x_mm, y, zs[i]), (x_mm, y, zs[i + 1]), rad))
    return w


def _solve(wires, ports, drive_idx, f_mhz, term_r, ref_load=50.0, ref_idx=0):
    """wires: list of (tag, nseg, p0_mm, p1_mm, rad_mm). ports: list of (tag, seg). Returns (zin, eth, eph)."""
    ctx = nec_context()
    geo = ctx.get_geometry()
    for tag, nseg, p0, p1, rad in wires:
        geo.wire(int(tag), int(nseg), p0[0] * MM, p0[1] * MM, p0[2] * MM, p1[0] * MM, p1[1] * MM, p1[2] * MM,
                 rad * MM, 1.0, 1.0)
    ctx.geometry_complete(0)
    ctx.set_extended_thin_wire_kernel(True)
    for i, (tag, seg) in enumerate(ports):
        if i == drive_idx:
            continue
        r = ref_load if i == ref_idx else term_r
        ctx.ld_card(0, int(tag), int(seg), int(seg), float(r), 0.0, 0.0)
    ctx.ld_card(5, 0, 0, 0, 5.8e7, 0.0, 0.0)
    ctx.fr_card(0, 1, float(f_mhz), 0.0)
    dtag, dseg = ports[drive_idx]
    ctx.ex_card(0, int(dtag), int(dseg), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ctx.rp_card(0, 37, 72, 1, 0, 0, 0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0)
    rp = ctx.get_radiation_pattern(0)
    # PyNEC returns a flat array ordered phi-major (verified with an offset dipole): -> (72 phi, 37 theta)
    eth = np.asarray(rp.get_e_theta()).reshape(72, 37)
    eph = np.asarray(rp.get_e_phi()).reshape(72, 37)
    zin = complex(ctx.get_input_parameters(0).get_impedance()[0])
    return zin, eth.astype(np.complex128), eph.astype(np.complex128)


def simulate_ring(f_mhz, n_elem, diam_mm, ref_dz_mm, box_gap_mm, term_mode="absorptive", kind="dipole",
                  gp_half_mm=56.25, gp_mesh_mm=None, invert=False, box_model="fine", rot_deg=0.0,
                  elem_xyz_mm=None, elem_slant_deg=None, tilt_deg=0.0, plate=None, tilt_axis="x"):
    """kind="custom": dipoles at arbitrary centres elem_xyz_mm (N, 3) [mm], vertical unless elem_slant_deg gives a
    tilt of each dipole inside its tangential plane (dual-slant heads); same box and centre reference as "dipole".
    tilt_deg: the whole head nodded about the x axis (box, reference and elements)."""
    """kind="dipole": resonant half-wave dipoles floating box_gap_mm above the 90x90x35 box (original design).
    kind="gp_monopole": resonant quarter-wave monopoles standing on a square wire-grid ground plane
        (2*gp_half_mm wide) with the box directly underneath the plate (no gap needed).
        invert=True flips the whole structure (antennas point down, plate + box above them).
    """
    lam_mm = C0 / (f_mhz * 1e6) * 1e3
    term_r = {"absorptive": 50.0, "open": 1e6, "short": 0.01}[term_mode]
    xy = _uca_xy(n_elem, diam_mm)
    wires, ports = [], []
    tag = 1
    if kind == "dipole":
        hl = 0.235 * lam_mm
        nseg, mid = 9, 5
        if box_gap_mm is not None:
            if box_model == "coarse":   # connected version of the original 22.5 mm cage (top/bottom + 4 posts)
                bw = _box_wires(z_top_mm=-(hl + box_gap_mm), mesh_mm=22.5, sides=False, rad_mm=0.8)
            else:                       # <= 0.1 lambda mesh on all six faces
                bw = _box_wires(z_top_mm=-(hl + box_gap_mm), mesh_mm=(11.25 if f_mhz < 4000 else 5.625), sides=True)
            for nsg, p0, p1, rad in bw:
                wires.append((tag, nsg, p0, p1, rad))
                tag += 1
        wires.append((9000, nseg, (0.0, 0.0, ref_dz_mm - hl), (0.0, 0.0, ref_dz_mm + hl), 0.6))
        ports.append((9000, mid))
        for i, (x, y) in enumerate(xy):
            wires.append((9001 + i, nseg, (x, y, -hl), (x, y, hl), 0.6))
            ports.append((9001 + i, mid))
    elif kind == "custom":
        hl = 0.235 * lam_mm
        nseg, mid = 9, 5
        P = np.asarray(elem_xyz_mm, dtype=np.float64)
        n_elem = len(P)
        zmin = float(P[:, 2].min()) if len(P) else 0.0
        if box_gap_mm is not None:
            bw = _box_wires(z_top_mm=zmin - (hl + box_gap_mm), mesh_mm=(11.25 if f_mhz < 4000 else 5.625), sides=True)
            for nsg, p0, p1, rad in bw:
                wires.append((tag, nsg, p0, p1, rad))
                tag += 1
        wires.append((9000, nseg, (0.0, 0.0, ref_dz_mm - hl), (0.0, 0.0, ref_dz_mm + hl), 0.6))
        ports.append((9000, mid))
        sl = np.zeros(n_elem) if elem_slant_deg is None else np.radians(np.asarray(elem_slant_deg, dtype=np.float64))
        for i, (x, y, z) in enumerate(P):
            phi = np.arctan2(y, x) if (abs(x) + abs(y)) > 1e-9 else 0.0
            ax = np.array([np.sin(sl[i]) * -np.sin(phi), np.sin(sl[i]) * np.cos(phi), np.cos(sl[i])])
            c = np.array([x, y, z])
            wires.append((9001 + i, nseg, tuple(c - hl * ax), tuple(c + hl * ax), 0.6))
            ports.append((9001 + i, mid))
        xy = P[:, :2].copy()
        if plate is not None:
            # vertical wire-grid back plate in the plane x = plate["x_mm"] (wall-mounted heads; local +x faces the room)
            mesh = plate.get("mesh_mm") or (12.5 if f_mhz < 4000 else 6.67)
            for nsg, p0, p1, rad in _vplate_wires(plate["x_mm"], plate["w_mm"], plate["h_mm"], mesh, plate.get("zc_mm", 0.0)):
                wires.append((tag, nsg, p0, p1, rad))
                tag += 1
    elif kind == "gp_monopole":
        # Quarter-wave monopoles standing on a square wire-grid ground plane (112.5 x 112.5 mm, 6.25 mm mesh),
        # electronics box (87.5 x 87.5 x 35 mm) directly under the plate, bonded at its four corners.
        # Elements sit exactly on grid nodes: (+-43.75, 0), (0, +-43.75), (+-31.25, +-31.25) -> ring diameter ~88 mm.
        ql = 0.235 * lam_mm
        mesh = 6.25
        pw, xs = _plate_wires(0.0, gp_half_mm, mesh)
        for nsg, p0, p1, rad in pw:
            wires.append((tag, nsg, p0, p1, rad))
            tag += 1
        for nsg, p0, p1, rad in _box_wires(z_top_mm=-6.25, half_w=43.75, height=35.0, mesh_mm=12.5, sides=True):
            wires.append((tag, nsg, p0, p1, rad))
            tag += 1
        for sx in (-1, 1):
            for sy in (-1, 1):
                wires.append((tag, 1, (sx * 43.75, sy * 43.75, -6.25), (sx * 43.75, sy * 43.75, 0.0), 1.0))
                tag += 1
        r1, r2 = 43.75, 31.25
        xy = np.array([[r1, 0], [r2, r2], [0, r1], [-r2, r2], [-r1, 0], [-r2, -r2], [0, -r1], [r2, -r2]], dtype=np.float64)
        nseg = 5
        wires.append((9000, nseg, (0.0, 0.0, 0.0), (0.0, 0.0, ql), 0.6))
        ports.append((9000, 1))
        for i, (x, y) in enumerate(xy):
            wires.append((9001 + i, nseg, (x, y, 0.0), (x, y, ql), 0.6))
            ports.append((9001 + i, 1))
    else:
        raise ValueError(kind)

    if invert:
        wires = [(t, n_, (p0[0], p0[1], -p0[2]), (p1[0], p1[1], -p1[2]), r) for t, n_, p0, p1, r in wires]
    xyz = (np.asarray(elem_xyz_mm, dtype=np.float64) if (kind == "custom") else
           np.hstack([xy, np.zeros((len(xy), 1))]))
    if tilt_deg:
        ct, st = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
        if tilt_axis == "y":   # nod of a wall-mounted head about the wall-parallel horizontal axis (+ = looks down)
            rx = lambda q: (ct * q[0] + st * q[2], q[1], -st * q[0] + ct * q[2])
        else:
            rx = lambda q: (q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])
        wires = [(t, n_, rx(p0), rx(p1), r) for t, n_, p0, p1, r in wires]
        xyz = np.array([rx(q) for q in xyz])
        xy = xyz[:, :2].copy()
    if rot_deg:
        # whole head (box + ring + reference) turned about the vertical axis: a pan step of the receiver head
        ca, sa = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
        rz = lambda q: (ca * q[0] - sa * q[1], sa * q[0] + ca * q[1], q[2])
        wires = [(t, n_, rz(p0), rz(p1), r) for t, n_, p0, p1, r in wires]
        xy = np.stack([ca * xy[:, 0] - sa * xy[:, 1], sa * xy[:, 0] + ca * xy[:, 1]], axis=1)
        xyz = np.stack([xy[:, 0], xy[:, 1], xyz[:, 2]], axis=1)

    ref_zin, ref_eth, ref_eph = _solve(wires, ports, 0, f_mhz, term_r)
    zins, eths, ephs = [], [], []
    for i in range(n_elem):
        z, et, ep = _solve(wires, ports, 1 + i, f_mhz, term_r)
        zins.append(z)
        eths.append(et)
        ephs.append(ep)
    zins = np.asarray(zins)
    eths = np.asarray(eths)
    ephs = np.asarray(ephs)
    scale = float(np.mean(np.abs(ref_eth[:, 18])) + 1e-30)
    if kind == "gp_monopole":
        # horizon gain of a ground-plane monopole can be low; normalise by RMS over the sphere band actually used
        scale = float(np.sqrt(np.mean(np.abs(ref_eth[:, 6:31]) ** 2)) + 1e-30)
    s11 = 20.0 * np.log10(np.abs((zins - 50.0) / (zins + 50.0)) + 1e-12)
    ref_s11 = float(20.0 * np.log10(abs((ref_zin - 50.0) / (ref_zin + 50.0)) + 1e-12))
    return dict(
        f_mhz=float(f_mhz), n_elem=int(n_elem), diam_mm=float(diam_mm), ref_dz_mm=float(ref_dz_mm),
        box_gap_mm=float(box_gap_mm) if box_gap_mm is not None else float("nan"), term_mode=term_mode, kind=kind,
        invert=bool(invert), elem_xy_m=(xy * MM).astype(np.float64), ref_dz_m=float(ref_dz_mm * MM),
        elem_xyz_m=(np.asarray(xyz) * MM).astype(np.float64), tilt_deg=float(tilt_deg),
        eth=(eths / scale).astype(np.complex64), eph=(ephs / scale).astype(np.complex64),
        ref_eth=(ref_eth / scale).astype(np.complex64), ref_eph=(ref_eph / scale).astype(np.complex64),
        zin=zins, ref_zin=ref_zin, s11_db=s11, ref_s11_db=ref_s11, scale=scale,
    )


BASE_SPECS = {
    "spec002_dz65_abs":       dict(n_elem=8, diam_mm=100.0, ref_dz_mm=65.0, box_gap_mm=70.0, term_mode="absorptive"),
    "spec002plus_8_dz0_abs":  dict(n_elem=8, diam_mm=100.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive"),
    "spec002plus_8_dz0_open": dict(n_elem=8, diam_mm=100.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="open"),
    "spec002plus_d85_dz0":    dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive"),
    "spec002plus_d75_dz0":    dict(n_elem=8, diam_mm=75.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive"),
    "spec002pp_12_dz0_abs":   dict(n_elem=12, diam_mm=100.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive"),
    # same rings with the coarse (original-style, but connected) box cage: sensitivity to the box model
    "d100_cbox":              dict(n_elem=8, diam_mm=100.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive", box_model="coarse"),
    "d85_cbox":               dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="absorptive", box_model="coarse"),
}


def build(specs, out_path, bands=(("2g4", 2437.0), ("5g", 5220.0))):
    out = {}
    if os.path.exists(out_path):
        out = np.load(out_path, allow_pickle=True).item()
    for band, f in bands:
        for key, cfg in specs.items():
            if (key, band) in out:
                continue
            res = simulate_ring(f_mhz=f, **cfg)
            out[(key, band)] = res
            print(f"[NEC2/PyNEC] {key:26s} {band:3s} | S11 ring mean={res['s11_db'].mean():6.1f} dB  ref={res['ref_s11_db']:6.1f} dB"
                  f" | Zin ring={res['zin'].mean():.1f}", flush=True)
            np.save(out_path, out, allow_pickle=True)
    return out


SIZE_SPECS = {
    "d85_gap40": dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=40.0),
    "d85_gap20": dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=20.0),
    "d85_gap10": dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=10.0),
    "gp88_up":   dict(n_elem=8, diam_mm=88.0, ref_dz_mm=0.0, box_gap_mm=None, kind="gp_monopole", invert=False),
    "gp88_inv":  dict(n_elem=8, diam_mm=88.0, ref_dz_mm=0.0, box_gap_mm=None, kind="gp_monopole", invert=True),
}

# antenna-to-box gap 60 -> 80 / 100 mm (review finding: 80 mm fits the same height with the 83.1 mm LCD antenna)
GAP_SPECS = {
    "d85_gap80":  dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=80.0),
    "d85_gap100": dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=100.0),
}

# bigger rings (head up to ~20 x 20 cm), 16-element rings, and pan-rotated copies (22.5 deg = half the 8-element
# spacing) for a virtual 16-element ring made by turning the head between two sweeps
RING_SPECS = {
    "d85_r225":    dict(n_elem=8, diam_mm=85.0, ref_dz_mm=0.0, box_gap_mm=60.0, rot_deg=22.5),
    "d120_8":      dict(n_elem=8, diam_mm=120.0, ref_dz_mm=0.0, box_gap_mm=60.0),
    "d120_8_r225": dict(n_elem=8, diam_mm=120.0, ref_dz_mm=0.0, box_gap_mm=60.0, rot_deg=22.5),
    "d160_8":      dict(n_elem=8, diam_mm=160.0, ref_dz_mm=0.0, box_gap_mm=60.0),
    "d160_8_r225": dict(n_elem=8, diam_mm=160.0, ref_dz_mm=0.0, box_gap_mm=60.0, rot_deg=22.5),
    "d160_16":     dict(n_elem=16, diam_mm=160.0, ref_dz_mm=0.0, box_gap_mm=60.0),
    "d120_16":     dict(n_elem=16, diam_mm=120.0, ref_dz_mm=0.0, box_gap_mm=60.0),
}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "ring":
        build(RING_SPECS, sys.argv[1])
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[2] == "size":
        build(SIZE_SPECS, sys.argv[1])
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[2] == "gap":
        build(GAP_SPECS, sys.argv[1])
        sys.exit(0)
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy")
    build(BASE_SPECS, out_path)
