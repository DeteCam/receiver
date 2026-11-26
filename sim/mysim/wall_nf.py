"""Near-field effect of the wall behind a wall-mounted D120 ring (NEC2, wall = Sommerfeld/PEC ground plane).

The ray tracer already adds the wave reflected by the wall behind the node, so the truth pattern here must not
contain it again.  Truth = free-space radiation of the element currents solved WITH the wall present (the wall
changes the currents through near-field coupling); reference = the same radiation formula applied to currents
solved WITHOUT the wall.  Both use the Thevenin convention: every port, the driven one included, carries a 50 ohm
load, so a pattern is proportional to the voltage at the receiver's 50 ohm input.

Head frame as in ring_pynec (local +x faces the room).  The wall is the plane x = -d_wall (d = 100 mm: node point =
reference antenna, 0.10 m from the wall, as in data_wall).  NEC needs the ground at z = 0, so the structure is
turned (x, y, z) -> (-z, y, x + d) before solving; currents are radiated back in the head frame.

Output (np.save dict, same keys as ring_pynec): (key, band) for
  d120_16_fv  free space            (estimator default: what a turntable calibration away from walls gives)
  d120_16_wc  in front of concrete  (ITU-R P.2040 a=5.24, c=0.0462, d=0.7822 - the ray tracer's wall material)
  d120_16_wp  in front of plasterboard (a=2.73, c=0.0085, d=0.9395)
  d120_16_wm  in front of metal (PEC: locker front, steel door)
"""
import os
import sys
import time
import numpy as np
from PyNEC import nec_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ring_pynec as R  # noqa: E402

C0 = 299792458.0
MM = 1e-3
WALLS = {
    "fv": None,
    "wc": ("diel", 5.24, 0.0462, 0.7822),
    "wp": ("diel", 2.73, 0.0085, 0.9395),
    "wm": ("pec",),
}


def ring_geometry(f_mhz, tilt_deg=0.0):
    """Same wires as ring_pynec.simulate_ring(n_elem=16, diam_mm=120, ref_dz_mm=0, box_gap_mm=60, fine box)."""
    lam_mm = C0 / (f_mhz * 1e6) * 1e3
    hl = 0.235 * lam_mm
    nseg, mid = 9, 5
    wires, ports = [], []
    tag = 1
    for nsg, p0, p1, rad in R._box_wires(z_top_mm=-(hl + 60.0), mesh_mm=(11.25 if f_mhz < 4000 else 5.625), sides=True):
        wires.append((tag, nsg, p0, p1, rad))
        tag += 1
    wires.append((9000, nseg, (0.0, 0.0, -hl), (0.0, 0.0, hl), 0.6))
    ports.append((9000, mid))
    xy = R._uca_xy(16, 120.0)
    for i, (x, y) in enumerate(xy):
        wires.append((9001 + i, nseg, (x, y, -hl), (x, y, hl), 0.6))
        ports.append((9001 + i, mid))
    xyz = np.hstack([xy, np.zeros((16, 1))])
    if tilt_deg:
        ct, st = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
        rx = lambda q: (ct * q[0] + st * q[2], q[1], -st * q[0] + ct * q[2])
        wires = [(t, n_, rx(p0), rx(p1), r) for t, n_, p0, p1, r in wires]
        xyz = np.array([rx(q) for q in xyz])
    return wires, ports, xyz


def segments(wires):
    """Segment centres (m), unit directions and lengths (m) in NEC numbering order."""
    c, d, L = [], [], []
    for _, nseg, p0, p1, _ in wires:
        p0 = np.asarray(p0, float) * MM
        p1 = np.asarray(p1, float) * MM
        v = p1 - p0
        ln = np.linalg.norm(v)
        for k in range(nseg):
            c.append(p0 + (k + 0.5) / nseg * v)
            d.append(v / ln)
            L.append(ln / nseg)
    return np.array(c), np.array(d), np.array(L)


def to_nec(p, d_mm):
    return (-p[2], p[1], p[0] + d_mm)


def solve_currents(wires, ports, f_mhz, wall, d_mm=100.0, check_rp=False):
    ctx = nec_context()
    geo = ctx.get_geometry()
    for tag, nseg, p0, p1, rad in wires:
        q0, q1 = to_nec(p0, d_mm), to_nec(p1, d_mm)
        geo.wire(int(tag), int(nseg), q0[0] * MM, q0[1] * MM, q0[2] * MM, q1[0] * MM, q1[1] * MM, q1[2] * MM,
                 rad * MM, 1.0, 1.0)
    ctx.geometry_complete(1 if wall else 0)
    ctx.set_extended_thin_wire_kernel(True)
    for tag, seg in ports:
        ctx.ld_card(0, int(tag), int(seg), int(seg), 50.0, 0.0, 0.0)
    ctx.ld_card(5, 0, 0, 0, 5.8e7, 0.0, 0.0)
    if wall is not None:
        if wall[0] == "pec":
            ctx.gn_card(1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            fg = f_mhz / 1000.0
            ctx.gn_card(2, 0, float(wall[1]), float(wall[2] * fg ** wall[3]), 0.0, 0.0, 0.0, 0.0)
    ctx.fr_card(0, 1, float(f_mhz), 0.0)
    cur = []
    rp = None
    for k, (tag, seg) in enumerate(ports):
        ctx.ex_card(0, int(tag), int(seg), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ctx.xq_card(0)
        cur.append(np.asarray(ctx.get_structure_currents(k).get_current(), dtype=np.complex128))
    if check_rp:
        # NEC's own far field for the last excitation, in the NEC frame (free space only)
        ctx.rp_card(0, 37, 72, 1, 0, 0, 0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0)
        r = ctx.get_radiation_pattern(0)
        rp = (np.asarray(r.get_e_theta()).reshape(72, 37), np.asarray(r.get_e_phi()).reshape(72, 37))
    return np.array(cur), rp


def radiate(cur, c, d, L, f_mhz, frame="head", d_mm=100.0):
    """Far field (E_theta, E_phi) on the 72 x 37 grid of each excitation's currents, up to the common factor."""
    k = 2.0 * np.pi * f_mhz * 1e6 / C0
    if frame == "nec":
        c = np.stack([-c[:, 2], c[:, 1], c[:, 0] + d_mm * MM], axis=1)
        d = np.stack([-d[:, 2], d[:, 1], d[:, 0]], axis=1)
    th = np.radians(np.arange(37) * 5.0)[None, :]
    ph = np.radians(np.arange(72) * 5.0)[:, None]
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    r = np.stack([st * cp, st * sp, ct * np.ones_like(ph)], axis=-1)            # (72, 37, 3)
    t = np.stack([ct * cp, ct * sp, -st * np.ones_like(ph)], axis=-1)
    p = np.stack([-sp * np.ones_like(th), cp * np.ones_like(th), np.zeros((72, 37))], axis=-1)
    ph_ = np.exp(1j * k * (r @ c.T))                                                # (72, 37, S)
    at = ph_ * ((t @ d.T) * L)
    ap = ph_ * ((p @ d.T) * L)
    eth = np.einsum("ijs,ns->nij", at, cur)
    eph = np.einsum("ijs,ns->nij", ap, cur)
    return eth, eph


def build(out_path, bands=(("2g4", 2437.0), ("5g", 5220.0)), tilts=(0.0,), walls=("fv", "wc", "wp", "wm"), d_mm=100.0):
    out = np.load(out_path, allow_pickle=True).item() if os.path.exists(out_path) else {}
    for band, f in bands:
        for tilt in tilts:
            wires, ports, xyz = ring_geometry(f, tilt)
            c, d, L = segments(wires)
            sfx = "" if not tilt else f"_ty{int(tilt):+d}"
            scale = None
            for wk in walls:
                key = f"d120_16_{wk}{sfx}" + ("" if d_mm == 100.0 else f"_d{int(d_mm)}")
                t0 = time.time()
                cur, rp = solve_currents(wires, ports, f, WALLS[wk], d_mm=d_mm, check_rp=(wk == "fv" and not tilt))
                assert cur.shape[1] == len(c), (cur.shape, len(c))
                eth, eph = radiate(cur, c, d, L, f)
                if rp is not None:
                    # check the radiation formula against NEC's own far field (last port, free space, NEC frame)
                    et_n, ep_n = radiate(cur[-1:], c, d, L, f, frame="nec", d_mm=d_mm)
                    m = np.abs(rp[0]) > 0.3 * np.abs(rp[0]).max()
                    ratio = rp[0][m] / et_n[0][m]
                    g = np.median(ratio.real) + 1j * np.median(ratio.imag)
                    err = np.abs(rp[0] - g * et_n[0]).max() / np.abs(rp[0]).max()
                    print(f"[check] {band} formula vs NEC: const={g:.3g}  max rel err={err:.3%}", flush=True)
                if scale is None:      # free-space reference sets the scale, walls keep their relative level
                    scale = float(np.mean(np.abs(eth[0, :, 18])) + 1e-30)
                out[(key, band)] = dict(
                    f_mhz=float(f), n_elem=16, diam_mm=120.0, ref_dz_mm=0.0, box_gap_mm=60.0, term_mode="thevenin50",
                    kind="dipole", invert=False, elem_xy_m=(xyz[:, :2] * MM), ref_dz_m=0.0, elem_xyz_m=(xyz * MM),
                    tilt_deg=float(tilt), eth=(eth[1:] / scale).astype(np.complex64),
                    eph=(eph[1:] / scale).astype(np.complex64), ref_eth=(eth[0] / scale).astype(np.complex64),
                    ref_eph=(eph[0] / scale).astype(np.complex64), scale=scale, wall=wk, d_wall_mm=float(d_mm))
                print(f"[wall-nf] {key:22s} {band:3s} segs={len(c)} {time.time() - t0:6.1f}s", flush=True)
                np.save(out_path, out, allow_pickle=True)
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "nec_wallnf.npy")
    bands = (("2g4", 2437.0), ("5g", 5220.0))
    if len(sys.argv) > 2:
        bands = tuple(b for b in bands if b[0] in sys.argv[2].split(","))
    dmm = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    build(path, bands=bands, d_mm=dmm)
