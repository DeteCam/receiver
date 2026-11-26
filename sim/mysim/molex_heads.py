"""Box distance and antenna-cable check for the Molex 146153 build (plan sim/plan_20261126_molex.md).  [simulation]
The film antennas come with a 100 mm cable that plugs straight into the switch board (SMA-U.FL adapters), so the
electronics sit closer to the ring than in the d120_16 model and 16 cables run down just inside the elements.
Same NEC model as ring_pynec.simulate_ring(kind="dipole"); the cables are added by wrapping ring_pynec._box_wires:
each cable is the 1.13 mm coax shield as one wire, starting 6 mm inside its element at mid height (left open there:
balanced antenna, no shield current is driven), running 5-9 mm inward and straight down to a node of the box-top grid
(bonded: SMA ground).  The whole head (box, ring, reference, cables) nods with tilt_deg like the other heads.
Run: python3 mysim/molex_heads.py mysim/nec_molex.npy [keys]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402

CABLE = None            # set per spec: dict(n_elem, ring_r_mm, r_start_mm, r_drop_mm, rad_mm)
_orig_box = R._box_wires


def _box_with_cables(z_top_mm, half_w=45.0, height=35.0, mesh_mm=11.25, sides=True, rad_mm=None):
    w = _orig_box(z_top_mm, half_w=half_w, height=height, mesh_mm=mesh_mm, sides=sides, rad_mm=rad_mm)
    if CABLE is None:
        return w
    n = max(2, int(round(2 * half_w / mesh_mm)))
    xs = np.linspace(-half_w, half_w, n + 1)          # the same grid nodes as the box top
    used = set()
    seg_mm = 6.0
    for k in range(CABLE["n_elem"]):
        a = 2.0 * np.pi * k / CABLE["n_elem"]
        p0 = (CABLE["r_start_mm"] * np.cos(a), CABLE["r_start_mm"] * np.sin(a), 0.0)
        tx, ty = CABLE["r_drop_mm"] * np.cos(a), CABLE["r_drop_mm"] * np.sin(a)
        gx = xs[int(np.argmin(np.abs(xs - tx)))]
        gy = xs[int(np.argmin(np.abs(xs - ty)))]
        if (gx, gy) in used:
            raise ValueError(f"two cables on one box node {(gx, gy)}")
        used.add((gx, gy))
        p1 = (float(gx), float(gy), 0.0)
        p2 = (float(gx), float(gy), float(z_top_mm))
        lh = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        lv = float(abs(z_top_mm))
        w.append((max(1, int(round(lh / seg_mm))), p0, p1, CABLE["rad_mm"]))
        w.append((max(2, int(round(lv / seg_mm))), p1, p2, CABLE["rad_mm"]))
    return w


R._box_wires = _box_with_cables

CAB = dict(n_elem=16, ring_r_mm=60.0, r_start_mm=54.0, r_drop_mm=48.0, rad_mm=0.565)
BASE = R.RING_SPECS["d120_16"]
SPECS = {}
for gap in (30, 15, 60):
    for cab in (False, True):
        if gap == 60 and not cab:
            continue                                   # G60 without cables = the existing d120_16 tables
        name = f"d120_16g{gap}" + ("c" if cab else "")
        for t in (0, 20, -20):
            key = name if t == 0 else f"{name}_ty{t:+d}"
            SPECS[key] = (dict(BASE, box_gap_mm=float(gap), tilt_deg=float(t), tilt_axis="y"), cab)


def build(keys, out_path, f_mhz=2437.0, band="2g4"):
    global CABLE
    out = np.load(out_path, allow_pickle=True).item() if os.path.exists(out_path) else {}
    for key in keys:
        if (key, band) in out:
            continue
        cfg, cab = SPECS[key]
        if not cfg["tilt_deg"]:
            cfg = {k: v for k, v in cfg.items() if k not in ("tilt_deg", "tilt_axis")}
        CABLE = CAB if cab else None
        res = R.simulate_ring(f_mhz=f_mhz, **cfg)
        CABLE = None
        out[(key, band)] = res
        print(f"[NEC2/PyNEC] {key:22s} {band} | S11 ring mean={res['s11_db'].mean():6.1f} dB  ref={res['ref_s11_db']:6.1f} dB"
              f" | Zin ring={res['zin'].mean():.1f}", flush=True)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, out, allow_pickle=True)
        os.replace(tmp, out_path)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_molex.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SPECS)
    build(keys, out)
    print("MOLEX_NEC_DONE", keys, flush=True)
