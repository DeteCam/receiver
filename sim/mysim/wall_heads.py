"""Wall-mounted receiver heads (plan sim/plan_20261126_wall.md, section 4).
Local frame: +x = the room-facing normal of the wall, +z = up, the reference antenna at the origin.
16 switched vertical half-wave dipoles + the centre reference; back plates are vertical wire grids.
Run: python3 mysim/wall_heads.py mysim/nec_wall.npy [keys]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402


def semi(n=16, r=85.0, z=None):
    a = np.radians(-90.0 + 180.0 * np.arange(n) / (n - 1))
    zz = np.zeros(n) if z is None else np.asarray(z, dtype=np.float64)
    return np.stack([r * np.cos(a), r * np.sin(a), zz], axis=1)


def planar(cols=8, pitch=22.0, rows=(60.0, -60.0)):
    ys = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    return np.array([[0.0, y, z] for z in rows for y in ys])


GEOM = {
    "semi16": (semi(), dict(x_mm=-40.0, w_mm=200.0, h_mm=180.0)),
    "semi16s": (semi(z=[45.0 if k % 2 == 0 else -45.0 for k in range(16)]), dict(x_mm=-40.0, w_mm=200.0, h_mm=180.0)),
    "plan2x8": (planar(), dict(x_mm=-22.0, w_mm=200.0, h_mm=200.0)),
    # same semicircle with a larger back plate (better front-to-back at 2.4 GHz, 30 x 26 cm)
    "semi16L": (semi(), dict(x_mm=-40.0, w_mm=300.0, h_mm=260.0)),
}
SPECS = {k: dict(n_elem=len(p), diam_mm=0.0, ref_dz_mm=0.0, box_gap_mm=None, kind="custom", elem_xyz_mm=p.tolist(),
                 plate=pl) for k, (p, pl) in GEOM.items()}
# nodded versions (about the wall-parallel horizontal axis; + = looks down)
for base in ("semi16", "semi16s", "plan2x8"):
    for t in (20, -20):
        SPECS[f"{base}_t{t:+d}"] = dict(SPECS[base], tilt_deg=float(t), tilt_axis="y")

# the present D120 16-element ring (box underneath) nodded about the wall-parallel axis, for wall mounting
for t in (20, -20):
    SPECS[f"d120_16_ty{t:+d}"] = dict(R.RING_SPECS["d120_16"], tilt_deg=float(t), tilt_axis="y")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_wall.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SPECS)
    R.build({k: SPECS[k] for k in keys}, out)
