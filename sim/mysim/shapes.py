"""Receiver head shapes for the shape study (16 vertical dual-band dipoles unless noted, centre reference, the same
90 x 90 x 35 mm box 60 mm below the lowest element).  Folded size <= ~20 cm; "unfolded" heads are what a folding
head would open to while measuring.  Run: python3 mysim/shapes.py mysim/nec_shapes.npy"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402


def perimeter_points(verts, n):
    """n points equally spaced along a closed polygon (first point on the first vertex)."""
    V = np.asarray(verts, dtype=np.float64)
    seg = np.roll(V, -1, axis=0) - V
    L = np.linalg.norm(seg, axis=1)
    s = np.arange(n) * L.sum() / n
    cum = np.concatenate([[0.0], np.cumsum(L)])
    pts = []
    for d in s:
        i = int(np.searchsorted(cum, d, side="right") - 1)
        t = (d - cum[i]) / L[i]
        pts.append(V[i] + t * seg[i])
    return np.array(pts)


def regular(nv, circ_r, rot=0.0):
    a = np.radians(rot) + 2 * np.pi * np.arange(nv) / nv
    return np.stack([circ_r * np.cos(a), circ_r * np.sin(a)], axis=1)


def xyz(p2, z=0.0):
    p2 = np.asarray(p2, dtype=np.float64)
    return np.hstack([p2, np.full((len(p2), 1), z)])


def circle(n, d, rot=0.0, z=0.0):
    return xyz(regular(n, d / 2.0, rot), z)


GEOM = {
    # folded size about 12-17 cm
    "sq120_16": xyz(perimeter_points([(60, 60), (-60, 60), (-60, -60), (60, -60)], 16)),
    "tri150_16": xyz(perimeter_points(regular(3, 150 / np.sqrt(3), 90), 16)),
    "hex65_16": xyz(perimeter_points(regular(6, 65.0), 16)),
    "cross170_16": xyz([(r * c, r * s) for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1)) for r in (25, 45, 65, 85)]),
    "ura105_16": xyz([(x, y) for x in (-52.5, -17.5, 17.5, 52.5) for y in (-52.5, -17.5, 17.5, 52.5)]),
    "ring2_16": np.vstack([circle(8, 80.0, 22.5), circle(8, 160.0)]),
    "stack120_16": np.vstack([circle(8, 120.0, 0.0, -35.0), circle(8, 120.0, 22.5, 35.0)]),
    "slant120_16": circle(16, 120.0),
    # unfolded (folding head opened while measuring)
    "c240_16": circle(16, 240.0),
    "cross320_16": xyz([(r * c, r * s) for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1)) for r in (40, 80, 120, 160)]),
    "tri280_16": xyz(perimeter_points(regular(3, 280 / np.sqrt(3), 90), 16)),
}
SLANT = {"slant120_16": [45.0 if i % 2 == 0 else -45.0 for i in range(16)]}

SPECS = {k: dict(n_elem=len(v), diam_mm=0.0, ref_dz_mm=0.0, box_gap_mm=60.0, kind="custom", elem_xyz_mm=v.tolist(),
                 elem_slant_deg=SLANT.get(k)) for k, v in GEOM.items()}
# nodded heads (tilt about the x axis through the reference): D120 16-element ring
for t in (-20, 20):
    SPECS[f"c120_16_t{t:+d}"] = dict(n_elem=16, diam_mm=0.0, ref_dz_mm=0.0, box_gap_mm=60.0, kind="custom",
                                     elem_xyz_mm=circle(16, 120.0).tolist(), tilt_deg=float(t))

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_shapes.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SPECS)
    R.build({k: SPECS[k] for k in keys}, out)
