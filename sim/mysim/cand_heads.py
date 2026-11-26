"""Candidate heads for the realistic-room study (plan sim/plan_20261126_real.md, section 6).
Zig-zag rings: 16 vertical half-wave dipoles on a circle, alternately raised / lowered, centre reference at mid
height, the 90 x 90 x 35 mm box underneath (same model as the D120 ring).  A turn of the whole head about the vertical
axis needs no extra table (the pattern is rotated in the evaluation).
Run: python3 mysim/cand_heads.py mysim/nec_cand.npy [keys]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_pynec as R  # noqa: E402


def zigzag(diam_mm, dz_mm, n=16):
    a = 2.0 * np.pi * np.arange(n) / n
    z = np.where(np.arange(n) % 2 == 0, dz_mm, -dz_mm)
    return np.stack([0.5 * diam_mm * np.cos(a), 0.5 * diam_mm * np.sin(a), z], axis=1)


def zz_spec(diam_mm, dz_mm):
    return dict(n_elem=16, diam_mm=float(diam_mm), ref_dz_mm=0.0, box_gap_mm=60.0, kind="custom",
                elem_xyz_mm=zigzag(diam_mm, dz_mm).tolist())


SPECS = {
    "d190zz25": zz_spec(190.0, 25.0),
    "d190zz40": zz_spec(190.0, 40.0),
    "d170zz25": zz_spec(170.0, 25.0),
    "d190_16": dict(n_elem=16, diam_mm=190.0, ref_dz_mm=0.0, box_gap_mm=60.0),
    "d190_16_ty+20": dict(n_elem=16, diam_mm=190.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=20.0, tilt_axis="y"),
    "d190_16_ty-20": dict(n_elem=16, diam_mm=190.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=-20.0, tilt_axis="y"),
    # POST-HOC additions (after the first set-A results): flat rings between D120 and D190 with the nod
    "d160_16_ty+20": dict(n_elem=16, diam_mm=160.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=20.0, tilt_axis="y"),
    "d160_16_ty-20": dict(n_elem=16, diam_mm=160.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=-20.0, tilt_axis="y"),
    "d150_16": dict(n_elem=16, diam_mm=150.0, ref_dz_mm=0.0, box_gap_mm=60.0),
    "d150_16_ty+20": dict(n_elem=16, diam_mm=150.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=20.0, tilt_axis="y"),
    "d150_16_ty-20": dict(n_elem=16, diam_mm=150.0, ref_dz_mm=0.0, box_gap_mm=60.0, tilt_deg=-20.0, tilt_axis="y"),
    "d190zz25_ty+20": dict(zz_spec(190.0, 25.0), tilt_deg=20.0, tilt_axis="y"),
    "d190zz25_ty-20": dict(zz_spec(190.0, 25.0), tilt_deg=-20.0, tilt_axis="y"),
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_cand.npy")
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SPECS)
    bands = (("2g4", 2437.0), ("5g", 5220.0))
    if len(sys.argv) > 3:
        bands = tuple(b for b in bands if b[0] in sys.argv[3].split(","))
    R.build({k: SPECS[k] for k in keys}, out, bands=bands)
    print("CAND_DONE", keys, flush=True)
