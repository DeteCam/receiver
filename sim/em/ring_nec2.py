"""Full-wave NEC2 (nec2c) active element pattern generator for DeteCam switch-ring arrays.
Computes 3D complex far-field patterns (eth, eph) on a 5-deg grid (72 azimuth x 37 polar)
including mutual coupling, unselected-port load impedance (50 Ohm absorptive vs open reflective),
and a 90x90x35 mm metallic host/switch enclosure below the array.

Usage:
  python3 sim/em/ring_nec2.py
Outputs:
  sim/em/data/ring_nec2_patterns.npy (ignored by git; reproducible from this script)
"""
import os
import re
import subprocess
import tempfile
import numpy as np

C0 = 299792458.0


def _make_metal_box(z_top_mm, half_w=45.0, height=35.0, start_tag=1):
    """Floating wire-grid metallic enclosure (90x90x35 mm)."""
    gw = []
    tag = start_tag
    z1 = z_top_mm
    z2 = z_top_mm - height
    pts = [(-half_w, -half_w), (half_w, -half_w), (half_w, half_w), (-half_w, half_w)]
    for z in (z1, z2):
        for k in range(4):
            (x0, y0), (x1, y1) = pts[k], pts[(k + 1) % 4]
            gw.append(f"GW {tag} 7 {x0:.2f} {y0:.2f} {z:.2f} {x1:.2f} {y1:.2f} {z:.2f} 0.8")
            tag += 1
        for t in (-22.5, 0.0, 22.5):
            gw.append(f"GW {tag} 7 {-half_w:.2f} {t:.2f} {z:.2f} {half_w:.2f} {t:.2f} {z:.2f} 0.8")
            tag += 1
            gw.append(f"GW {tag} 7 {t:.2f} {-half_w:.2f} {z:.2f} {t:.2f} {half_w:.2f} {z:.2f} 0.8")
            tag += 1
    for (x0, y0) in pts:
        gw.append(f"GW {tag} 4 {x0:.2f} {y0:.2f} {z1:.2f} {x0:.2f} {y0:.2f} {z2:.2f} 0.8")
        tag += 1
    return gw


def _uca_xy(n_elem, diam_mm):
    a = 2.0 * np.pi * np.arange(n_elem) / n_elem
    return np.stack([0.5 * diam_mm * np.cos(a), 0.5 * diam_mm * np.sin(a)], axis=1)


def simulate_ring_nec2(f_mhz, n_elem, diam_mm, ref_dz_mm, box_gap_mm, term_mode="absorptive", workdir="/tmp"):
    """Run nec2c for reference dipole (tag 9000) and all N ring dipoles (tags 9001..9000+N).
    Returns dict with keys: eth (N,72,37), eph (N,72,37), ref_eth (72,37), ref_eph (72,37),
    zin (N,), ref_zin, elem_xy_m (N,2), ref_dz_m, s11_db (N,), ref_s11_db.
    """
    lam_mm = C0 / (f_mhz * 1e6) * 1e3
    hl = 0.235 * lam_mm  # resonant half-wave dipole per band (represents true dual-band matched dipole)
    nseg = 9
    mid = 5
    xy = _uca_xy(n_elem, diam_mm)
    gw = _make_metal_box(z_top_mm=-(hl + box_gap_mm)) if box_gap_mm is not None else []
    for i, (x, y) in enumerate(xy):
        gw.append(f"GW {9001+i} {nseg} {x:.3f} {y:.3f} {-hl:.3f} {x:.3f} {y:.3f} {hl:.3f} 0.6")
    gw.append(f"GW 9000 {nseg} 0.000 0.000 {ref_dz_mm - hl:.3f} 0.000 0.000 {ref_dz_mm + hl:.3f} 0.6")

    term_r = {"absorptive": 50.0, "open": 1e6, "short": 0.01}[term_mode]
    all_tags = [9000] + [9001 + i for i in range(n_elem)]

    def _run_port(drive_tag):
        lines = [
            "CM DeteCam V1 Switch Ring",
            "CE",
            *gw,
            "GS 0 0 0.001",
            "GE 0",
            "EK",
            f"FR 0 1 0 0 {f_mhz:.2f} 0",
        ]
        for t in all_tags:
            if t == drive_tag:
                continue
            rload = 50.0 if t == 9000 else term_r
            lines.append(f"LD 0 {t} {mid} {mid} {rload} 0 0")
        lines += [
            "LD 5 0 0 0 5.8e7",
            f"EX 0 {drive_tag} {mid} 0 1.0 0.0",
            "RP 0 37 72 1000 0 0 5 5",
            "EN",
        ]
        fin = os.path.join(workdir, f"ring_{drive_tag}.nec")
        fout = os.path.join(workdir, f"ring_{drive_tag}.out")
        with open(fin, "w") as f:
            f.write("\n".join(lines) + "\n")
        subprocess.run(["nec2c", "-i", fin, "-o", fout], check=True, stdout=subprocess.DEVNULL)
        txt = open(fout).read()
        zin = np.nan + 1j * np.nan
        for ln in txt.split("ANTENNA INPUT PARAMETERS")[1].split("\n")[1:8]:
            p = ln.split()
            if len(p) >= 11 and p[0] == str(drive_tag):
                zin = complex(float(p[6]), float(p[7]))
                break
        m2 = re.search(r"VOLTS/M\s+DEGREES\s+VOLTS/M\s+DEGREES\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S)
        rows = []
        for r in m2.group(1).split("\n"):
            if not r.strip():
                continue
            q = r.replace("LINEAR", " ").replace("RIGHT", " ").replace("LEFT", " ").split()
            rows.append([float(q[0]), float(q[1])] + [float(v) for v in q[-4:]])
        rows = np.asarray(rows, dtype=np.float64)
        # NEC2 RP 0 37 72 loops theta (0..180, 37) fastest inside phi (0..355, 72)
        eth = (rows[:, 2] * np.exp(1j * np.radians(rows[:, 3]))).reshape(72, 37)
        eph = (rows[:, 4] * np.exp(1j * np.radians(rows[:, 5]))).reshape(72, 37)
        return zin, eth, eph

    ref_zin, ref_eth, ref_eph = _run_port(9000)
    # Exploit N-fold rotational symmetry? No — explicitly solve every port to verify box-corner asymmetry!
    zins, eths, ephs = [], [], []
    for i in range(n_elem):
        z, et, ep = _run_port(9001 + i)
        zins.append(z)
        eths.append(et)
        ephs.append(ep)
    zins = np.asarray(zins, dtype=np.complex128)
    eths = np.asarray(eths, dtype=np.complex64)
    ephs = np.asarray(ephs, dtype=np.complex64)
    # Normalize scale by mean horizon magnitude of reference dipole so gains are O(1)
    scale = float(np.mean(np.abs(ref_eth[:, 18])) + 1e-30)
    s11_db = 20.0 * np.log10(np.abs((zins - 50.0) / (zins + 50.0)) + 1e-12)
    ref_s11_db = float(20.0 * np.log10(abs((ref_zin - 50.0) / (ref_zin + 50.0)) + 1e-12))
    return dict(
        f_mhz=float(f_mhz),
        n_elem=int(n_elem),
        diam_mm=float(diam_mm),
        ref_dz_mm=float(ref_dz_mm),
        box_gap_mm=float(box_gap_mm),
        term_mode=term_mode,
        elem_xy_m=(xy * 1e-3).astype(np.float64),
        ref_dz_m=float(ref_dz_mm * 1e-3),
        eth=(eths / scale).astype(np.complex64),
        eph=(ephs / scale).astype(np.complex64),
        ref_eth=(ref_eth / scale).astype(np.complex64),
        ref_eph=(ref_eph / scale).astype(np.complex64),
        zin=zins,
        ref_zin=ref_zin,
        s11_db=s11_db,
        ref_s11_db=ref_s11_db,
    )


def build_all(out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ring_nec2_patterns.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    specs = {
        "spec002_dz65_abs":      dict(n_elem=8,  diam_mm=100.0, ref_dz_mm=65.0, box_gap_mm=70.0, term_mode="absorptive"),
        "spec002_dz65_open":     dict(n_elem=8,  diam_mm=100.0, ref_dz_mm=65.0, box_gap_mm=70.0, term_mode="open"),
        "spec002plus_8_dz0_abs": dict(n_elem=8,  diam_mm=100.0, ref_dz_mm=0.0,  box_gap_mm=60.0, term_mode="absorptive"),
        "spec002plus_8_dz0_open":dict(n_elem=8,  diam_mm=100.0, ref_dz_mm=0.0,  box_gap_mm=60.0, term_mode="open"),
        "spec002plus_d85_dz0":   dict(n_elem=8,  diam_mm=85.0,  ref_dz_mm=0.0,  box_gap_mm=60.0, term_mode="absorptive"),
        "spec002pp_12_dz0_abs":  dict(n_elem=12, diam_mm=100.0, ref_dz_mm=0.0,  box_gap_mm=60.0, term_mode="absorptive"),
    }
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for band, f_mhz in (("2g4", 2437.0), ("5g", 5220.0)):
            for key, cfg in specs.items():
                res = simulate_ring_nec2(f_mhz=f_mhz, workdir=td, **cfg)
                out[(key, band)] = res
                print(
                    f"[NEC2] {key:24s} {band:3s} ({f_mhz:.0f}MHz) | "
                    f"S11 mean={res['s11_db'].mean():5.1f}dB ref={res['ref_s11_db']:5.1f}dB",
                    flush=True,
                )
    np.save(out_path, out, allow_pickle=True)
    print(f"Saved NEC2 patterns to {out_path}")
    return out_path


if __name__ == "__main__":
    build_all()
