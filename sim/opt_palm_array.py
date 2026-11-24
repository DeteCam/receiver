"""
DeteCam 5.0 — Palm-Sized (~124 x 124 mm) Wall-Mounted Dual-Band AoA Array Optimizer
Rigorous electromagnetic comparison of compact palm-sized 2.4 / 5.2 / 5.8 GHz arrays
inside a 112 x 112 mm active aperture (124 x 124 x 30 mm palm puck):

  1. Palm-8+1 (Irregular 8-element perimeter + center ref, 9 elements total)
  2. Palm-12-EdgeRing (12 perimeter-edge elements + ref, supports SMD edge-mount chip antennas)
  3. Palm-12-DualRing (8 outer + 4 inner non-uniform ring + ref, PIFA / compact patch)
  4. Palm-12-MolexStag (12 staggered vertical Molex 146153 34.9x9 mm + ref, zero RF retuning)
  5. Baseline refs: v4.0 8-el 420mm L-arm, Fold12-Mini 300mm X-frame

Includes:
  - Peak Sidelobe Level (PSL) across full frontal hemisphere (theta <= 65 deg) at 2.45, 5.2, 5.8 GHz
  - Exact Cramér-Rao Lower Bound (CRLB) at SNR = 15 dB, 20 snapshots
  - Realistic sub-lambda/2 Mutual Coupling matrix C = (I + Z_mc / Z0)^(-1) using induced-EMF / sinc-Hankel model
  - Monte Carlo 2D Capon AoA WITH mutual coupling (Raw vs Calibrated C_inv) + 15% random 180 deg phase flips
    + 2 indoor multipath reflections (3D indoor wall-mounted scenario)
"""
import json
import os
import numpy as np

C0 = 299792458.0
FREQS = {"2.45G": 2.45e9, "5.2G": 5.2e9, "5.8G": 5.8e9}


def make_hemi_grid(step_deg=4.0, max_el_deg=65.0):
    """Direction cosines (u, v) on frontal hemisphere of wall-mounted puck (u^2+v^2 <= sin^2(65))."""
    vals = np.linspace(-np.sin(np.radians(max_el_deg)), np.sin(np.radians(max_el_deg)), int(2 * max_el_deg / step_deg) + 1)
    uu, vv = np.meshgrid(vals, vals)
    mask = (uu ** 2 + vv ** 2) <= np.sin(np.radians(max_el_deg)) ** 2
    return np.stack([uu[mask], vv[mask]], axis=1)


def compute_psl(xy_mm, freq_hz, uv_grid):
    """Worst-case sidelobe peak outside mainlobe (for boresight + 30 deg steered beams)."""
    lam_mm = (C0 / freq_hz) * 1e3
    k = 2.0 * np.pi / lam_mm
    N = len(xy_mm)
    worst = 0.0
    for (u0, v0) in ((0.0, 0.0), (0.5, 0.0), (0.35, 0.35)):
        du = uv_grid[:, 0] - u0
        dv = uv_grid[:, 1] - v0
        dist_uv = np.hypot(du, dv)
        # Mainlobe exclusion radius ~ 1.2 * lam / D_eff
        span = max(np.ptp(xy_mm[:, 0]), np.ptp(xy_mm[:, 1]), 50.0)
        ml_rad = min(0.65, 1.35 * lam_mm / span)
        phase = k * (du[:, None] * xy_mm[None, :, 0] + dv[:, None] * xy_mm[None, :, 1])
        af = np.abs(np.sum(np.exp(1j * phase), axis=1)) / N
        sl = af[dist_uv > ml_rad]
        if len(sl) > 0:
            worst = max(worst, float(np.max(sl)))
    return worst


def crlb_deg(xy_mm, freq_hz, snr_db=15.0, snapshots=20):
    """2D boresight AoA Cramér-Rao Lower Bound (deg RMS)."""
    lam_mm = (C0 / freq_hz) * 1e3
    k = 2.0 * np.pi / lam_mm
    snr = 10.0 ** (snr_db / 10.0)
    xy_c = xy_mm - xy_mm.mean(axis=0)
    sx2 = float(np.sum(xy_c[:, 0] ** 2))
    sy2 = float(np.sum(xy_c[:, 1] ** 2))
    # FIM for u, v at boresight: 2 * L * SNR * k^2 * sum(x_i^2)
    var_u = 1.0 / max(2.0 * snapshots * snr * (k ** 2) * sx2, 1e-12)
    var_v = 1.0 / max(2.0 * snapshots * snr * (k ** 2) * sy2, 1e-12)
    return float(np.degrees(np.sqrt(var_u + var_v)))


def mutual_coupling_matrix(xy_mm, freq_hz, strength=0.38):
    """
    Physically grounded mutual coupling matrix C = (I + Z_mc / Z0)^(-1) for planar dipole/PIFA array.
    Uses free-space spherical-wave decay ~ exp(-j k d) / (k d) with realistic coupling strength
    giving |S21| ~ -12 dB at d = 0.25*lam and |S21| ~ -19 dB at d = 0.50*lam.
    """
    lam_mm = (C0 / freq_hz) * 1e3
    k = 2.0 * np.pi / lam_mm
    N = len(xy_mm)
    Z_norm = np.eye(N, dtype=np.complex128)
    s21_max_db = -99.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = float(np.hypot(xy_mm[i, 0] - xy_mm[j, 0], xy_mm[i, 1] - xy_mm[j, 1]))
            kd = max(k * d, 0.4)
            # Induced-EMF style complex mutual impedance ratio Z_ij / Z0
            z_ij = strength * (np.exp(-1j * kd) / kd) * (1.0 - 1j / (kd + 0.5))
            Z_norm[i, j] = z_ij
            s21_db = 20.0 * np.log10(min(abs(z_ij), 0.99) + 1e-9)
            if s21_db > s21_max_db:
                s21_max_db = s21_db
    C = np.linalg.inv(Z_norm)
    return C, float(s21_max_db)


def min_spacing_mm(xy_mm):
    N = len(xy_mm)
    dmin = 1e9
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.hypot(xy_mm[i, 0] - xy_mm[j, 0], xy_mm[i, 1] - xy_mm[j, 1]))
            if d < dmin:
                dmin = d
    return dmin


def check_molex_no_overlap(xy_mm, w=10.5, h=36.0):
    """Verify 34.9x9 mm vertical Molex 146153 strips have >= 1.5mm clearance."""
    N = len(xy_mm)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(xy_mm[i, 0] - xy_mm[j, 0]) < w and abs(xy_mm[i, 1] - xy_mm[j, 1]) < h:
                return False
    return True


def capon_mc(xy_mm, freq_hz, n_trials=200, snr_db=15.0, flip_prob=0.15,
             calibrate_mc=True, cal_err_db=-26.0, seed=42):
    """
    Monte Carlo 2D Capon AoA evaluation under:
      - Realistic mutual coupling matrix C
      - 15% random 180-deg AX210 CSI phase flips (mitigated via RCLPA median sign consensus)
      - 2 indoor specular multipath rays (-6 dB, -9 dB)
      - Optional coupling matrix C_inv calibration (with residual calibration error -26 dB)
    """
    rng = np.random.default_rng(seed)
    lam_mm = (C0 / freq_hz) * 1e3
    k = 2.0 * np.pi / lam_mm
    N = len(xy_mm)
    C_true, s21_max = mutual_coupling_matrix(xy_mm, freq_hz)
    # Calibrated estimate C_hat = C_true + small turntable measurement noise
    cal_sig = 10.0 ** (cal_err_db / 20.0)
    C_hat = C_true + cal_sig * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0)
    C_inv = np.linalg.inv(C_hat)

    # Coarse + fine UV grid for fast accurate Capon search
    u_vals = np.linspace(-0.85, 0.85, 55)
    uu, vv = np.meshgrid(u_vals, u_vals)
    mask = (uu ** 2 + vv ** 2) <= 0.85 ** 2
    uv = np.stack([uu[mask], vv[mask]], axis=1)
    A_grid = np.exp(1j * k * (uv[:, 0:1] * xy_mm[None, :, 0] + uv[:, 1:2] * xy_mm[None, :, 1]))  # (G, N)

    errs = []
    n_snap = 24
    noise_sig = 10.0 ** (-snr_db / 20.0)

    for _ in range(n_trials):
        # True LoS direction inside +-50 deg FoV
        az = rng.uniform(-50.0, 50.0)
        el = rng.uniform(-35.0, 35.0)
        u_t = np.sin(np.radians(az)) * np.cos(np.radians(el))
        v_t = np.sin(np.radians(el))
        w_t = np.sqrt(max(1.0 - u_t ** 2 - v_t ** 2, 1e-6))
        a_los = np.exp(1j * k * (u_t * xy_mm[:, 0] + v_t * xy_mm[:, 1]))

        # 2 coherent-ish indoor multipath reflections (walls/floor)
        a_mp = np.zeros(N, dtype=np.complex128)
        for amp in (0.45, 0.30):
            az_m = rng.uniform(-65.0, 65.0)
            el_m = rng.uniform(-45.0, 45.0)
            um = np.sin(np.radians(az_m)) * np.cos(np.radians(el_m))
            vm = np.sin(np.radians(el_m))
            ph = rng.uniform(0, 2 * np.pi)
            a_mp += amp * np.exp(1j * ph) * np.exp(1j * k * (um * xy_mm[:, 0] + vm * xy_mm[:, 1]))

        # Generate snapshots across 24 packets / subcarriers (frequency averaging decorrelates multipath)
        Y = np.zeros((N, n_snap), dtype=np.complex128)
        for s in range(n_snap):
            # Multipath phase varies across subcarriers/packets relative to LoS
            mp_rot = np.exp(1j * rng.uniform(0, 2 * np.pi))
            h_clean = C_true @ (a_los + mp_rot * a_mp)
            noise = noise_sig * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
            y = h_clean + noise
            # 15% random 180-deg phase flip on AX210 switched port
            flips = np.where(rng.random(N) < flip_prob, -1.0, 1.0)
            Y[:, s] = y * flips

        # RCLPA: per-element majority-cluster sign consensus
        # With 85% unflipped and 15% 180-deg flipped snapshots, the complex sample mean
        # has expected value (1 - 2*p_flip) * h_clean = +0.70 * h_clean, preserving TRUE sign!
        Y_corr = np.zeros_like(Y)
        for i in range(N):
            # Iterative robust mean (2 steps) removes the 15% flipped outliers while preserving true phase
            mu = np.mean(Y[i, :])
            for _ in range(2):
                signs = np.where(np.real(Y[i, :] * np.conj(mu)) >= 0, 1.0, -1.0)
                mu = np.mean(Y[i, :] * signs)
            Y_corr[i, :] = Y[i, :] * signs

        # Apply mutual coupling compensation C_inv if enabled
        if calibrate_mc:
            Y_proc = C_inv @ Y_corr
        else:
            Y_proc = Y_corr

        # Covariance + diagonal loading Capon (MVDR)
        R = (Y_proc @ Y_proc.conj().T) / n_snap
        R += (0.04 * np.trace(R).real / N) * np.eye(N)
        R_inv = np.linalg.inv(R)
        # Spectrum: 1 / Re(a^H R^-1 a)
        quad = np.einsum("gi,ij,gj->g", A_grid.conj(), R_inv, A_grid).real
        best_idx = int(np.argmin(quad))
        u_hat, v_hat = uv[best_idx]
        # Fine local refinement around peak
        du_f = np.linspace(u_hat - 0.04, u_hat + 0.04, 17)
        dv_f = np.linspace(v_hat - 0.04, v_hat + 0.04, 17)
        uuf, vvf = np.meshgrid(du_f, dv_f)
        mf = (uuf ** 2 + vvf ** 2) <= 0.96
        uvf = np.stack([uuf[mf], vvf[mf]], axis=1)
        Af = np.exp(1j * k * (uvf[:, 0:1] * xy_mm[None, :, 0] + uvf[:, 1:2] * xy_mm[None, :, 1]))
        qf = np.einsum("gi,ij,gj->g", Af.conj(), R_inv, Af).real
        u_hat, v_hat = uvf[int(np.argmin(qf))]
        w_hat = np.sqrt(max(1.0 - u_hat ** 2 - v_hat ** 2, 1e-6))

        cos_err = np.clip(u_t * u_hat + v_t * v_hat + w_t * w_hat, -1.0, 1.0)
        errs.append(float(np.degrees(np.arccos(cos_err))))

    errs = np.array(errs)
    return dict(
        med_deg=round(float(np.median(errs)), 2),
        p90_deg=round(float(np.percentile(errs, 90)), 2),
        mean_deg=round(float(np.mean(errs)), 2),
        s21_max_db=round(s21_max, 1),
    )


def optimize_palm_12_dualring(uv_grid):
    """Optimize 12 elements inside 110 x 110 mm (8 outer + 4 inner) for minimum dual-band PSL & CRLB."""
    rng = np.random.default_rng(202605)
    best_score = 1e9
    best_xy = None
    # Start from structured golden/staggered dual-ring & random-restart refine
    for trial in range(400):
        r_out = rng.uniform(49.0, 54.5, 8)
        ang_out = np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(-0.14, 0.14, 8)
        r_in = rng.uniform(19.0, 31.0, 4)
        ang_in = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4.0 + rng.uniform(-0.22, 0.22, 4)
        xy = np.vstack([
            np.column_stack([r_out * np.cos(ang_out), r_out * np.sin(ang_out)]),
            np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in)]),
        ])
        xy = np.clip(xy, -53.0, 53.0)
        if min_spacing_mm(xy) < 25.0:
            continue
        p24 = compute_psl(xy, 2.45e9, uv_grid)
        p52 = compute_psl(xy, 5.2e9, uv_grid)
        p58 = compute_psl(xy, 5.8e9, uv_grid)
        c24 = crlb_deg(xy, 2.45e9)
        score = max(p52, p58) * 1.4 + p24 * 0.6 + 0.15 * c24
        if score < best_score:
            best_score = score
            best_xy = xy.copy()
    return np.round(best_xy, 1)


def optimize_palm_12_edgering(uv_grid):
    """Optimize 12 elements strictly along the outer edge of a 112x112 mm PCB (for SMD edge-mount chip antennas!)."""
    rng = np.random.default_rng(202606)
    best_score = 1e9
    best_xy = None
    E = 53.0  # 106 x 106 mm element centers inside 114 x 114 mm PCB
    base_3 = np.array([-35.0, 0.0, 35.0])
    for _ in range(800):
        t_top = np.clip(base_3 + rng.uniform(-10.0, 10.0, 3), -44.0, 44.0)
        t_bot = np.clip(base_3 + rng.uniform(-10.0, 10.0, 3), -44.0, 44.0)
        t_lft = np.clip(base_3 + rng.uniform(-10.0, 10.0, 3), -40.0, 40.0)
        t_rgt = np.clip(base_3 + rng.uniform(-10.0, 10.0, 3), -40.0, 40.0)
        xy = np.vstack([
            np.column_stack([t_top, np.full(3, E)]),
            np.column_stack([t_bot, np.full(3, -E)]),
            np.column_stack([np.full(3, -E), t_lft]),
            np.column_stack([np.full(3, E), t_rgt]),
        ])
        if min_spacing_mm(xy) < 26.0:
            continue
        p24 = compute_psl(xy, 2.45e9, uv_grid)
        p52 = compute_psl(xy, 5.2e9, uv_grid)
        p58 = compute_psl(xy, 5.8e9, uv_grid)
        score = max(p52, p58) * 1.3 + p24 * 0.7
        if score < best_score:
            best_score = score
            best_xy = xy.copy()
    return np.round(best_xy, 1)


def optimize_palm_12_molex(uv_grid):
    """Optimize 12 vertical Molex 146153 (34.9x9 mm) elements + 1 center ref inside 114x114 mm with ZERO overlap."""
    rng = np.random.default_rng(202607)
    best_score = 1e9
    best_xy = None
    # 3 vertical rows of 4 elements: y in {+37.5, 0, -37.5} (each strip spans 35mm tall -> [-55, +55] = 110mm!)
    # Ref antenna sits at center (0, 0) or corner; let's reserve (0, 0) for Ref!
    ref = np.array([[0.0, 0.0]])
    for _ in range(800):
        # Top row (4), Mid row (4, left & right of center ref), Bot row (4)
        xt = np.array([-47.0, -16.0, 16.0, 47.0]) + rng.uniform(-4.0, 4.0, 4)
        xm = np.array([-48.0, -23.0, 23.0, 48.0]) + rng.uniform(-3.0, 3.0, 4)
        xb = np.array([-47.0, -16.0, 16.0, 47.0]) + rng.uniform(-4.0, 4.0, 4)
        yt = np.full(4, 38.0) + rng.uniform(-2.0, 2.0, 4)
        ym = np.full(4, 0.0) + rng.uniform(-3.0, 3.0, 4)
        yb = np.full(4, -38.0) + rng.uniform(-2.0, 2.0, 4)
        xy = np.vstack([
            np.column_stack([xt, yt]),
            np.column_stack([xm, ym]),
            np.column_stack([xb, yb]),
        ])
        xy[:, 0] = np.clip(xy[:, 0], -51.0, 51.0)
        xy[:, 1] = np.clip(xy[:, 1], -39.0, 39.0)
        all_pts = np.vstack([xy, ref])
        if not check_molex_no_overlap(all_pts, w=12.0, h=36.0):
            continue
        if min_spacing_mm(xy) < 22.0:
            continue
        p24 = compute_psl(xy, 2.45e9, uv_grid)
        p52 = compute_psl(xy, 5.2e9, uv_grid)
        p58 = compute_psl(xy, 5.8e9, uv_grid)
        c24 = crlb_deg(xy, 2.45e9)
        score = max(p52, p58) * 1.4 + p24 * 0.6 + 0.15 * c24
        if score < best_score:
            best_score = score
            best_xy = xy.copy()
    return np.round(best_xy, 1)


def main():
    uv_grid = make_hemi_grid(step_deg=4.0, max_el_deg=65.0)

    # Candidate 1: Palm-8+1 (8 outer irregular elements inside 110x110mm)
    ang8 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.array([0.0, 0.12, -0.08, 0.10, -0.05, 0.14, -0.10, 0.06])
    r8 = np.array([53, 47, 53, 45, 53, 48, 53, 46], dtype=float)
    palm8 = np.round(np.column_stack([r8 * np.cos(ang8), r8 * np.sin(ang8)]), 1)

    # Candidate 2: Palm-12-EdgeRing (12 SMD edge-mount elements on 114x114mm PCB edge)
    palm12_edge = optimize_palm_12_edgering(uv_grid)

    # Candidate 3: Palm-12-DualRing (8 outer + 4 inner on 114x114mm PCB)
    palm12_dual = optimize_palm_12_dualring(uv_grid)

    # Candidate 4: Palm-12-MolexStag (12 vertical Molex 146153 + center ref inside 114x114mm)
    palm12_molex = optimize_palm_12_molex(uv_grid)

    # Baseline refs: v4.0 8-el 420mm L-arm & abandoned Fold12-Mini 300mm X-frame
    v4_l8 = np.array([[-120, 0], [-60, 0], [0, 0], [60, 0], [120, 0], [0, 60], [0, 120], [0, 180]], dtype=float)
    fold12 = np.array([
        [34, 34], [-34, 34], [-34, -34], [34, -34],
        [70, 56], [-56, 70], [-70, -56], [56, -70],
        [106, 102], [-102, 106], [-106, -102], [102, -106],
    ], dtype=float)

    configs = [
        ("Palm-8+1 (114mm 8소자 링)", palm8),
        ("Palm-12-Edge (114mm 테두리12소자 SMD)", palm12_edge),
        ("Palm-12-DualRing (114mm 이중링12소자)", palm12_dual),
        ("Palm-12-Molex (114mm 수직12소자 모듈형)", palm12_molex),
        ("Ref: Fold12-Mini (300mm 접이식 X자)", fold12),
        ("Ref: v4.0 L-8 (420mm L자 팔)", v4_l8),
    ]

    results = {}
    print(f"{'Array':38s} | {'N':2s} | {'Span':6s} | {'dmin':5s} | {'PSL2.4':6s} | {'PSL5.2':6s} | {'PSL5.8':6s} | {'CRLB2.4':7s} | {'Raw2.4':6s} | {'Cal2.4':6s} | {'Cal5.2':6s}")
    print("-" * 124)
    for name, xy in configs:
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        dmin = min_spacing_mm(xy)
        p24 = compute_psl(xy, 2.45e9, uv_grid)
        p52 = compute_psl(xy, 5.2e9, uv_grid)
        p58 = compute_psl(xy, 5.8e9, uv_grid)
        c24 = crlb_deg(xy, 2.45e9)
        c52 = crlb_deg(xy, 5.2e9)
        mc_raw24 = capon_mc(xy, 2.45e9, n_trials=160, calibrate_mc=False)
        mc_cal24 = capon_mc(xy, 2.45e9, n_trials=160, calibrate_mc=True)
        mc_cal52 = capon_mc(xy, 5.2e9, n_trials=160, calibrate_mc=True)
        print(
            f"{name:38s} | {len(xy):2d} | {span:5.0f}m | {dmin:4.1f}m | "
            f"{p24:6.3f} | {p52:6.3f} | {p58:6.3f} | {c24:6.2f}° | "
            f"{mc_raw24['med_deg']:5.2f}° | {mc_cal24['med_deg']:5.2f}° | {mc_cal52['med_deg']:5.2f}°"
        )
        results[name] = dict(
            N=len(xy),
            coords_mm=xy.tolist(),
            span_mm=round(float(span), 1),
            dmin_mm=round(float(dmin), 1),
            psl={"2.45G": round(p24, 3), "5.2G": round(p52, 3), "5.8G": round(p58, 3)},
            crlb_deg={"2.45G": round(c24, 2), "5.2G": round(c52, 2)},
            capon_245_raw=mc_raw24,
            capon_245_cal=mc_cal24,
            capon_52_cal=mc_cal52,
        )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palm_array_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
