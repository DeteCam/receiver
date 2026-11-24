"""
DeteCam 5.0 (Palm-12 Wall-Mounted Swivel) — AR-OPA Benchmark
Validates real-world optimal WALL-MOUNTED placement + discrete Pan/Tilt click-detent aiming
across 6 diverse indoor spaces (NO tall 2.5m light stands — standing reach 1.85~2.05 m & 1.05~1.15 m!):
  1. motel      (숙소: 4.5 x 3.8 x 2.6 m)
  2. toilet     (화장실: 3.6 x 2.8 x 2.5 m)
  3. bath       (목욕탕: 6.0 x 5.0 x 2.8 m)
  4. locker     (탈의실: 5.5 x 4.0 x 2.6 m)
  5. classroom  (40석 교실: 8.5 x 7.5 x 2.8 m, 3.6m 금속 칠판 + 착석 인체 40명)
  6. hall150    (150석 대형 강의실: 14.0 x 12.0 x 3.6 m, 기둥 2개 + 계단식 좌석)

Compares 4 strategies via Monte Carlo 3D AoA localization:
  (A) Naive (entrance-clustered single-wall flat mount, 0°/0°)
  (B) Wall-Flush Fixed (3 wall spots, 0° pan / 0° tilt — no swivel bracket)
  (C) Fixed High Corners (blind 3-corner rule regardless of room obstacles)
  (D) AR-OPA Wall-Swivel (Optimal standing-reach wall spots + discrete Pan ±30° / Tilt 0°,-15°,-25° clicks)
"""
import json
import os
import time
import numpy as np

PAN_CLICKS_DEG = (-30, -15, 0, 15, 30)
TILT_CLICKS_DEG = (0, -15, -25)


class Box:
    """Axis-aligned 3D obstacle or reflector box [x0, x1, y0, y1, z0, z1]."""
    def __init__(self, x0, x1, y0, y1, z0, z1, kind="obstacle", loss_db=6.0, refl=0.3):
        self.b = np.array([x0, x1, y0, y1, z0, z1], dtype=float)
        self.kind = kind
        self.loss_db = loss_db
        self.refl = refl


def segment_hits_box(p0, p1, box):
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        lo, hi = box.b[2 * i], box.b[2 * i + 1]
        if abs(d[i]) < 1e-9:
            if p0[i] < lo or p0[i] > hi:
                return False
        else:
            t1 = (lo - p0[i]) / d[i]
            t2 = (hi - p0[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return tmax > 0.02 and tmin < 0.98


def dist_point_to_box(p, box):
    q = np.array([
        np.clip(p[0], box.b[0], box.b[1]),
        np.clip(p[1], box.b[2], box.b[3]),
        np.clip(p[2], box.b[4], box.b[5]),
    ])
    return float(np.linalg.norm(p - q))


def swivel_boresight(wall_normal, pan_deg, tilt_deg):
    """Compute 3D boresight unit vector from wall inward normal + discrete pan (yaw) & tilt (pitch) clicks."""
    n = np.asarray(wall_normal, dtype=float)
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(n, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    # Apply horizontal pan around +Z, then vertical tilt
    ap = np.radians(pan_deg)
    at = np.radians(tilt_deg)
    horiz = np.cos(ap) * n + np.sin(ap) * right
    bore = np.cos(at) * horiz + np.sin(at) * up
    return bore / np.linalg.norm(bore)


def snap_to_detent(wall_normal, target_dir):
    """Find best discrete (pan_deg, tilt_deg) click detent aligning closest to target_dir."""
    td = np.asarray(target_dir, dtype=float)
    td = td / max(np.linalg.norm(td), 1e-9)
    best_dot = -2.0
    best = (0, 0, swivel_boresight(wall_normal, 0, 0))
    for pan in PAN_CLICKS_DEG:
        for tilt in TILT_CLICKS_DEG:
            b = swivel_boresight(wall_normal, pan, tilt)
            d = float(np.dot(b, td))
            if d > best_dot:
                best_dot = d
                best = (pan, tilt, b)
    return best


def build_spaces():
    spaces = {}

    # 1. 숙소 (Motel / 원룸): 4.5 x 3.8 x 2.6 m
    spaces["motel"] = dict(
        ko="숙소 (모텔·원룸)",
        dims=(4.5, 3.8, 2.6),
        obstacles=[
            Box(0.3, 2.3, 1.8, 3.6, 0.0, 0.75, "furniture", 4.0, 0.15),
            Box(3.8, 4.5, 2.0, 3.7, 0.0, 2.1, "furniture", 8.0, 0.25),
            Box(1.2, 3.2, 0.0, 0.45, 0.0, 0.85, "furniture", 5.0, 0.2),
            Box(0.0, 0.06, 0.8, 2.4, 0.7, 2.1, "mirror", 16.0, 0.92),
        ],
        roi_fn=lambda: np.array([
            [x, y, z]
            for x in np.linspace(0.3, 4.2, 6)
            for y in np.linspace(0.3, 3.5, 5)
            for z in (0.6, 1.3, 2.1)
        ]),
        naive=[([0.5, 0.06, 1.4], [0.0, 1.0, 0.0]), ([1.4, 0.06, 1.4], [0.0, 1.0, 0.0]), ([2.3, 0.06, 1.4], [0.0, 1.0, 0.0])],
    )

    # 2. 화장실 (Restroom): 3.6 x 2.8 x 2.5 m
    spaces["toilet"] = dict(
        ko="화장실 (칸막이·거울)",
        dims=(3.6, 2.8, 2.5),
        obstacles=[
            Box(1.5, 1.6, 1.0, 2.8, 0.15, 2.05, "partition", 11.0, 0.65),
            Box(2.55, 2.65, 1.0, 2.8, 0.15, 2.05, "partition", 11.0, 0.65),
            Box(0.0, 0.06, 0.3, 1.8, 0.9, 2.0, "mirror", 18.0, 0.95),
            Box(0.0, 0.6, 0.2, 1.9, 0.0, 0.85, "furniture", 5.0, 0.3),
        ],
        roi_fn=lambda: np.array([
            [x, y, z]
            for x in np.linspace(0.3, 3.3, 6)
            for y in np.linspace(0.3, 2.5, 5)
            for z in (0.5, 1.2, 2.1)
        ]),
        naive=[([0.08, 0.25, 1.3], [1.0, 0.0, 0.0]), ([0.08, 1.0, 1.3], [1.0, 0.0, 0.0]), ([0.08, 1.7, 1.3], [1.0, 0.0, 0.0])],
    )

    # 3. 목욕탕 (Bathhouse): 6.0 x 5.0 x 2.8 m
    spaces["bath"] = dict(
        ko="목욕탕 (습기 타일·기둥)",
        dims=(6.0, 5.0, 2.8),
        obstacles=[
            Box(2.6, 3.3, 2.1, 2.8, 0.0, 2.8, "tile", 14.0, 0.85),
            Box(0.0, 0.08, 0.5, 4.5, 0.0, 2.5, "mirror", 15.0, 0.90),
            Box(3.8, 3.9, 0.0, 2.6, 0.0, 2.1, "partition", 10.0, 0.75),
        ],
        roi_fn=lambda: np.array([
            [x, y, z]
            for x in np.linspace(0.4, 5.6, 6)
            for y in np.linspace(0.4, 4.6, 6)
            for z in (0.7, 1.5, 2.3)
        ]),
        naive=[([0.8, 0.06, 1.5], [0.0, 1.0, 0.0]), ([2.0, 0.06, 1.5], [0.0, 1.0, 0.0]), ([3.2, 0.06, 1.5], [0.0, 1.0, 0.0])],
    )

    # 4. 탈의실 (Metal Locker Room): 5.5 x 4.0 x 2.6 m
    spaces["locker"] = dict(
        ko="탈의실 (금속 사물함)",
        dims=(5.5, 4.0, 2.6),
        obstacles=[
            Box(0.0, 0.55, 0.5, 3.5, 0.0, 2.0, "metal", 22.0, 0.98),
            Box(4.95, 5.5, 0.5, 3.5, 0.0, 2.0, "metal", 22.0, 0.98),
            Box(2.3, 3.2, 1.1, 2.9, 0.0, 0.50, "furniture", 3.5, 0.15),
            Box(1.5, 4.0, 3.94, 4.0, 0.6, 2.1, "mirror", 18.0, 0.94),
        ],
        roi_fn=lambda: np.array([
            [x, y, z]
            for x in np.linspace(0.65, 4.85, 6)
            for y in np.linspace(0.4, 3.6, 5)
            for z in (0.55, 1.25, 2.05)
        ]),
        naive=[([1.0, 0.06, 1.85], [0.0, 1.0, 0.0]), ([2.75, 0.06, 1.85], [0.0, 1.0, 0.0]), ([4.5, 0.06, 1.85], [0.0, 1.0, 0.0])],
    )

    # 5. 40석 교실 (Standard Classroom)
    cls_obs = [
        Box(2.45, 6.05, 0.0, 0.08, 0.9, 2.1, "metal", 20.0, 0.98),
        Box(3.7, 4.8, 0.7, 1.3, 0.0, 1.05, "furniture", 4.0, 0.2),
    ]
    for ry in np.linspace(2.2, 6.4, 5):
        cls_obs.append(Box(0.9, 7.6, ry - 0.18, ry + 0.18, 0.72, 1.25, "body", 9.0, 0.25))
    spaces["classroom"] = dict(
        ko="40석 표준 교실 (대형 칠판)",
        dims=(8.5, 7.5, 2.8),
        obstacles=cls_obs,
        roi_fn=lambda: np.array([
            [x, y, z]
            for x in np.linspace(1.1, 7.4, 8)
            for y in np.linspace(2.0, 6.6, 5)
            for z in (0.55, 0.85, 1.15)
        ]),
        naive=[([1.5, 0.06, 1.85], [0.0, 1.0, 0.0]), ([4.25, 0.06, 1.85], [0.0, 1.0, 0.0]), ([7.0, 0.06, 1.85], [0.0, 1.0, 0.0])],
    )

    # 6. 150석 대형 강의실 (Lecture Hall)
    hall_obs = [
        Box(3.5, 10.5, 0.0, 0.1, 1.0, 2.6, "metal", 20.0, 0.96),
        Box(6.2, 7.8, 0.9, 1.6, 0.0, 1.15, "furniture", 4.0, 0.2),
        Box(4.4, 4.9, 5.5, 6.0, 0.0, 3.6, "tile", 16.0, 0.70),
        Box(9.1, 9.6, 5.5, 6.0, 0.0, 3.6, "tile", 16.0, 0.70),
    ]
    for ry in np.linspace(2.6, 10.6, 7):
        z_tier = 0.08 * (ry - 2.6)
        hall_obs.append(Box(1.2, 12.8, ry - 0.16, ry + 0.16, 0.72 + z_tier, 1.25 + z_tier, "body", 8.5, 0.25))
    spaces["hall150"] = dict(
        ko="150석 대형 강의실 (기둥·계단식)",
        dims=(14.0, 12.0, 3.6),
        obstacles=hall_obs,
        roi_fn=lambda: np.array([
            [x, y, 0.65 + 0.08 * (y - 2.6) + dz]
            for x in np.linspace(1.5, 12.5, 8)
            for y in np.linspace(2.6, 10.8, 6)
            for dz in (0.0, 0.45)
        ]),
        naive=[([2.5, 0.06, 1.95], [0.0, 1.0, 0.0]), ([7.0, 0.06, 1.95], [0.0, 1.0, 0.0]), ([11.5, 0.06, 1.95], [0.0, 1.0, 0.0])],
    )
    return spaces


def generate_wall_candidates(dims, obstacles, roi_center, allow_swivel=True):
    """
    Generate realistic STANDING-REACH wall-mounted candidates (NO tall 2.5m light stands!).
      - Standoff: 0.06 m (compact 45mm swivel bracket + 15mm puck half-depth)
      - Heights: z_high = 1.95 m (comfortable standing hand reach) & z_mid = 1.10 m (waist/podium/shelf)
      - Each candidate tests wall-mounted discrete click-detent (pan, tilt) angles!
    """
    L, W, H = dims
    z_high = min(2.00, H - 0.35)
    z_mid = 1.10
    off = 0.06  # wall-mounted swivel standoff from wall surface
    xs = np.linspace(0.45, L - 0.45, max(4, min(7, int(L / 1.4))))
    ys = np.linspace(0.45, W - 0.45, max(4, min(7, int(W / 1.4))))

    wall_specs = []
    for x in xs:
        wall_specs.append((x, off, (0.0, 1.0, 0.0), "앞벽"))
        wall_specs.append((x, W - off, (0.0, -1.0, 0.0), "뒷벽"))
    for y in ys[1:-1]:
        wall_specs.append((off, y, (1.0, 0.0, 0.0), "왼벽"))
        wall_specs.append((L - off, y, (-1.0, 0.0, 0.0), "오른벽"))
    # Podium / tabletop wedge candidate in classrooms/halls
    if W >= 6.5:
        wall_specs.append((L / 2.0, 1.15, (0.0, 1.0, 0.0), "교탁"))
    # Metal locker front face (magnetic snap-on! x=0.59 or x=L-0.59)
    for ob in obstacles:
        if ob.kind == "metal" and (ob.b[1] - ob.b[0]) > 0.3 and (ob.b[3] - ob.b[2]) > 1.5:
            ymid = 0.5 * (ob.b[2] + ob.b[3])
            if ob.b[0] < 0.2:
                wall_specs.append((ob.b[1] + off, ymid, (1.0, 0.0, 0.0), "좌측 사물함(자석)"))
            elif ob.b[1] > L - 0.2:
                wall_specs.append((ob.b[0] - off, ymid, (-1.0, 0.0, 0.0), "우측 사물함(자석)"))

    cands = []
    for (x, y, normal, wall_ko) in wall_specs:
        for z in (z_high, z_mid):
            if wall_ko == "교탁" and z > 1.3:
                continue
            p = np.array([x, y, z], dtype=float)
            # Skip if embedded inside any obstacle
            inside = any(
                (ob.b[0] - 0.03 <= x <= ob.b[1] + 0.03 and
                 ob.b[2] - 0.03 <= y <= ob.b[3] + 0.03 and
                 ob.b[4] - 0.03 <= z <= ob.b[5] + 0.03)
                for ob in obstacles
            )
            if inside:
                continue

            if allow_swivel:
                pan, tilt, bore = snap_to_detent(normal, roi_center - p)
            else:
                pan, tilt = 0, 0
                bore = swivel_boresight(normal, 0, 0)

            # Reflector penalty: Palm-12 has a 14mm rear Ni-Cu reflector shielding the wall BEHIND it,
            # so only side-grazing mirrors/metal within 0.45m in the forward hemisphere cause specular distortion.
            refl_pen = 0.0
            for ob in obstacles:
                if ob.kind in ("metal", "mirror"):
                    d_ob = dist_point_to_box(p, ob)
                    if d_ob < 0.45:
                        # Check if obstacle center is behind the node's rear shield
                        ob_c = np.array([0.5*(ob.b[0]+ob.b[1]), 0.5*(ob.b[2]+ob.b[3]), 0.5*(ob.b[4]+ob.b[5])])
                        rel = ob_c - p
                        rel_u = rel / max(np.linalg.norm(rel), 1e-6)
                        behind_shield = float(np.dot(rel_u, bore)) < -0.35
                        factor = 0.25 if behind_shield else 1.0
                        refl_pen += factor * ob.refl * ((0.45 - d_ob) / 0.45) ** 2

            cands.append((p, bore, refl_pen, wall_ko, pan, tilt))
    return cands


def link_sigma_rad(p_node, bore, p_target, obstacles):
    """Effective 3D AoA std (rad) for Palm-12 calibrated wall-mounted puck."""
    v = p_target - p_node
    dist = float(np.linalg.norm(v))
    if dist < 0.25:
        return 0.35, False
    u = v / dist
    cos_ang = float(np.dot(u, bore))
    if cos_ang < 0.15:  # blocked by rear Ni-Cu reflector
        return 1.25, False
    # Palm-12 calibrated indoor boresight AoA std ~ 2.6 deg (between 5.2G 0.57 deg and 2.45G 1.68 deg + indoor residual)
    sig_deg = 2.6 / max(cos_ang, 0.28) * (1.0 + 0.035 * dist)
    loss_db = 0.0
    for ob in obstacles:
        if segment_hits_box(p_node, p_target, ob):
            loss_db += ob.loss_db
    is_los = loss_db < 3.0
    if not is_los:
        sig_deg *= (1.0 + 0.28 * loss_db)
    return float(np.radians(min(sig_deg, 55.0))), is_los


def aoa_fim_3d(p_node, p_target, sigma_rad):
    v = p_target - p_node
    r = max(float(np.linalg.norm(v)), 0.3)
    u = v / r
    P_perp = np.eye(3) - np.outer(u, u)
    return P_perp / ((r * sigma_rad) ** 2)


def precompute_fims(cands, roi, obstacles):
    nc, nr = len(cands), len(roi)
    F = np.zeros((nc, nr, 3, 3), dtype=np.float64)
    L = np.zeros((nc, nr), dtype=bool)
    for i, c in enumerate(cands):
        p, bore = c[0], c[1]
        for j, q in enumerate(roi):
            sig, is_los = link_sigma_rad(p, bore, q, obstacles)
            F[i, j] = aoa_fim_3d(p, q, sig)
            L[i, j] = is_los
    return F, L


def score_subset(idxs, F, L, cands):
    F_sum = F[idxs].sum(axis=0) + 1e-4 * np.eye(3)[None, :, :]
    _, logdet = np.linalg.slogdet(F_sum)
    d_opt = float(np.mean(logdet))
    invF = np.linalg.inv(F_sum)
    crlb_m = np.sqrt(np.clip(np.trace(invF, axis1=1, axis2=2), 1e-9, 100.0))
    p90_crlb = float(np.percentile(crlb_m, 90))
    med_crlb = float(np.median(crlb_m))
    los_cov = float(np.mean(np.sum(L[idxs], axis=0) >= 2))
    refl_pen = sum(cands[i][2] for i in idxs)
    min_pair = 1e9
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            min_pair = min(min_pair, float(np.linalg.norm(cands[idxs[a]][0] - cands[idxs[b]][0])))
    space_pen = 3.0 * max(0.0, 1.4 - min_pair)
    zs = [cands[i][0][2] for i in idxs]
    z_div = 0.25 if (max(zs) - min(zs) >= 0.65) else 0.0
    obj = d_opt - 1.6 * p90_crlb + 1.2 * los_cov - 1.3 * refl_pen - space_pen + z_div
    return obj, med_crlb, p90_crlb, los_cov


def solve_ar_opa(dims, obstacles, roi, k=3, allow_swivel=True):
    t0 = time.perf_counter()
    roi_center = roi.mean(axis=0)
    cands = generate_wall_candidates(dims, obstacles, roi_center, allow_swivel=allow_swivel)
    F, L = precompute_fims(cands, roi, obstacles)

    chosen = []
    for _ in range(k):
        best_i, best_val = -1, -1e18
        for i in range(len(cands)):
            if i in chosen:
                continue
            val, _, _, _ = score_subset(chosen + [i], F, L, cands)
            if val > best_val:
                best_val = val
                best_i = i
        chosen.append(best_i)

    improved = True
    best_val, _, _, _ = score_subset(chosen, F, L, cands)
    while improved:
        improved = False
        for pos in range(k):
            for i in range(len(cands)):
                if i in chosen:
                    continue
                trial = chosen.copy()
                trial[pos] = i
                val, _, _, _ = score_subset(trial, F, L, cands)
                if val > best_val + 1e-4:
                    best_val = val
                    chosen = trial
                    improved = True

    dt_ms = (time.perf_counter() - t0) * 1000.0
    _, med_crlb, p90_crlb, los_cov = score_subset(chosen, F, L, cands)
    placement = [cands[i] for i in chosen]
    return placement, dt_ms, med_crlb, p90_crlb, los_cov


def fixed_high_corners(dims, roi_center):
    L, W, H = dims
    zh = min(2.00, H - 0.35)
    pts = [
        (np.array([0.4, 0.06, zh]), (0.0, 1.0, 0.0)),
        (np.array([L - 0.4, 0.06, zh]), (0.0, 1.0, 0.0)),
        (np.array([L / 2.0, W - 0.06, zh]), (0.0, -1.0, 0.0)),
    ]
    out = []
    for p, n in pts:
        _, _, b = snap_to_detent(n, roi_center - p)
        out.append((p, b))
    return out


def eval_mc_localization(placement, dims, obstacles, roi, trials=250, seed=7):
    rng = np.random.default_rng(seed)
    L, W, H = dims
    errs = []
    los2_cnt = 0
    for _ in range(trials):
        gt = roi[rng.integers(0, len(roi))] + rng.uniform(-0.12, 0.12, 3)
        gt = np.clip(gt, [0.1, 0.1, 0.2], [L - 0.1, W - 0.1, H - 0.2])
        rays, weights = [], []
        n_los = 0
        for item in placement:
            p_node, bore = item[0], item[1]
            sig, is_los = link_sigma_rad(p_node, bore, gt, obstacles)
            near_bias = 0.0
            for ob in obstacles:
                if ob.kind in ("metal", "mirror"):
                    d_ob = dist_point_to_box(p_node, ob)
                    if d_ob < 0.45:
                        ob_c = np.array([0.5*(ob.b[0]+ob.b[1]), 0.5*(ob.b[2]+ob.b[3]), 0.5*(ob.b[4]+ob.b[5])])
                        rel = ob_c - p_node
                        behind = float(np.dot(rel / max(np.linalg.norm(rel), 1e-6), bore)) < -0.35
                        near_bias += np.radians((2.2 if behind else 8.0) * ob.refl * (0.45 - d_ob) / 0.45)
            v = gt - p_node
            u_true = v / max(np.linalg.norm(v), 1e-6)
            tmp = np.array([1.0, 0.0, 0.0]) if abs(u_true[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            e1 = np.cross(u_true, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(u_true, e1)
            noise = rng.normal(0.0, sig + near_bias, 2)
            u_meas = u_true + noise[0] * e1 + noise[1] * e2
            u_meas /= np.linalg.norm(u_meas)
            w = 1.0 / (sig ** 2 + 1e-4)
            if not is_los:
                w *= 0.45
            else:
                n_los += 1
            rays.append((p_node, u_meas))
            weights.append(w)

        if n_los >= 2:
            los2_cnt += 1
        A = np.zeros((3, 3)); b = np.zeros(3)
        for (p_node, u_meas), w in zip(rays, weights):
            P = np.eye(3) - np.outer(u_meas, u_meas)
            A += w * P
            b += w * (P @ p_node)
        x_hat = np.linalg.solve(A + 1e-3 * np.eye(3), b)
        x_hat = np.clip(x_hat, [0.0, 0.0, 0.0], [L, W, H])
        errs.append(float(np.linalg.norm(x_hat - gt)))

    return {
        "med_m": round(float(np.median(errs)), 3),
        "p90_m": round(float(np.percentile(errs, 90)), 3),
        "hit_1m_pct": round(100.0 * float(np.mean(np.array(errs) <= 1.0)), 1),
        "los2_pct": round(100.0 * los2_cnt / trials, 1),
    }


def main():
    spaces = build_spaces()
    results = {}
    for key, sp in spaces.items():
        roi = sp["roi_fn"]()
        roi_center = roi.mean(axis=0)
        dims = sp["dims"]
        obs = sp["obstacles"]

        naive_pl = [(np.array(p, float), np.array(b, float) / np.linalg.norm(b)) for p, b in sp["naive"]]
        corner_pl = fixed_high_corners(dims, roi_center)
        flush_pl, _, _, _, _ = solve_ar_opa(dims, obs, roi, k=3, allow_swivel=False)
        opa_pl, dt_ms, _, _, _ = solve_ar_opa(dims, obs, roi, k=3, allow_swivel=True)

        r_naive = eval_mc_localization(naive_pl, dims, obs, roi)
        r_flush = eval_mc_localization(flush_pl, dims, obs, roi)
        r_corner = eval_mc_localization(corner_pl, dims, obs, roi)
        r_opa = eval_mc_localization(opa_pl, dims, obs, roi)

        results[key] = {
            "ko": sp["ko"],
            "dims_m": list(dims),
            "solve_ms": round(dt_ms, 1),
            "naive": r_naive,
            "wall_flush_0deg": r_flush,
            "fixed_corners": r_corner,
            "ar_opa": r_opa,
            "ar_opa_nodes": [
                {
                    "wall": c[3],
                    "pos_m": [round(float(v), 2) for v in c[0]],
                    "pan_deg": int(c[4]),
                    "tilt_deg": int(c[5]),
                    "aim": [round(float(v), 2) for v in c[1]],
                }
                for c in opa_pl
            ],
            "p90_gain_vs_naive_pct": round(100.0 * (r_naive["p90_m"] - r_opa["p90_m"]) / max(r_naive["p90_m"], 1e-6), 1),
            "p90_gain_vs_flush_pct": round(100.0 * (r_flush["p90_m"] - r_opa["p90_m"]) / max(r_flush["p90_m"], 1e-6), 1),
            "p90_gain_vs_corner_pct": round(100.0 * (r_corner["p90_m"] - r_opa["p90_m"]) / max(r_corner["p90_m"], 1e-6), 1),
        }

    overall = {}
    for k in ("naive", "wall_flush_0deg", "fixed_corners", "ar_opa"):
        overall[f"{k}_med_m"] = round(float(np.mean([r[k]["med_m"] for r in results.values()])), 3)
        overall[f"{k}_p90_m"] = round(float(np.mean([r[k]["p90_m"] for r in results.values()])), 3)
    overall["med_reduction_vs_naive_pct"] = round(100.0 * (overall["naive_med_m"] - overall["ar_opa_med_m"]) / overall["naive_med_m"], 1)
    overall["p90_reduction_vs_naive_pct"] = round(100.0 * (overall["naive_p90_m"] - overall["ar_opa_p90_m"]) / overall["naive_p90_m"], 1)
    overall["p90_reduction_vs_flush_pct"] = round(100.0 * (overall["wall_flush_0deg_p90_m"] - overall["ar_opa_p90_m"]) / overall["wall_flush_0deg_p90_m"], 1)
    overall["p90_reduction_vs_corners_pct"] = round(100.0 * (overall["fixed_corners_p90_m"] - overall["ar_opa_p90_m"]) / overall["fixed_corners_p90_m"], 1)

    summary = {"spaces": results, "overall": overall}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_opa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    for k, r in results.items():
        nodes_str = ", ".join(f"{n['wall']}(z={n['pos_m'][2]}m, pan={n['pan_deg']:+d}°, tilt={n['tilt_deg']}°)" for n in r["ar_opa_nodes"])
        print(f"  {r['ko']:22s} | Naive P90={r['naive']['p90_m']:.2f}m | Flush0°={r['wall_flush_0deg']['p90_m']:.2f}m | Corner={r['fixed_corners']['p90_m']:.2f}m | AR-OPA={r['ar_opa']['p90_m']:.2f}m (Med {r['ar_opa']['med_m']:.2f}m) | {nodes_str}")


if __name__ == "__main__":
    main()
