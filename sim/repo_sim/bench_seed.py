# -*- coding: utf-8 -*-
"""
DeteCam 6.0 (Nano-2 AI) — Comprehensive 10-Space 3D Ray-Tracing + PyTorch AI Localization Benchmark
===================================================================================================
Evaluates 5 complete architectures across ALL 10 diverse indoor spaces (including 4 completely UNSEEN
rooms for ReconCovNet!) at BOTH 2.45 GHz (40 MHz) and 5.22 GHz (160 MHz):

  (A) v5_Palm12_3Node_Classic : v5.0 3-Node (120 mm, 112 mm PCB), Classical Capon + C^-1 + CM-RCLPA
  (B) Nano2_2Node_Classic     : 2-Node (96 mm, 84 mm PCB), Classical Capon + C^-1 (no AI, no mirrors)
  (C) Nano2_2Node_AI          : DeteCam 6.0 — ONLY 2 Pocket Nodes (96 mm) + ReconCovNet-v2 (PyTorch)
                                + MA-PINN 3D Virtual Mirror Anchors + AR-OPA-2 Mirror-Aware Placement
  (D) Nano1_1Node_AI          : 1-Node (96 mm) + ReconCovNet-v2 + MA-PINN Virtual Mirrors (1-panel limit study)
  (E) Nano2_2Node_AI_Phone    : DeteCam 6.0 (2 Nodes + AI) + 0-KRW Smartphone 5s Walk RSSI Gradient

Saves full per-space & aggregate metrics to sim/nano2_10space_results.json.
"""
import json
import os
import time
import numpy as np
import torch
import sys
RECV = os.environ.get("DETECAM_V6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "receiver-v6"))   # DeteCam/receiver at branch archive/v6-nano2
sys.path.insert(0, os.path.join(RECV, "sim"))

from raytrace_engine import (
    C0, NANO12_EDGE_M, PALM12_EDGE_M,
    build_10_diverse_spaces, trace_multipath_rays,
    synthesize_wideband_csi, mutual_coupling_matrix, panel_axes,
    segment_hits_box, dist_point_to_box, fresnel_reflection,
)
from train_ai_models import ReconCovNet, extract_wideband_cov_features, cm_rclpa_repair_fast

PAN_CLICKS = (-30, -15, 0, 15, 30)
TILT_CLICKS = (0, -15, -25)


def swivel_bore(normal, pan_deg, tilt_deg):
    n = np.asarray(normal, dtype=float)
    n /= max(np.linalg.norm(n), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(n, up)
    right /= max(np.linalg.norm(right), 1e-9)
    ap, at = np.radians(pan_deg), np.radians(tilt_deg)
    horiz = np.cos(ap) * n + np.sin(ap) * right
    b = np.cos(at) * horiz + np.sin(at) * up
    return b / max(np.linalg.norm(b), 1e-9)


def snap_detent(normal, target_dir):
    td = np.asarray(target_dir, dtype=float)
    td /= max(np.linalg.norm(td), 1e-9)
    best = (-2.0, 0, 0, swivel_bore(normal, 0, 0))
    for pan in PAN_CLICKS:
        for tilt in TILT_CLICKS:
            b = swivel_bore(normal, pan, tilt)
            d = float(np.dot(b, td))
            if d > best[0]:
                best = (d, pan, tilt, b)
    return best[1], best[2], best[3]


def _pen_loss_db(p0, p1, obstacles):
    loss = 0.0
    for ob in obstacles:
        if segment_hits_box(p0, p1, ob):
            loss += min(ob.pen_db, 16.0)
    return min(loss, 28.0)


def place_wall_nodes_ar_opa(space, k=2, mirror_aware=True):
    """
    AR-OPA-2 Wall-Swivel Optimizer:
    Places k wall-mounted swivel nodes to maximize 3D Fisher Information + combined Direct-LoS &
    1st-Order Virtual Mirror Visibility across the entire room (solving L-corners & pillars!).
    """
    L, W, H = space["dims"]
    obs = space["obstacles"]
    roi = space["roi"]
    center = roi.mean(axis=0)
    z_hi = min(1.98, H - 0.35)
    z_lo = 1.10
    off = 0.06

    xs = np.linspace(0.45, L - 0.45, max(4, min(6, int(L / 1.4))))
    ys = np.linspace(0.45, W - 0.45, max(4, min(6, int(W / 1.4))))
    specs = [(x, off, (0.0, 1.0, 0.0), "앞벽") for x in xs]
    specs += [(x, W - off, (0.0, -1.0, 0.0), "뒷벽") for x in xs]
    specs += [(off, y, (1.0, 0.0, 0.0), "왼벽") for y in ys[1:-1]]
    specs += [(L - off, y, (-1.0, 0.0, 0.0), "오른벽") for y in ys[1:-1]]

    cands = []
    for x, y, norm, wko in specs:
        for z in (z_hi, z_lo):
            p = np.array([x, y, z], dtype=float)
            if any(ob.b[0] - 0.05 <= x <= ob.b[1] + 0.05 and ob.b[2] - 0.05 <= y <= ob.b[3] + 0.05
                   and ob.b[4] - 0.05 <= z <= ob.b[5] + 0.05 for ob in obs):
                continue
            pan, tilt, bore = snap_detent(norm, center - p)
            cands.append((p, bore, wko, pan, tilt))

    # Subsample RoI for fast <0.15s AR-OPA scoring
    roi_s = roi[::max(1, len(roi) // 36)]
    nc, nr = len(cands), len(roi_s)
    F = np.zeros((nc, nr, 3, 3))
    Vis = np.zeros((nc, nr), dtype=bool)
    planes = [(0, 0.0), (0, L), (1, 0.0), (1, W)]

    for i, (p, bore, _, _, _) in enumerate(cands):
        for j, q in enumerate(roi_s):
            v = q - p
            r = max(float(np.linalg.norm(v)), 0.3)
            u = v / r
            ca = float(np.dot(u, bore))
            loss = _pen_loss_db(p, q, obs)
            if ca > 0.20 and loss < 3.5:
                sig = np.radians(2.2 / max(ca, 0.30) * (1.0 + 0.03 * r))
                F[i, j] += (np.eye(3) - np.outer(u, u)) / ((r * sig) ** 2)
                Vis[i, j] = True
            elif mirror_aware:
                # Check if a 1st-order wall mirror path sees around the L-corner/pillar!
                best_m = 0.0
                for ax, pval in planes:
                    q_img = q.copy(); q_img[ax] = 2.0 * pval - q_img[ax]
                    den = q_img[ax] - p[ax]
                    if abs(den) < 1e-5:
                        continue
                    t = (pval - p[ax]) / den
                    if not (0.02 < t < 0.98):
                        continue
                    hit = p + t * (q_img - p)
                    if not (0.0 <= hit[0] <= L and 0.0 <= hit[1] <= W and 0.0 <= hit[2] <= H):
                        continue
                    um = (hit - p) / max(float(np.linalg.norm(hit - p)), 1e-6)
                    cm = float(np.dot(um, bore))
                    if cm < 0.20:
                        continue
                    mloss = _pen_loss_db(p, hit, obs) + _pen_loss_db(hit, q, obs)
                    if mloss < 4.5:
                        rm = float(np.linalg.norm(q_img - p))
                        sig_m = np.radians(4.0 / max(cm, 0.30) * (1.0 + 0.03 * rm))
                        # Reflect unit ray across mirror plane for true Fisher geometry
                        um_r = um.copy(); um_r[ax] = -um_r[ax]
                        F[i, j] += 0.45 * (np.eye(3) - np.outer(um_r, um_r)) / ((rm * sig_m) ** 2)
                        Vis[i, j] = True
                        best_m = 1.0
                        break
                if best_m == 0.0:
                    sig = np.radians(38.0)
                    F[i, j] += (np.eye(3) - np.outer(u, u)) / ((r * sig) ** 2)

    def eval_set(idxs):
        Fs = F[idxs].sum(axis=0) + 1e-3 * np.eye(3)[None, :, :]
        _, ld = np.linalg.slogdet(Fs)
        vis_cov = float(np.mean(Vis[idxs].any(axis=0)))
        vis2 = float(np.mean(Vis[idxs].sum(axis=0) >= min(2, len(idxs))))
        min_d = min(
            (float(np.linalg.norm(cands[idxs[a]][0] - cands[idxs[b]][0]))
             for a in range(len(idxs)) for b in range(a + 1, len(idxs))),
            default=2.5,
        )
        z_div = 0.40 if (len(idxs) >= 2 and max(cands[i][0][2] for i in idxs) - min(cands[i][0][2] for i in idxs) >= 0.65) else 0.0
        # Reward orthogonal wall pairing (different walls) for 90-deg triangulation!
        walls = len(set(cands[i][2] for i in idxs))
        return float(np.mean(ld)) + 2.8 * vis_cov + 1.8 * vis2 + 0.35 * walls + z_div - 2.5 * max(0.0, 1.8 - min_d)

    chosen = []
    for _ in range(k):
        best_i = max((i for i in range(nc) if i not in chosen), key=lambda i: eval_set(chosen + [i]))
        chosen.append(best_i)
    best_val = eval_set(chosen)
    for _ in range(2):
        for pos in range(k):
            for i in range(nc):
                if i in chosen:
                    continue
                tr = chosen.copy(); tr[pos] = i
                v = eval_set(tr)
                if v > best_val + 1e-4:
                    best_val, chosen = v, tr

    return [
        dict(pos=cands[c][0], bore=cands[c][1], wall=cands[c][2], pan=cands[c][3], tilt=cands[c][4])
        for c in chosen
    ]


def build_room_grid(space):
    """Build 3D candidate voxel grid inside room (excluding solid structural boxes)."""
    L, W, H = space["dims"]
    obs = space["obstacles"]
    step = 0.25 if max(L, W) < 8.0 else 0.40
    xs = np.arange(0.25, L - 0.20, step)
    ys = np.arange(0.25, W - 0.20, step)
    zs = np.array([0.45, 0.95, 1.45, min(2.05, H - 0.35)])
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                inside = any(
                    ob.b[0] <= x <= ob.b[1] and ob.b[2] <= y <= ob.b[3] and ob.b[4] <= z <= ob.b[5]
                    and ob.mat in ("concrete", "steel_pec", "wet_tile", "partition")
                    for ob in obs
                )
                if not inside:
                    pts.append([x, y, z])
    return np.array(pts, dtype=float)


def precompute_node_geometry(node, grid, space):
    """
    Precompute purely geometric & ARCore-obstacle properties from a wall node to every 3D room voxel:
      - u_dir [N, 3]       : Direct ray unit vector at node
      - pen_dir [N]        : Obstacle attenuation (dB) along direct ray
      - cos_front [N]      : Direct ray dot product with node boresight
      - best_mir_u [N, 3]  : Best unobstructed 1st-order specular bounce unit vector at node
      - best_mir_gain [N]  : Reflection visibility quality [0..1] of best specular bounce
      - best_mir_exd [N]   : Excess path length (m) = d_mirror - d_direct
    Depends ONLY on ARCore 3D room geometry & node placement (ZERO target knowledge!).
    """
    L, W, H = space["dims"]
    obs = space["obstacles"]
    wall_mat = space["wall_mat"]
    rx = node["pos"]
    bore = node["bore"]
    N = len(grid)

    v_dir = grid - rx[None, :]
    d_dir = np.maximum(np.linalg.norm(v_dir, axis=1), 0.20)
    u_dir = v_dir / d_dir[:, None]
    cos_front = u_dir @ bore

    pen_dir = np.zeros(N, dtype=float)
    best_mir_u = u_dir.copy()
    best_mir_gain = np.zeros(N, dtype=float)
    best_mir_exd = np.zeros(N, dtype=float)

    planes = [
        (0, 0.0, np.array([1.0, 0.0, 0.0]), wall_mat),
        (0, L,   np.array([-1.0, 0.0, 0.0]), wall_mat),
        (1, 0.0, np.array([0.0, 1.0, 0.0]), wall_mat),
        (1, W,   np.array([0.0, -1.0, 0.0]), wall_mat),
        (2, 0.0, np.array([0.0, 0.0, 1.0]), "concrete"),
        (2, H,   np.array([0.0, 0.0, -1.0]), "concrete"),
    ]

    for j in range(N):
        q = grid[j]
        pen_dir[j] = _pen_loss_db(rx, q, obs)
        best_sc = 0.0
        for ax, pval, nvec, mat in planes:
            q_img = q.copy(); q_img[ax] = 2.0 * pval - q_img[ax]
            den = q_img[ax] - rx[ax]
            if abs(den) < 1e-6:
                continue
            t = (pval - rx[ax]) / den
            if not (0.02 < t < 0.98):
                continue
            hit = rx + t * (q_img - rx)
            if not (-0.02 <= hit[0] <= L + 0.02 and -0.02 <= hit[1] <= W + 0.02 and -0.02 <= hit[2] <= H + 0.02):
                continue
            vr = hit - rx
            ur = vr / max(float(np.linalg.norm(vr)), 1e-9)
            cf = float(np.dot(ur, bore))
            if cf < 0.16:
                continue
            pen_m = _pen_loss_db(rx, hit, obs) + _pen_loss_db(hit, q, obs)
            if pen_m > 8.0:
                continue
            eff_mat = mat
            for ob in obs:
                if ob.mat in ("mirror", "steel_pec") and dist_point_to_box(hit, ob) < 0.12:
                    eff_mat = ob.mat
                    break
            gam = abs(fresnel_reflection(abs(float(np.dot(ur, nvec))), eff_mat))
            d_m = float(np.linalg.norm(q_img - rx))
            sc = gam * cf * (10.0 ** (-pen_m / 20.0)) * (d_dir[j] / max(d_m, 0.4))
            if sc > best_sc:
                best_sc = sc
                best_mir_u[j] = ur
                best_mir_gain[j] = min(1.0, sc * 1.4)
                best_mir_exd[j] = max(0.0, d_m - d_dir[j])

    return dict(
        u_dir=u_dir, d_dir=d_dir, pen_dir=pen_dir, cos_front=cos_front,
        best_mir_u=best_mir_u, best_mir_gain=best_mir_gain, best_mir_exd=best_mir_exd,
    )


def steering_for_dirs(U, elem_local_m, bore, fc_hz):
    ex, ey, ez = panel_axes(bore)
    Rw = elem_local_m[:, 0:1] * ex + elem_local_m[:, 1:2] * ey + elem_local_m[:, 2:3] * ez
    lam = C0 / fc_hz
    return np.exp(2j * np.pi * (U @ Rw.T) / lam) / np.sqrt(len(elem_local_m))


def capon_loglik_dirs(R, A_norm, load=0.08):
    M = R.shape[0]
    tr = float(np.real(np.trace(R))) / M
    R_inv = np.linalg.inv(R + load * max(tr, 1e-12) * np.eye(M))
    quad = np.maximum(np.real(np.einsum("nm,mk,nk->n", A_norm.conj(), R_inv, A_norm)), 1e-12)
    p = 1.0 / quad
    return np.log(p / (np.max(p) + 1e-12) + 1e-6)


def localize_classical(node_obs_list, precomps, grid):
    """Pure Classical Capon triangulation (no AI, no virtual mirror anchors)."""
    total = np.zeros(len(grid), dtype=float)
    for obs, pre in zip(node_obs_list, precomps):
        s_dir = capon_loglik_dirs(obs["R_all"], pre["A_dir"], load=0.10)
        front_w = np.clip((pre["cos_front"] - 0.10) / 0.30, 0.15, 1.0)
        soft_w = 1.0 / (1.0 + np.exp((obs["cv_f"] - 0.08) / 0.02))
        total += (0.35 + 0.65 * soft_w) * front_w * s_dir
    return grid[int(np.argmax(total))]


def localize_ma_pinn(node_obs_list, precomps, grid, phone_prior=None):
    """
    DeteCam 6.0 MA-PINN (Bayesian Log-Sum-Exp Multipath-Assisted Physics-Informed Neural Localizer):
      1. Uses PyTorch ReconCovNet-v4 super-resolved covariance R_ai for sharp direct/dominant peaks.
      2. Uses exact Bayesian generative mixture P_n(q) = w_dir(q)*L_dir(q) + w_mir(q)*L_mir(q)
         derived from the ARCore 3D obstacle map:
           - On Direct LoS voxels: w_dir dominates (ZERO false penalty!), while matching specular echoes
             strictly BOOST likelihood as FREE 3rd/4th Virtual Mirror Anchors!
           - On Blocked NLoS blind spots (behind L-corner, wet pillar, toilet cubicle, metal locker):
             w_dir naturally attenuates by 10^(-pen_dir/20), handing off smoothly to w_mir so the
             unobstructed specular wall/mirror bounce folds around the obstacle!
      3. P_tx-invariant Differential Power-Ratio Fusion (Delta P_12 = 10 log10(P_1/P_2)) between the
         2 physical nodes: cancels unknown camera TX power completely and collapses shallow-angle depth
         ambiguity in large halls (classroom_40, hall_150) at ZERO extra hardware cost!
      4. Optional 0-KRW smartphone 5-second walk RSSI gradient prior (private rooms).
    """
    N = len(grid)
    total = np.zeros(N, dtype=float)
    pred_amps = []

    for obs, pre in zip(node_obs_list, precomps):
        s_dir_ai = capon_loglik_dirs(obs["R_ai"], pre["A_dir"], load=0.08)
        s_mir_ai = capon_loglik_dirs(obs["R_ai"], pre["A_mir"], load=0.08)
        s_mir_all = capon_loglik_dirs(obs["R_all"], pre["A_mir"], load=0.10)

        # Normalized positive likelihoods in (0, 1] -> scale-consistent Bayesian mixture!
        L_dir = np.exp(np.clip(s_dir_ai, -12.0, 0.0))
        L_mir = np.exp(np.clip(np.maximum(s_mir_ai, s_mir_all), -12.0, 0.0))

        # Exact physical path amplitudes predicted by ARCore 3D room/obstacle map
        w_dir = np.clip(pre["cos_front"], 0.08, 1.0) * (10.0 ** (-pre["pen_dir"] / 20.0))
        w_mir = 0.55 * pre["best_mir_gain"]
        w_norm = w_dir + w_mir + 1e-6

        # Bayesian mixture likelihood: NEVER penalizes LoS when reflection absent, smoothly recovers NLoS!
        p_mix = (w_dir * L_dir + w_mir * L_mir) / w_norm
        node_logp = np.log(p_mix + 1e-4)

        # Slight reliability boost for direct-dominant node (higher SNR / lower frequency selectivity)
        cv_w = 0.70 + 0.30 / (1.0 + np.exp((obs["cv_f"] - 0.12) / 0.03))
        total += cv_w * node_logp

        # Expected total received amplitude at voxel q (for P_tx-invariant Delta P_12 ratio!)
        d_mir = pre["d_dir"] + pre["best_mir_exd"]
        eff_amp = np.sqrt((w_dir / np.maximum(pre["d_dir"], 0.35)) ** 2 + (w_mir / np.maximum(d_mir, 0.50)) ** 2) + 1e-5
        pred_amps.append(eff_amp)

    # P_tx-invariant Differential Power-Ratio Fusion (when >=2 physical nodes present!)
    if len(node_obs_list) >= 2:
        p0 = max(node_obs_list[0]["raw_pow"], 1e-12)
        p1 = max(node_obs_list[1]["raw_pow"], 1e-12)
        dp_meas_db = 10.0 * np.log10(p0 / p1)
        dp_pred_db = 20.0 * np.log10(pred_amps[0] / pred_amps[1])
        # Conservative sigma=6.5 dB accommodates small-scale indoor Rayleigh fading while breaking depth ambiguity!
        total += -0.45 * ((dp_meas_db - dp_pred_db) / 6.5) ** 2

    if phone_prior is not None:
        total += phone_prior

    # Sub-voxel weighted centroid refinement over top neighboring voxels within 0.55m of MAP peak
    best_idx = int(np.argmax(total))
    top_k = np.argsort(-total)[:6]
    near = [i for i in top_k if np.linalg.norm(grid[i] - grid[best_idx]) <= 0.55]
    w = np.exp((total[near] - total[best_idx]) * 2.0)
    w /= np.sum(w)
    return np.sum(grid[near] * w[:, None], axis=0)


def run_benchmark():
    t0 = time.perf_counter()
    spaces = build_10_diverse_spaces(rng_seed=2026)

    # Load REAL trained PyTorch ReconCovNet-v4 weights!
    w_path = os.path.join(RECV, "firmware", "core", "weights", "recon_covnet.pt")
    model = ReconCovNet(n_ant=12, in_ch=6, hidden=192, rank=3)
    model.load_state_dict(torch.load(w_path, map_location="cpu", weights_only=True))
    model.eval()

    C_nano = {f: mutual_coupling_matrix(NANO12_EDGE_M, f) for f in (2.45e9, 5.22e9)}
    C_inv_nano = {f: np.linalg.inv(C) for f, C in C_nano.items()}
    C_palm = {f: mutual_coupling_matrix(PALM12_EDGE_M, f) for f in (2.45e9, 5.22e9)}
    C_inv_palm = {f: np.linalg.inv(C) for f, C in C_palm.items()}

    rng = np.random.default_rng(int(os.environ.get('SEED', 202606)))
    unseen_set = {"classroom_40", "hall_150", "glass_office", "cluttered_dorm"}
    methods = [
        "v5_Palm12_3Node_Classic",
        "Nano2_2Node_Classic",
        "Nano2_2Node_AI",
        "Nano1_1Node_AI",
        "Nano2_2Node_AI_Phone",
    ]
    report = dict(per_space={}, aggregate={})

    print(f"{'Space':24s} | {'Band':5s} | {'3N Palm12':10s} | {'2N Nano Cl':10s} | {'2N Nano AI':10s} | {'1N Nano AI':10s} | {'2N+Phone':10s}")
    print("-" * 102)

    agg = {b: {m: [] for m in methods} for b in ("2.45G", "5.22G")}
    agg_nlos = {b: {m: [] for m in methods} for b in ("2.45G", "5.22G")}
    agg_unseen = {b: {m: [] for m in methods} for b in ("2.45G", "5.22G")}

    for sk, sp in spaces.items():
        grid = build_room_grid(sp)
        nodes_3 = place_wall_nodes_ar_opa(sp, k=3, mirror_aware=False)
        nodes_2 = place_wall_nodes_ar_opa(sp, k=2, mirror_aware=True)
        nodes_1 = [nodes_2[0]]

        # Precompute room geometry once per node set
        pre_3_base = [precompute_node_geometry(n, grid, sp) for n in nodes_3]
        pre_2_base = [precompute_node_geometry(n, grid, sp) for n in nodes_2]

        # Test 28 diverse target camera locations per space (including NLoS corners, behind pillars, desks!)
        roi = sp["roi"]
        n_trials = min(28, len(roi))
        trial_idxs = rng.choice(len(roi), size=n_trials, replace=False)

        report["per_space"][sk] = dict(ko=sp["ko"], unseen=sk in unseen_set, dims=sp["dims"], bands={})

        for band_lbl, fc, bw, bflag in [("2.45G", 2.45e9, 40e6, 0.0), ("5.22G", 5.22e9, 160e6, 1.0)]:
            # Attach band-specific steering matrices to precomputed geometry
            pre_3 = []
            for n, p0 in zip(nodes_3, pre_3_base):
                d = dict(p0)
                d["A_dir"] = steering_for_dirs(p0["u_dir"], PALM12_EDGE_M, n["bore"], fc)
                pre_3.append(d)
            pre_2 = []
            for n, p0 in zip(nodes_2, pre_2_base):
                d = dict(p0)
                d["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], fc)
                d["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], fc)
                pre_2.append(d)
            pre_1 = [pre_2[0]]

            errs = {m: [] for m in methods}
            for ti in trial_idxs:
                tx = roi[ti]
                snr = float(rng.uniform(11.0, 20.0))

                # Realistic uncalibrated coupling drift per node (temp / wall / manufacturing)
                def observe_node(node, elem_m, C_nom, C_inv_nom, run_ai=False):
                    M = len(elem_m)
                    rays = trace_multipath_rays(tx, node["pos"], node["bore"], sp)
                    pjit = np.exp(1j * rng.normal(0.0, np.radians(4.5), M))
                    dmat = 0.07 * (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))) / np.sqrt(2.0 * M)
                    C_true = (np.diag(pjit) @ C_nom) + dmat

                    feat_acc = np.zeros((6, M, M), dtype=np.float32)
                    R_all_acc = np.zeros((M, M), dtype=np.complex128)
                    cvs = []
                    tot_pow = 0.0
                    for _ in range(4):
                        H_raw = synthesize_wideband_csi(rays, node["bore"], elem_m, fc, bw, n_sub=32, snr_db=snr, flip_prob=0.15, C_mat=C_true, rng=rng)
                        tot_pow += float(np.mean(np.abs(H_raw) ** 2))
                        H_fix = cm_rclpa_repair_fast(H_raw, elem_m, fc, node["bore"], C_nom)
                        H_cal = C_inv_nom @ H_fix
                        amp = np.abs(H_cal)
                        cvs.append(float(np.median(np.std(amp, axis=1) / (np.mean(amp, axis=1) + 1e-12))))
                        f_p, R_p = extract_wideband_cov_features(H_cal, bflag)
                        feat_acc += f_p
                        R_all_acc += R_p
                    feat_acc /= 4.0
                    R_all_acc /= 4.0

                    R_ai = R_all_acc
                    if run_ai:
                        with torch.no_grad():
                            xt = torch.from_numpy(feat_acc[None])
                            bt = torch.tensor([bflag], dtype=torch.float32)
                            R_ai = model(xt, bt)[0].numpy()

                    return dict(R_all=R_all_acc, R_ai=R_ai, cv_f=float(np.median(cvs)), raw_pow=tot_pow / 4.0, is_los=rays[0]["is_los"])

                obs_3 = [observe_node(n, PALM12_EDGE_M, C_palm[fc], C_inv_palm[fc], run_ai=False) for n in nodes_3]
                obs_2 = [observe_node(n, NANO12_EDGE_M, C_nano[fc], C_inv_nano[fc], run_ai=True) for n in nodes_2]
                max_p2 = max(o["raw_pow"] for o in obs_2) + 1e-12
                for o in obs_2:
                    o["rx_pow_rel"] = np.sqrt(o["raw_pow"] / max_p2)

                # Optional 0-KRW smartphone 5-second walk RSSI spatial prior (coarse ~1.1m sigma)
                phone_est = tx + rng.normal(0.0, [0.85, 0.85, 0.70])
                phone_prior = -0.35 * np.sum(((grid - phone_est[None, :]) / np.array([1.2, 1.2, 1.1])) ** 2, axis=1)

                p_v5 = localize_classical(obs_3, pre_3, grid)
                p_n2c = localize_classical(obs_2, pre_2, grid)
                p_n2a = localize_ma_pinn(obs_2, pre_2, grid, phone_prior=None)
                p_n1a = localize_ma_pinn([obs_2[0]], pre_1, grid, phone_prior=None)
                p_n2p = localize_ma_pinn(obs_2, pre_2, grid, phone_prior=phone_prior)

                ests = {
                    "v5_Palm12_3Node_Classic": p_v5,
                    "Nano2_2Node_Classic": p_n2c,
                    "Nano2_2Node_AI": p_n2a,
                    "Nano1_1Node_AI": p_n1a,
                    "Nano2_2Node_AI_Phone": p_n2p,
                }
                is_nlos_case = any(not o["is_los"] for o in obs_2)
                for m, est in ests.items():
                    e = float(np.linalg.norm(est - tx))
                    errs[m].append(e)
                    agg[band_lbl][m].append(e)
                    if is_nlos_case:
                        agg_nlos[band_lbl][m].append(e)
                    if sk in unseen_set:
                        agg_unseen[band_lbl][m].append(e)

            b_res = {}
            for m in methods:
                arr = np.array(errs[m])
                b_res[m] = dict(
                    med_m=round(float(np.median(arr)), 3),
                    p90_m=round(float(np.percentile(arr, 90)), 3),
                    within_1m_pct=round(float(100.0 * np.mean(arr <= 1.0)), 1),
                )
            report["per_space"][sk]["bands"][band_lbl] = b_res
            tag = "*" if sk in unseen_set else " "
            print(
                f"{tag}{sk:23s} | {band_lbl:5s} | "
                f"{b_res['v5_Palm12_3Node_Classic']['med_m']:4.2f}({b_res['v5_Palm12_3Node_Classic']['p90_m']:4.2f}) | "
                f"{b_res['Nano2_2Node_Classic']['med_m']:4.2f}({b_res['Nano2_2Node_Classic']['p90_m']:4.2f}) | "
                f"{b_res['Nano2_2Node_AI']['med_m']:4.2f}({b_res['Nano2_2Node_AI']['p90_m']:4.2f}) | "
                f"{b_res['Nano1_1Node_AI']['med_m']:4.2f}({b_res['Nano1_1Node_AI']['p90_m']:4.2f}) | "
                f"{b_res['Nano2_2Node_AI_Phone']['med_m']:4.2f}({b_res['Nano2_2Node_AI_Phone']['p90_m']:4.2f})"
            )

    print("\n=== AGGREGATE 10-SPACE SUMMARY (Median m / P90 m / <=1.0m Rate) ===")
    for grp_name, grp in [("all_10_spaces", agg), ("unseen_4_spaces", agg_unseen), ("nlos_blindspot_cases", agg_nlos)]:
        report["aggregate"][grp_name] = {}
        print(f"\n[{grp_name}]")
        for b in ("2.45G", "5.22G"):
            report["aggregate"][grp_name][b] = {}
            for m in methods:
                arr = np.array(grp[b][m])
                d = dict(
                    med_m=round(float(np.median(arr)), 3),
                    p90_m=round(float(np.percentile(arr, 90)), 3),
                    within_1m_pct=round(float(100.0 * np.mean(arr <= 1.0)), 1),
                    n=int(len(arr)),
                )
                report["aggregate"][grp_name][b][m] = d
                print(f"  {b} | {m:24s} | Med {d['med_m']:.2f} m | P90 {d['p90_m']:.2f} m | <=1.0m: {d['within_1m_pct']:5.1f}% (n={d['n']})")

    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    out_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("OUTNAME", "seedtest.json"))
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved 10-space benchmark report -> {out_p} (runtime {report['runtime_s']}s)")


if __name__ == "__main__":
    run_benchmark()
