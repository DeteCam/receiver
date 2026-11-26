"""Independent re-evaluation harness for the DeteCam switch-ring simulation (review of SPEC-002+ Rev.2).

Differences from receiver/sim/v1_ring/run_eval.py (the estimator and the CSI synthesiser are imported
unchanged from the repository):
  * every (scene, seed, band, camera, node) gets its own RNG stream, so two configurations are compared on
    exactly the same noise / flips / human-motion draws  -> paired differences, not two independent samples
  * pose errors use shared unit normals scaled by each configuration's sigma (S5 on/off stays paired)
  * per-sample errors are stored (not only summaries) so that bootstrap confidence intervals, per-scene
    tables and coverage analyses can be computed afterwards
  * 2-position fusion is stored for both weightings, plus 2-D position error and suspect-zone statistics
"""
import argparse
import os
import sys
import time
import zlib
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
sys.path.insert(0, HERE)
from channel_sim import NEC2ArrayPattern, synthesize_ring_csi  # noqa: E402  (repository code, unchanged)
from estimators import RingMapEstimator, make_room_grid, az_err_deg  # noqa: E402
from pattern3d import make_pattern  # noqa: E402

DEV_SCENES = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
EVAL_SCENES = ["lecture_a", "lecture_b", "bathhouse_a", "bathhouse_b", "classroom_b", "lodging_b",
               "restroom_b", "locker_b", "worst_metal"]
PAIRS = {"adj": (0, 1), "diag": (0, 3), "p02": (0, 2), "p12": (1, 2), "p13": (1, 3), "p23": (2, 3)}
REFINE_PAIRS = ("adj", "p02", "diag", "p23")
REFINE_ONLY = None      # optional set of pair / group names to refine (refinement is the costly part)
# optional 3- and 4-position equal-weight fusions (position error only); set by experiment scripts
EXTRA_GROUPS = {}
SINGLE_GROUPS = {}   # {name: node indices of ONE receiver's looks}: azimuth error of that receiver's summed map, seen from the first node
# optional per-node antenna pattern key (e.g. {0: "..._inv"} for a ring hung upside down from the ceiling)
NODE_PAT = {}
# nodes that are known small moves of another node share its pose error: {node: base_node}
POSE_SHARE = {}
# optional extra outputs: estimated points per fused pair/group, per-node likelihood maps (base nodes only)
STORE_PTS = False
STORE_LL = False
STORE_LL_ALL = False    # store every node's map (float16, max-normalised) + the grid, for offline fusion studies
STORE_LL_GROUPS = {}    # {receiver b (0-3): node indices}: store the summed map of a receiver's looks instead
# virtual extra nodes: node n_base + i is a copy of base node AUG_NODES[i] (same ray paths, same pose error);
# a config can give them another antenna pattern with cfg["aug_pat"] (e.g. the head turned by 22.5 deg)
AUG_NODES = []
# per-node facing direction (rad, azimuth of the head's local +x): wall-mounted heads face into the room.
# The same yaw is applied in synthesis and in the estimator (the estimator's own heading error is added on top).
NODE_YAW = {}
# store a spatial spread (m) of each EXTRA_GROUPS fused map (for "measure more when unsure" policies)
GROUP_SPREAD = False
# optional subset of node indices to evaluate (others are skipped; pairs / groups that need a skipped node are
# left out).  Every RNG stream is keyed by (scene, band, seed, camera, node), so the kept nodes give exactly the
# same numbers as a full run.
ONLY_NODES = None
# extra exam-room scene ids (realistic rooms whose names do not start with classroom / lecture)
EXAM_EXTRA = ()


def is_exam(scene_id):
    return scene_id.startswith(("classroom", "lecture")) or scene_id in EXAM_EXTRA


class YawPat:
    """Antenna pattern turned about the vertical axis by yaw (rad)."""
    def __init__(self, p, yaw):
        self.p, self.yaw = p, float(yaw)
        self.n_elem = p.n_elem

    def element_responses(self, th, ph, aV=None, aH=None):
        if aV is None:
            return self.p.element_responses(th, np.asarray(ph) - self.yaw)
        return self.p.element_responses(th, np.asarray(ph) - self.yaw, aV, aH)


def _yaw(p, n):
    return YawPat(p, NODE_YAW[n]) if n in NODE_YAW else p


class VirtPat:
    """Coherent virtual ring from two head orientations measured against the same reference antenna position.
    Rows 0..N0-1 = orientation 0, rows N0.. = orientation 1 rescaled by a_ref0/a_ref1 so that every row is the
    ratio element/reference of its own sweep (exact for the dominant path).  V-pol manifold only (estimator use)."""
    def __init__(self, p0, p1, drot_rad=0.0):
        self.p0, self.p1, self.drot = p0, p1, float(drot_rad)
        self.n_elem = p0.n_elem + p1.n_elem

    def element_responses(self, th, ph, aV=None, aH=None):
        assert aV is None and aH is None
        e0, r0 = self.p0.element_responses(th, ph)
        e1, r1 = self.p1.element_responses(th, np.asarray(ph) - self.drot)
        return np.concatenate([e0, e1 * (r0 / (r1 + 1e-12))[None]], axis=0), r0


class VirtPatN:
    """Coherent virtual array from several head poses measured against the same reference antenna position.
    pats[0] is the first pose; rows of pose k are rescaled by a_ref0/a_refk (exact for the dominant path, and exact
    for every path when the reference response is the same in all poses, i.e. pure turns about the reference axis)."""
    def __init__(self, pats):
        self.pats = list(pats)
        self.n_elem = sum(p.n_elem for p in self.pats)

    def element_responses(self, th, ph, aV=None, aH=None):
        assert aV is None and aH is None
        e0, r0 = self.pats[0].element_responses(th, ph)
        rows = [e0]
        for p in self.pats[1:]:
            e, r = p.element_responses(th, ph)
            rows.append(e * (r0 / (r + 1e-12))[None])
        return np.concatenate(rows, axis=0), r0


TAUS = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999])


def crc(*a):
    return zlib.crc32(repr(a).encode()) & 0x7FFFFFFF


def make_df(bw_hz):
    k = 16 if bw_hz <= 20e6 else (24 if bw_hz <= 40e6 else 32)
    return np.linspace(-0.45 * bw_hz, 0.45 * bw_hz, k)


def zone_stats(ll_tot, grid, cam_xyz):
    """2-D suspect-zone statistics from a (fused) log-likelihood over the 3-D candidate grid.
    Collapses height by max, softmax over xy cells, and returns
      s    : conformity score = posterior mass of all cells at least as likely as the true cell
      area : fraction of xy cells inside the highest-posterior set for each mass level in TAUS
    """
    xy = np.round(grid[:, :2], 4)
    keys, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    cell_ll = np.full(len(keys), -np.inf)
    np.maximum.at(cell_ll, inv, ll_tot)
    p = np.exp(cell_ll - cell_ll.max())
    p /= p.sum()
    true_cell = int(np.argmin(np.sum((keys - cam_xyz[None, :2]) ** 2, axis=1)))
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    rank_true = int(np.where(order == true_cell)[0][0])
    s = float(cum[rank_true])
    n_in = np.searchsorted(cum, TAUS, side="left") + 1
    area = np.minimum(n_in, len(p)) / float(len(p))
    return s, area.astype(np.float32)


def fuse(ll_a, ll_b, sp_a, sp_b, weighting, sp_floor=10.0):
    if weighting == "spread":
        wa, wb = 1.0 / max(sp_a, sp_floor), 1.0 / max(sp_b, sp_floor)
        tot = wa + wb + 1e-12
        wa, wb = 2.0 * wa / tot, 2.0 * wb / tot
    else:
        wa = wb = 1.0
    return wa * (ll_a - ll_a.max()) + wb * (ll_b - ll_b.max())


def grid_cart(dims_est, step=0.30, zfracs=(0.25, 0.50, 0.80), max_xy_pts=None):
    """Cartesian candidate grid with an explicit step (the repository grid floors the step at 0.30 m and
    lets it grow to ~0.76 m in a 15 x 10 m room)."""
    L, W, H = dims_est
    if max_xy_pts:
        step = max(step, np.sqrt((L * W) / float(max_xy_pts)))
    xs = np.arange(0.15, L - 0.10, step)
    ys = np.arange(0.15, W - 0.10, step)
    zs = np.array(zfracs, dtype=np.float64) * H
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float64)


def grid_polar_union(dims_est, node_pos_all, base_pts=260, radii=(0.35, 0.5, 0.7, 0.9, 1.15, 1.45, 1.8, 2.2, 2.7, 3.3),
                     az_step_deg=5.0, zfracs=(0.25, 0.50, 0.80), z_near=(0.08, 0.25, 0.42, 0.60, 0.80, 0.95)):
    """Repository grid + a polar fan around every receiver position (fine in angle where a Cartesian grid is
    coarse: a 0.3 m step is 17 deg at 1 m).  Near the receiver more height layers are added because the
    elevation angle changes quickly there."""
    L, W, H = dims_est
    g = [make_room_grid(dims_est, max_xy_pts=base_pts)]
    az = np.radians(np.arange(0.0, 360.0, az_step_deg))
    for npz in node_pos_all:
        for r in radii:
            zf = z_near if r <= 1.5 else zfracs
            x = npz[0] + r * np.cos(az)
            y = npz[1] + r * np.sin(az)
            ok = (x > 0.04) & (x < L - 0.04) & (y > 0.04) & (y < W - 0.04)
            for z in zf:
                g.append(np.stack([x[ok], y[ok], np.full(ok.sum(), z * H)], axis=1))
    return np.vstack(g).astype(np.float64)


_SCENE_META = None


def scene_meta(scene_id, seed):
    global _SCENE_META
    if _SCENE_META is None:
        with open("/opt/detecam/sim/results/scene_meta.pkl", "rb") as f:
            _SCENE_META = pickle.load(f)
    return _SCENE_META[(scene_id, seed)]


NON_FURN = {"floor", "ceil", "wx0", "wxl", "wy0", "wyw", "legs", "torso", "head", "desk", "deskleg"}


def candidate_points(scene_id, seed, kind, dims_est, reg_sig, rng):
    """Candidate camera points an AR app could register (with a common registration error).
    seat: a small 3-height cloud over every desk (desk top, chest, glasses heights) -> exam rooms
    surface: points 8 cm off every wall and 7 cm under the ceiling (from the user-entered room box)
    surface_furn: surface + 6 cm above the top of every large piece of furniture
    Returns (points (P, 3), desk id per point or -1)."""
    m = scene_meta(scene_id, seed)
    L, W, H = dims_est
    off = np.array([rng.normal(0, reg_sig), rng.normal(0, reg_sig), 0.0])
    pts, dk = [], []
    if kind == "seat":
        for i, (x0, x1, y0, y1, zt) in enumerate(m["desks"]):
            xs = np.linspace(x0 - 0.10, x1 + 0.10, max(2, int(np.ceil((x1 - x0 + 0.2) / 0.2)) + 1))
            ys = np.linspace(y0 - 0.10, y1 + 0.10, max(2, int(np.ceil((y1 - y0 + 0.2) / 0.2)) + 1))
            zs = zt + np.array([0.04, 0.25, 0.51])
            XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
            p = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1) + off
            pts.append(p)
            dk.append(np.full(len(p), i))
    else:
        d = 0.08
        for z in np.arange(0.3, H - 0.2, 0.3):
            for x in np.arange(0.2, L - 0.1, 0.25):
                pts += [[[x, d, z]], [[x, W - d, z]]]
            for y in np.arange(0.2, W - 0.1, 0.25):
                pts += [[[d, y, z]], [[L - d, y, z]]]
        for x in np.arange(0.2, L - 0.1, 0.3):
            for y in np.arange(0.2, W - 0.1, 0.3):
                pts.append([[x, y, H - 0.07]])
        pts = [np.array(q, dtype=np.float64) for q in pts]
        if kind == "surface_furn":
            for lo, hi, mat, tag in m["boxes"]:
                if tag in NON_FURN or mat == "body" or hi[2] > H - 0.3 or (hi[0] - lo[0]) * (hi[1] - lo[1]) < 0.04:
                    continue
                xs = np.arange(lo[0] + 0.05, hi[0], 0.25)
                ys = np.arange(lo[1] + 0.05, hi[1], 0.25)
                for x in xs:
                    for y in ys:
                        pts.append(np.array([[x, y, hi[2] + 0.06]]) + off)
        dk = [np.full(len(q), -1) for q in pts]
    P = np.vstack(pts)
    D = np.concatenate(dk)
    keep = (P[:, 0] > 0.02) & (P[:, 0] < L - 0.02) & (P[:, 1] > 0.02) & (P[:, 1] < W - 0.02) & (P[:, 2] > 0.05) & (P[:, 2] < H - 0.02)
    return P[keep], D[keep]


DEFAULT_CFG = dict(
    grid=None,                    # None -> repository grid; ("cart", step, zfracs) or ("polar",) -> see above
    pat="spec002plus_d85_dz0",   # antenna pattern used to generate the signal
    est_pat=None,                 # pattern the estimator believes in (None -> same as pat)
    agg_mode="conj", flip_mode="sign_anchor", anchor_alpha=3.5, include_ref=True,
    robust_huber=True, exact_manifold=False, tc_step_ns=8.0, second_order=True, use_reflectors=False,
    sig_xy=0.15, sig_z=0.075, sig_head=5.0,   # receiver pose error
    sig_dims=(0.15, 0.15, 0.075),             # room-size error (user-entered room box)
    cand=None,           # None | "seat" | "surface" | "surface_furn": add AR-registered candidate points to the grid
    prior_logeps=-3.0,   # log prior weight of ordinary grid points relative to candidate points
    reg_sig=0.10,        # AR registration error of the seat layout / furniture (common offset, m)
    est_cls=None, est_kwargs=None,            # custom estimator class (must mimic RingMapEstimator API)
    synth_fn=None,                            # custom observation synthesiser (same signature as synthesize_ring_csi)
    synth_kw=None,                            # extra keyword arguments for synth_fn (risk models)
    aug_pat=None,                             # antenna pattern of the AUG_NODES copies (None -> same as pat)
    virt=None,                                # (rot_pat_key, rot_err_deg): coherent virtual ring from two head turns
    refine=None,                              # (radius_m, step_m): second pass on a fine local grid round the MAP point
    poses=None,                               # [(pattern key or None, yaw offset deg), ...]: extra head poses stacked coherently
    pose_err_deg=0.5,                         # error of each extra pose's turn angle as the estimator knows it (1 sigma)
    snr_db=20.0, cal_phase_deg=4.5, cal_amp_db=0.4, dyn_frac=0.35, n_sweep=10,
)


def _seat_score(lt, desk_of, n_desk, td, s1, s3, r, dm=None):
    """Posterior mass per desk (candidate points only) -> top-1 / top-3 desk contains the true desk.
    dm: optional (n_cam, n_desk) array that keeps the normalised desk masses (for occupancy subsets later)."""
    mk = desk_of >= 0
    v = lt[mk]
    w = np.exp(v - v.max())
    mass = np.zeros(n_desk)
    np.add.at(mass, desk_of[mk], w)
    if dm is not None:
        dm[r] = (mass / mass.sum()).astype(np.float32)
    order = np.argsort(-mass)
    s1[r] = order[0] == td
    s3[r] = td in order[:3]


def eval_file(npy_path, nec_db, configs, bw_hz=20e6, store_zone=True, node_override=None):
    D = np.load(npy_path, allow_pickle=True).item()
    scene_id, band, fc, seed = D["scene_id"], D["band"], float(D["fc"]), int(D["seed"])
    dims_true = np.asarray(D["dims"], dtype=np.float64)
    nodes_true = np.asarray(D["nodes"], dtype=np.float64)
    cams = np.asarray(D["cams"], dtype=np.float64)
    refl_true = D["user_reflectors"]
    paths = D["paths"]
    n_base = len(nodes_true)
    if AUG_NODES:
        ia = np.asarray(AUG_NODES, dtype=int)
        nodes_true = np.vstack([nodes_true, nodes_true[ia]])
        paths = {k: (np.concatenate([v, v[:, ia]], axis=1) if k in ("a", "tau", "th", "ph") else v) for k, v in paths.items()}
    df_hz = make_df(bw_hz)
    dyn_std = 45.0 if band == "2g4" else 90.0
    n_cam, n_node = len(cams), len(nodes_true)
    need = set(range(n_node)) if ONLY_NODES is None else {int(i) for i in ONLY_NODES if int(i) < n_node}

    rp = np.random.default_rng(crc("pose", scene_id, seed))
    z_dims = rp.standard_normal(3)
    z_pos = rp.standard_normal(nodes_true.shape)
    z_head = rp.standard_normal(n_node)
    for i_, b_ in POSE_SHARE.items():
        if i_ < n_node:
            z_pos[i_] = z_pos[b_]
            z_head[i_] = z_head[b_]
    z_refl = rp.standard_normal(max(len(refl_true), 1))
    z_rot = rp.standard_normal(n_node)   # pan-step error of the second head orientation (virt configs)
    z_pose = rp.standard_normal((n_node, 8))   # turn-angle errors of extra poses (poses configs); drawn last
    grids = {}
    dims_cache = {}
    cand_info = {}

    pats, ests, xcache = {}, {}, {}

    def get_pat(key):
        if key not in pats:
            pats[key] = make_pattern(nec_db[(key, band)], fc)
        return pats[key]

    out = {}
    for cname, cfg0 in configs.items():
        cfg = dict(DEFAULT_CFG)
        cfg.update(cfg0)
        est_pat_key = cfg["est_pat"] or cfg["pat"]
        sdk = tuple(cfg["sig_dims"])
        if sdk not in dims_cache:
            de = np.maximum(dims_true + z_dims * np.array(sdk), [2.0, 2.0, 1.8])
            dims_cache[sdk] = (de, make_room_grid(de, max_xy_pts=260))
        dims_est, grid0 = dims_cache[sdk]
        node_pos_est = nodes_true + z_pos * np.array([cfg["sig_xy"], cfg["sig_xy"], cfg["sig_z"]])
        heading_est = np.radians(z_head * cfg["sig_head"])
        refl_est = [(ax, float(val + 0.15 * z_refl[i]), lo, hi, rf) for i, (ax, val, lo, hi, rf) in enumerate(refl_true)]
        cls = cfg["est_cls"] or RingMapEstimator
        ekw = dict(cfg["est_kwargs"] or {})
        gspec = cfg["grid"]
        cand_kind = cfg["cand"]
        if cand_kind == "auto":      # seat layout in exam rooms only, nothing elsewhere
            cand_kind = "seat" if is_exam(scene_id) else None
        gkey = (repr(gspec), cfg["sig_xy"], cfg["sig_z"], sdk, cand_kind, cfg["reg_sig"])
        if gkey not in grids:
            if gspec is None and cand_kind:
                cpts, cdesk = candidate_points(scene_id, seed, cand_kind, dims_est, cfg["reg_sig"],
                                               np.random.default_rng(crc("reg", scene_id, seed)))
                grids[gkey] = np.vstack([grid0, cpts])
                cand_info[gkey] = (len(grid0), cdesk)
            elif gspec is None:
                grids[gkey] = grid0
            elif gspec[0] == "examz":
                # exam mode: cameras for cheating sit on a desk or a seated body -> candidate heights 0.78-1.25 m
                # (+ the same heights on the raised rows of a stepped lecture hall); other rooms keep the default grid
                if scene_id.startswith(("classroom", "lecture")):
                    L_, W_, H_ = dims_est
                    stp = max(0.30, np.sqrt((L_ * W_) / 260.0))
                    xs_ = np.arange(0.15, L_ - 0.10, stp)
                    ys_ = np.arange(0.15, W_ - 0.10, stp)
                    zs_ = [0.78, 1.0, 1.25] + ([1.6, 2.05] if scene_id == "lecture_b" else [])
                    XX, YY, ZZ = np.meshgrid(xs_, ys_, np.array(zs_), indexing="ij")
                    grids[gkey] = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float64)
                else:
                    grids[gkey] = grid0
            elif gspec[0] == "cart":
                grids[gkey] = grid_cart(dims_est, step=gspec[1], zfracs=gspec[2], max_xy_pts=(gspec[3] if len(gspec) > 3 else None))
            elif gspec[0] == "polar":
                grids[gkey] = grid_polar_union(dims_est, node_pos_est, **(gspec[1] if len(gspec) > 1 else {}))
            else:
                raise ValueError(gspec)
        grid = grids[gkey]
        logprior, desk_of = None, None
        if gkey in cand_info:
            n0, cdesk = cand_info[gkey]
            logprior = np.concatenate([np.full(n0, cfg["prior_logeps"]), np.zeros(len(cdesk))])
            desk_of = np.concatenate([np.full(n0, -1), cdesk])
        seat_eval = cand_kind == "seat" and desk_of is not None
        if seat_eval:
            dks = np.array(scene_meta(scene_id, seed)["desks"])
            dctr = np.stack([0.5 * (dks[:, 0] + dks[:, 1]), 0.5 * (dks[:, 2] + dks[:, 3]), dks[:, 4]], axis=1)
            true_desk = np.array([int(np.argmin(np.linalg.norm(dctr[:, :2] - c[:2], axis=1)
                                                 + 3.0 * np.abs(dctr[:, 2] - (c[2] - 0.25)))) for c in cams])
            seat1, seat3, deskmass = {}, {}, {}
        ekey = (gkey, est_pat_key, cfg["include_ref"], cfg["robust_huber"], cfg["exact_manifold"], str(cfg["tc_step_ns"]),
                cfg["second_order"], cfg["use_reflectors"], cfg["sig_xy"], cfg["sig_z"], cfg["sig_head"],
                cls.__name__, repr(sorted(ekw.items())))
        def pat_key_for(n, key):
            if n in NODE_PAT:
                return NODE_PAT[n]
            if n >= n_base and cfg["aug_pat"]:
                return cfg["aug_pat"]
            return key

        def est_pat_for(n):
            if cfg["poses"]:
                y0 = NODE_YAW.get(n, 0.0)
                base = get_pat(pat_key_for(n, est_pat_key))
                ps_ = [YawPat(base, y0)]
                for k_, (pk_, yo_) in enumerate(cfg["poses"]):
                    ps_.append(YawPat(get_pat(pk_) if pk_ else base,
                                      y0 + np.radians(yo_) + np.radians(cfg["pose_err_deg"]) * z_pose[n, k_]))
                return VirtPatN(ps_)
            if cfg["virt"]:
                return VirtPat(_yaw(get_pat(pat_key_for(n, est_pat_key)), n), _yaw(get_pat(cfg["virt"][0]), n),
                               np.radians(cfg["virt"][1]) * z_rot[n])
            return _yaw(get_pat(pat_key_for(n, est_pat_key)), n)
        ekey = ekey + (cfg["aug_pat"], repr(cfg["virt"]), repr(cfg["poses"]), cfg["pose_err_deg"])
        if ekey not in ests:
            ests[ekey] = [
                cls(grid=grid, node_pos_est=node_pos_est[n], dims_est=dims_est, cal_pat=est_pat_for(n),
                    df_hz=df_hz, heading_est=float(heading_est[n]), include_ref=cfg["include_ref"],
                    reflector_faces=refl_est if cfg["use_reflectors"] else (), second_order=cfg["second_order"],
                    robust_huber=cfg["robust_huber"], exact_manifold=cfg["exact_manifold"],
                    tc_step_ns=cfg["tc_step_ns"], **ekw) if n in need else None
                for n in range(n_node)
            ]
        node_ests = ests[ekey]
        synth = cfg["synth_fn"] or synthesize_ring_csi
        xkey = (synth.__name__, cfg["pat"], cfg["agg_mode"], cfg["flip_mode"], cfg["anchor_alpha"], cfg["include_ref"], cfg["snr_db"],
                cfg["cal_phase_deg"], cfg["cal_amp_db"], cfg["dyn_frac"], cfg["n_sweep"],
                repr(sorted((cfg["synth_kw"] or {}).items())), cfg["aug_pat"], repr(cfg["virt"]), repr(cfg["poses"]))
        skw = dict(cfg["synth_kw"] or {})
        extra = {}

        one = np.zeros((n_cam, n_node), dtype=np.float32)
        spread = np.zeros((n_cam, n_node), dtype=np.float32)
        losf = np.zeros((n_cam, n_node), dtype=np.float32)
        sacc = np.zeros((n_cam, n_node), dtype=np.float32)
        fused = {f"{pn}_{w}": np.zeros((n_cam, 2), dtype=np.float32) for pn in PAIRS for w in ("spread", "equal")}
        pos_err = {f"{pn}_{w}": np.zeros(n_cam, dtype=np.float32) for pn in PAIRS for w in ("spread", "equal")}
        zs = {pn: np.zeros(n_cam, dtype=np.float32) for pn in PAIRS}
        za = {pn: np.zeros((n_cam, len(TAUS)), dtype=np.float32) for pn in PAIRS}
        zs1 = np.zeros((n_cam, n_node), dtype=np.float32)
        est_pts = {}
        gspread = {}
        ll_store = np.zeros((n_cam, min(n_node, 4), len(grid)), dtype=np.float16) if STORE_LL else None
        ll_all = np.zeros((n_cam, n_node, len(grid)), dtype=np.float16) if STORE_LL_ALL else None
        ll_max = np.zeros((n_cam, n_node), dtype=np.float32) if STORE_LL_ALL else None

        for r in range(n_cam):
            cam = cams[r]
            lls, sps = [], []
            for n in range(n_node):
                if n not in need:
                    lls.append(None)
                    sps.append(0.0)
                    continue
                ck = (xkey, r, n)
                if ck not in xcache:
                    rng = np.random.default_rng(crc("sig", scene_id, band, seed, r, n))
                    paths_rn = (paths["a"][r, n, 0], paths["a"][r, n, 1], paths["tau"][r, n], paths["th"][r, n], paths["ph"][r, n])
                    skw_all = dict(n_sweep=cfg["n_sweep"], dyn_frac=cfg["dyn_frac"], dyn_std_deg=dyn_std,
                                   cal_phase_deg=cfg["cal_phase_deg"], cal_amp_db=cfg["cal_amp_db"], snr_db=cfg["snr_db"],
                                   flip_mode=cfg["flip_mode"], anchor_alpha=cfg["anchor_alpha"], agg_mode=cfg["agg_mode"],
                                   include_ref=cfg["include_ref"], weight_by_ref_amp=True, **skw)
                    p0 = _yaw(get_pat(pat_key_for(n, cfg["pat"])), n)
                    if cfg["poses"]:
                        # several sweeps of the same device (one per head pose): shared calibration residuals,
                        # independent noise / people; rows stacked, the reference row taken from the first pose
                        from risk_synth import synth_risk
                        y0 = NODE_YAW.get(n, 0.0)
                        pb = get_pat(pat_key_for(n, cfg["pat"]))
                        Xc, stc = synth_risk(paths_rn, cam, nodes_true[n], p0, df_hz, rng, **skw_all)
                        rows_ = [Xc[:p0.n_elem]]
                        for k_, (pk_, yo_) in enumerate(cfg["poses"]):
                            rngk = np.random.default_rng(crc(f"pose{k_}", scene_id, band, seed, r, n))
                            pp = YawPat(get_pat(pk_) if pk_ else pb, y0 + np.radians(yo_))
                            Xk, _ = synth_risk(paths_rn, cam, nodes_true[n], pp, df_hz, rngk, cal_in=stc.get("cal_out"), **skw_all)
                            rows_.append(Xk[:pp.n_elem])
                        Xc = np.vstack(rows_ + [Xc[p0.n_elem:]])
                    elif cfg["virt"]:
                        # two sweeps of the same device: shared calibration residuals, independent noise / people
                        from risk_synth import synth_risk
                        Xc, stc = synth_risk(paths_rn, cam, nodes_true[n], p0, df_hz, rng, **skw_all)
                        rng2 = np.random.default_rng(crc("sig2", scene_id, band, seed, r, n))
                        p1 = _yaw(get_pat(cfg["virt"][0]), n)
                        X1, _ = synth_risk(paths_rn, cam, nodes_true[n], p1, df_hz, rng2, cal_in=stc.get("cal_out"), **skw_all)
                        Xc = np.vstack([Xc[:p0.n_elem], X1[:p1.n_elem], Xc[p0.n_elem:]])
                    else:
                        Xc, stc = synth(paths_rn, cam, nodes_true[n], p0, df_hz, rng, **skw_all)
                    xcache[ck] = (Xc, stc)
                X, st = xcache[ck]
                for ek, ev in st.items():
                    if ek not in ("los_frac", "sign_acc") and np.isscalar(ev):
                        extra.setdefault(ek, np.zeros((n_cam, n_node), dtype=np.float32))[r, n] = ev
                ll, sp = node_ests[n].log_likelihood(X)
                az, _ = node_ests[n].estimate_azimuth(ll)
                vt = cam - nodes_true[n]
                one[r, n] = az_err_deg(az, float(np.arctan2(vt[1], vt[0])))
                spread[r, n], losf[r, n], sacc[r, n] = sp, st["los_frac"], st["sign_acc"]
                lls.append(ll)
                sps.append(sp)
                if STORE_LL_ALL:
                    ll_all[r, n] = np.maximum(ll - ll.max(), -6.0e4).astype(np.float16)
                    ll_max[r, n] = float(ll.max())
                    lf_ = getattr(node_ests[n], "last", None)
                    if lf_ is not None:
                        extra.setdefault("feat", np.zeros((n_cam, n_node, len(lf_)), dtype=np.float32))[r, n] = lf_
                if STORE_LL and n < 4 and not STORE_LL_GROUPS:
                    ll_store[r, n] = (ll - ll.max()).astype(np.float16)
                if store_zone:
                    zs1[r, n], _ = zone_stats(ll, grid, cam)
            if STORE_LL and STORE_LL_GROUPS:
                # one stored map per receiver = its looks summed (e.g. the three positions of a one-axis move)
                for b, idx in STORE_LL_GROUPS.items():
                    lg = sum(lls[i] - lls[i].max() for i in idx)
                    ll_store[r, b] = (lg - lg.max()).astype(np.float16)
            for pn, (i0, i1) in PAIRS.items():
                if lls[i0] is None or lls[i1] is None:
                    continue
                for w in ("spread", "equal"):
                    lt = fuse(lls[i0], lls[i1], sps[i0], sps[i1], w)
                    if logprior is not None:
                        lt = lt + logprior
                    pt = grid[int(np.argmax(lt))]
                    if seat_eval and w == "equal":
                        _seat_score(lt, desk_of, len(dctr), true_desk[r], seat1.setdefault(pn, np.zeros(n_cam, bool)),
                                    seat3.setdefault(pn, np.zeros(n_cam, bool)), r)
                    for j, ni in enumerate((i0, i1)):
                        az = float(np.arctan2(pt[1] - node_pos_est[ni][1], pt[0] - node_pos_est[ni][0]))
                        vt = cam - nodes_true[ni]
                        fused[f"{pn}_{w}"][r, j] = az_err_deg(az, float(np.arctan2(vt[1], vt[0])))
                    pos_err[f"{pn}_{w}"][r] = float(np.linalg.norm(pt[:2] - cam[:2]))
                    if STORE_PTS:
                        est_pts.setdefault(f"{pn}_{w}", np.zeros((n_cam, 3), dtype=np.float32))[r] = pt
                    if w == "equal" and store_zone:
                        zs[pn][r], za[pn][r] = zone_stats(lt, grid, cam)
            for gn, idx in EXTRA_GROUPS.items():
                if any(i >= n_node or lls[i] is None for i in idx):
                    continue
                lt = sum(lls[i] - lls[i].max() for i in idx)
                if logprior is not None:
                    lt = lt + logprior
                pt = grid[int(np.argmax(lt))]
                if seat_eval:
                    _seat_score(lt, desk_of, len(dctr), true_desk[r], seat1.setdefault(gn, np.zeros(n_cam, bool)),
                                seat3.setdefault(gn, np.zeros(n_cam, bool)), r,
                                deskmass.setdefault(gn, np.zeros((n_cam, len(dctr)), np.float32)))
                pos_err.setdefault(f"{gn}_equal", np.zeros(n_cam, dtype=np.float32))[r] = float(np.linalg.norm(pt[:2] - cam[:2]))
                if GROUP_SPREAD:
                    pg = np.exp(lt - lt.max())
                    pg /= pg.sum()
                    gsp = float(np.sqrt(np.sum(pg * np.sum((grid[:, :2] - pt[:2]) ** 2, axis=1))))
                    gspread.setdefault(gn, np.zeros(n_cam, dtype=np.float32))[r] = gsp
                if STORE_PTS:
                    est_pts.setdefault(f"{gn}_equal", np.zeros((n_cam, 3), dtype=np.float32))[r] = pt
            for gn, idx in SINGLE_GROUPS.items():
                if any(i >= n_node or lls[i] is None for i in idx):
                    continue
                lt = sum(lls[i] - lls[i].max() for i in idx)
                az, _ = node_ests[idx[0]].estimate_azimuth(lt)
                vt = cam - nodes_true[idx[0]]
                extra.setdefault("sg_" + gn, np.zeros((n_cam, 1), dtype=np.float32))[r, 0] = az_err_deg(az, float(np.arctan2(vt[1], vt[0])))
            if cfg["refine"]:
                # second pass on a fine local grid round the fused MAP point (same observations, same estimator setup)
                rad, stp = cfg["refine"]
                xs = np.arange(-rad, rad + 1e-9, stp)
                gx, gy = np.meshgrid(xs, xs)
                mk = gx ** 2 + gy ** 2 <= rad ** 2 + 1e-9
                groups = {pn: PAIRS[pn] for pn in REFINE_PAIRS}
                groups.update(EXTRA_GROUPS)
                if REFINE_ONLY is not None:
                    groups = {k: v for k, v in groups.items() if k in REFINE_ONLY}
                for gn, idx in groups.items():
                    lt = sum(lls[i] - lls[i].max() for i in idx)
                    if logprior is not None:
                        lt = lt + logprior
                    c0 = grid[int(np.argmax(lt))]
                    zz = np.unique(np.clip(c0[2] + np.array([-0.4, 0.0, 0.4]), 0.1, dims_est[2] - 0.1))
                    pts = np.array([[c0[0] + a, c0[1] + b, z] for a, b in zip(gx[mk], gy[mk]) for z in zz])
                    inside = (pts[:, 0] > 0.05) & (pts[:, 0] < dims_est[0] - 0.05) & (pts[:, 1] > 0.05) & (pts[:, 1] < dims_est[1] - 0.05)
                    pts = np.vstack([pts[inside], c0[None]])
                    llr = 0.0
                    for i in idx:
                        e = cls(grid=pts, node_pos_est=node_pos_est[i], dims_est=dims_est, cal_pat=est_pat_for(i), df_hz=df_hz,
                                heading_est=float(heading_est[i]), include_ref=cfg["include_ref"],
                                reflector_faces=refl_est if cfg["use_reflectors"] else (), second_order=cfg["second_order"],
                                robust_huber=cfg["robust_huber"], exact_manifold=cfg["exact_manifold"],
                                tc_step_ns=cfg["tc_step_ns"], **ekw)
                        li, _ = e.log_likelihood(xcache[(xkey, r, i)][0])
                        llr = llr + (li - lls[i].max())
                    pt = pts[int(np.argmax(llr))]
                    pos_err.setdefault(f"{gn}_ref", np.zeros(n_cam, dtype=np.float32))[r] = float(np.linalg.norm(pt[:2] - cam[:2]))
        out[cname] = dict(one=one, spread=spread, los_frac=losf, sign_acc=sacc, fused=fused, pos_err=pos_err,
                          zone_s=zs, zone_area=za, zone_s1=zs1)
        if extra:
            out[cname]["extra"] = extra
        if STORE_PTS:
            out[cname]["est_pts"] = est_pts
        if gspread:
            out[cname]["gspread"] = gspread
        if seat_eval:
            out[cname]["seat1"] = seat1
            out[cname]["seat3"] = seat3
            out[cname]["desk_mass"] = deskmass
            out[cname]["true_desk"] = true_desk
            out[cname]["desk_ctr"] = dctr
        if STORE_LL:
            out[cname]["ll"] = ll_store
        if STORE_LL_ALL:
            out[cname]["ll_all"] = ll_all
            out[cname]["grid"] = grid.astype(np.float32)
            out[cname]["node_pos_est"] = node_pos_est.astype(np.float32)
            out[cname]["grid"] = grid.astype(np.float32)
            out[cname]["node_pos_est"] = node_pos_est.astype(np.float32)
            out[cname]["ll_max"] = ll_max
            if logprior is not None:
                out[cname]["logprior"] = logprior.astype(np.float32)
    meta = dict(scene=scene_id, band=band, seed=seed, cam_types=list(D["cam_types"]), dims=dims_true, n_cam=n_cam,
                cams=cams, nodes=nodes_true)
    return meta, out


def run(configs, scenes, bands, seeds, data_dir, nec_path, out_path, bw_hz=20e6, store_zone=True, verbose=True):
    nec_db = np.load(nec_path, allow_pickle=True).item()
    res = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            res = pickle.load(f)
    t0 = time.time()
    for band in bands:
        for sc in scenes:
            for s in seeds:
                key = (sc, band, s)
                todo = {c: v for c, v in configs.items() if not (key in res and c in res[key][1])}
                if not todo:
                    continue
                fp = os.path.join(data_dir, f"{sc}_{band}_s{s}.npy")
                if not os.path.exists(fp):
                    print("missing", fp, flush=True)
                    continue
                meta, out = eval_file(fp, nec_db, todo, bw_hz=bw_hz, store_zone=store_zone)
                if key in res:
                    res[key][1].update(out)
                else:
                    res[key] = (meta, out)
                if verbose:
                    msg = " ".join(f"{c}={np.mean(o['one']):.1f}/{np.mean(o['fused']['diag_equal']):.1f}" for c, o in out.items())
                    print(f"[{time.time()-t0:6.0f}s] {sc:12s} {band} s{s} | {msg}", flush=True)
                with open(out_path, "wb") as f:
                    pickle.dump(res, f)
    return res


REV2_CONFIGS = {
    "BASE_D100_Conj": dict(pat="spec002plus_8_dz0_abs"),
    "EXACT_D100_MAP": dict(pat="spec002plus_8_dz0_abs", exact_manifold=True, tc_step_ns="auto"),
    "BASE_D85_Conj":  dict(pat="spec002plus_d85_dz0"),
    "EXACT_D85_MAP":  dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto"),
    "D75_EXACT_MAP":  dict(pat="spec002plus_d75_dz0", exact_manifold=True, tc_step_ns="auto"),
    "S5CAL_D85_MAP":  dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5),
    "S5CAL_D100_MAP": dict(pat="spec002plus_8_dz0_abs", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval", "all"])
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-dir", default="/opt/detecam/sim/data")
    ap.add_argument("--nec", default=os.path.join(HERE, "nec_patterns.npy"))
    ap.add_argument("--out", default="/opt/detecam/sim/results/rev2.pkl")
    args = ap.parse_args()
    scenes = {"dev": DEV_SCENES, "eval": EVAL_SCENES, "all": DEV_SCENES + EVAL_SCENES}[args.split]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run(REV2_CONFIGS, scenes, args.bands.split(","), [int(x) for x in args.seeds.split(",")], args.data_dir, args.nec, args.out)
