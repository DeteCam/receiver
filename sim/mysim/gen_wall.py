"""Ray-traced data for wall-mounted receivers (plan sim/plan_20261126_wall.md).
Same rooms, people and cameras as receiver/sim/v1_ring/gen_scenes.py and mysim/fresh_scenes.py (builders imported
unchanged, RNG consumed in the same order -> identical cameras with PYTHONHASHSEED=0); only the receiver nodes
change: 8 wall positions x 2 heights, each facing into the room.  Content variants (cameras fixed to the base
build): base | nop (no people) | crowd (+3 / +8 standing people) | bare (no people, no movable objects).
Run: PYTHONHASHSEED=0 python3 mysim/gen_wall.py --variants base --seeds 0 [--scenes ...]"""
import argparse
import os
import sys
import tempfile
import time
import zlib
import numpy as np
import mitsuba as mi
import sionna.rt as rt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
sys.path.insert(0, HERE)
import gen_scenes as G  # noqa: E402
import fresh_scenes as F  # noqa: E402

SHELL = {"floor", "ceil", "wx0", "wxl", "wy0", "wyw"}
FLAT_ON_WALL = {"blackboard", "whiteboard", "mirror", "bath_mirror", "full_mirror", "shower_mirrors", "window",
                "door", "winframe", "sauna_door"}
MOVABLE = {"desk", "deskleg", "podium", "tv", "lockers", "lockers_s", "lockers_n", "island_lockers", "wardrobe", "bed",
           "fridge", "bench", "fan_dryer", "metal_rack", "faucet", "monitor", "cabinets", "partition"}
HEIGHTS = (2.0, 1.5)
D_WALL = 0.10
CORNER_OFF = 0.6
LABELS = ["c0", "c1", "c2", "c3", "m0", "m1", "m2", "m3"]


def crc(*a):
    return zlib.crc32(repr(a).encode()) & 0xFFFFFFFF


def _hit(p, boxes, margin=0.10, skip=()):
    for lo, hi, mat, tag in boxes:
        if tag in SHELL or tag in skip:
            continue
        if np.all(p >= lo - margin) and np.all(p <= hi + margin):
            return (lo, hi, mat, tag)
    return None


def wall_nodes(dims, boxes):
    """8 wall positions x HEIGHTS; returns nodes (16,3), yaw (16,), labels, notes."""
    L, W, H = dims
    spec = [("y0", CORNER_OFF), ("xl", CORNER_OFF), ("yw", L - CORNER_OFF), ("x0", W - CORNER_OFF),
            ("y0", L / 2), ("xl", W / 2), ("yw", L / 2), ("x0", W / 2)]
    yaw_of = {"y0": np.pi / 2, "xl": np.pi, "yw": -np.pi / 2, "x0": 0.0}
    nodes, yaws, labels, notes = [], [], [], []
    for h in HEIGHTS:
        for (wall, s), lab in zip(spec, LABELS):
            def place(s_, out_):
                if wall == "y0":
                    return np.array([s_, out_, h])
                if wall == "yw":
                    return np.array([s_, W - out_, h])
                if wall == "x0":
                    return np.array([out_, s_, h])
                return np.array([L - out_, s_, h])
            p = place(s, D_WALL)
            note = None
            hb = _hit(p, boxes, skip=FLAT_ON_WALL)
            if hb is not None:
                lo, hi, mat, tag = hb
                # depth of the object measured from the wall
                depth = {"y0": hi[1], "yw": W - lo[1], "x0": hi[0], "xl": L - lo[0]}[wall]
                if depth <= 0.70:
                    p2 = place(s, depth + D_WALL)
                    if _hit(p2, boxes, skip=FLAT_ON_WALL | {tag}) is None:
                        p, note = p2, f"on {tag} front"
                if note is None:
                    lim = L if wall in ("y0", "yw") else W
                    for ds in (0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2, 1.5, -1.5):
                        s2 = float(np.clip(s + ds, 0.3, lim - 0.3))
                        p2 = place(s2, D_WALL)
                        if _hit(p2, boxes, skip=FLAT_ON_WALL) is None:
                            p, note = p2, f"slid {ds:+.1f} m"
                            break
                if note is None:
                    p = p.copy()
                    p[2] = min(float(hi[2]) + 0.15, H - 0.15)
                    note = f"above {tag}"
            nodes.append(p)
            yaws.append(yaw_of[wall])
            labels.append(f"{lab}_{'A' if h == HEIGHTS[0] else 'B'}")
            notes.append(note)
    return np.array(nodes), np.array(yaws), labels, notes


def corner_nodes(dims, boxes):
    """Corner-mounted heads: 4 room corners x HEIGHTS, 0.12 m from both walls, facing the room diagonal."""
    L, W, H = dims
    cs = [((0.0, 0.0), np.pi / 4), ((L, 0.0), 3 * np.pi / 4), ((L, W), -3 * np.pi / 4), ((0.0, W), -np.pi / 4)]
    nodes, yaws, labels, notes = [], [], [], []
    for h in HEIGHTS:
        for k, ((cx, cy), yaw) in enumerate(cs):
            u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            base = np.array([cx, cy, h])
            p = base + 0.12 * np.array([np.sign(u[0]), np.sign(u[1]), 0.0])
            note = None
            if _hit(p, boxes, skip=FLAT_ON_WALL) is not None:
                for s in (0.3, 0.5, 0.7, 0.9, 1.1):
                    p2 = base + 0.12 * np.array([np.sign(u[0]), np.sign(u[1]), 0.0]) + s * u
                    if _hit(p2, boxes, skip=FLAT_ON_WALL) is None:
                        p, note = p2, f"out {s:.1f} m"
                        break
                if note is None:
                    hb = _hit(p, boxes, skip=FLAT_ON_WALL)
                    p = p.copy()
                    p[2] = min(float(hb[1][2]) + 0.15, H - 0.15)
                    note = f"above {hb[3]}"
            nodes.append(p)
            yaws.append(yaw)
            labels.append(f"k{k}_{'A' if h == HEIGHTS[0] else 'B'}")
            notes.append(note)
    return np.array(nodes), np.array(yaws), labels, notes



def free_nodes(dims, boxes):
    """Free-standing spots for a movable receiver (legacy plate on a stand): the four quadrant centres and the room
    centre at 2.0 m, slid toward the room centre when inside furniture."""
    L, W, H = dims
    pts = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75), (0.5, 0.5)]
    h = min(2.0, H - 0.3)
    nodes, yaws, labels, notes = [], [], [], []
    for k, (fx, fy) in enumerate(pts):
        p = np.array([fx * L, fy * W, h]); note = None
        if _hit(p, boxes, skip=FLAT_ON_WALL) is not None:
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                p2 = np.array([(fx + (0.5 - fx) * t * 2) * L, (fy + (0.5 - fy) * t * 2) * W, h])
                if _hit(p2, boxes, skip=FLAT_ON_WALL) is None:
                    p, note = p2, f"in {t:.1f}"
                    break
            if note is None:
                hb = _hit(p, boxes, skip=FLAT_ON_WALL); p = p.copy(); p[2] = min(float(hb[1][2]) + 0.15, H - 0.15); note = f"above {hb[3]}"
        nodes.append(p); yaws.append(0.0); labels.append(f"f{k}_A"); notes.append(note)
    return np.array(nodes), np.array(yaws), labels, notes

def _build(scene_id, fc, rng, mdir, variant, crowd_rng=None):
    """Build the room; variant filters boxes / adds people before finalize. Returns scene, dims, cams, types, refl, boxes."""
    orig_box, orig_fin = G.SceneBuilder.box, G.SceneBuilder.finalize
    cap = {}

    def box(self, lo, hi, mat, tag="o"):
        if variant in ("nop", "bare") and mat == "body":
            return
        if variant == "bare" and tag in MOVABLE:
            return
        return orig_box(self, lo, hi, mat, tag)

    def fin(self):
        if variant == "crowd":
            L, W, H = cap["dims_hint"]
            n_add = 3 if L * W < 40.0 else 8
            for _ in range(n_add):
                p = G._sample_free(crowd_rng, [0.5, 0.5, 0.0], [L - 0.5, W - 0.5, 0.1], self.boxes, (L, W, H))
                self.add_person_standing(p[0], p[1])
        cap["boxes"] = list(self.boxes)
        return orig_fin(self)

    G.SceneBuilder.box, G.SceneBuilder.finalize = box, fin
    try:
        if scene_id in F.FRESH_SCENES:
            # dims are needed by the crowd hook before finalize: build once without the hook to read them
            if variant == "crowd":
                cap["dims_hint"] = FRESH_DIMS[scene_id]
            scene, dims, _nodes, cams, types, refl, _boxes = F.build_fresh(scene_id, fc, rng, mdir)
        else:
            if variant == "crowd":
                cap["dims_hint"] = ROOM_DIMS[scene_id]
            scene, dims, _nodes, cams, types, refl = G.build_scene(scene_id, fc, rng, mdir)
    finally:
        G.SceneBuilder.box, G.SceneBuilder.finalize = orig_box, orig_fin
    return scene, dims, cams, types, refl, cap["boxes"]


ROOM_DIMS = {"classroom_a": (9.0, 7.5, 3.0), "classroom_b": (9.5, 7.0, 3.0), "lecture_a": (15.0, 10.0, 3.5),
             "lecture_b": (14.0, 10.5, 3.8), "lodging_a": (4.5, 6.0, 2.5), "lodging_b": (5.0, 5.5, 2.6),
             "restroom_a": (4.0, 6.0, 2.5), "restroom_b": (4.5, 5.5, 2.5), "bathhouse_a": (10.0, 8.0, 3.0),
             "bathhouse_b": (9.5, 8.5, 3.0), "locker_a": (6.0, 5.0, 2.7), "locker_b": (6.5, 4.8, 2.7),
             "worst_metal": (5.0, 4.5, 2.6)}
FRESH_DIMS = {"classroom_c": (8.0, 8.5, 2.8), "lodging_c": (3.6, 7.2, 2.4), "office_a": (8.0, 6.0, 2.7),
              "restroom_c": (3.2, 7.5, 2.6)}


def trace(scene_id, band, seed, variant, out_dir, nodeset="wall"):
    fc = 2.437e9 if band == "2g4" else 5.220e9
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="detecam_wall_") as mdir:
        rng = np.random.default_rng(2000 + seed * 101 + abs(hash(scene_id)) % 97)
        scene, dims, cams, types, refl, boxes = _build(scene_id, fc, rng, mdir, "base")
        oris = [rng.uniform(0, np.pi, 3) for _ in range(len(cams))]   # same stream position as the original data
        global HEIGHTS
        if nodeset == "wallhi":          # the same eight wall positions mounted high (2.5 m / 2.7 m)
            HEIGHTS = (2.5, 2.7)
        nodes, yaws, labels, notes = {"corner": corner_nodes, "free": free_nodes}.get(nodeset, wall_nodes)(dims, boxes)
        HEIGHTS = (2.0, 1.5)
        if variant != "base":
            del scene
            rng_v = np.random.default_rng(2000 + seed * 101 + abs(hash(scene_id)) % 97)
            crowd_rng = np.random.default_rng(crc("crowd", scene_id, seed))
            vdir = os.path.join(mdir, "v")
            os.makedirs(vdir, exist_ok=True)
            scene, dims_v, _c, _t, _r, boxes_v = _build(scene_id, fc, rng_v, vdir, variant, crowd_rng)
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        for i, p in enumerate(cams):
            scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*map(float, oris[i]))))
        ps = rt.PathSolver()
        parts = []
        chunk = 8
        for c0 in range(0, len(nodes), chunk):
            names = []
            for j in range(c0, min(c0 + chunk, len(nodes))):
                scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, nodes[j]))))
                names.append(f"s{j}")
            paths = ps(scene, synthetic_array=False, max_depth=4, samples_per_src=100000, refraction=True,
                       diffraction=True, edge_diffraction=True, diffuse_reflection=True, max_num_paths_per_src=300000)
            a, tau = paths.cir(out_type="numpy", normalize_delays=False)
            a = a[:, 0, :, :, :, 0]
            tau = tau[:, 0, :, 0, :]
            th = paths.theta_t.numpy()[:, 0, :, 0, :]
            ph = paths.phi_t.numpy()[:, 0, :, 0, :]
            pw = (np.abs(a) ** 2).sum(axis=2)
            k = min(G.KEEP_PATHS, pw.shape[-1])
            idx = np.argsort(-pw, axis=-1)[..., :k]
            take = lambda arr: np.take_along_axis(arr, idx, axis=-1)
            part = dict(a=np.stack([take(a[:, :, 0]), take(a[:, :, 1])], axis=2).astype(np.complex64),
                        tau=take(tau).astype(np.float64), th=take(th).astype(np.float32), ph=take(ph).astype(np.float32))
            if k < G.KEEP_PATHS:
                padw = G.KEEP_PATHS - k
                for key in part:
                    pad = [(0, 0)] * (part[key].ndim - 1) + [(0, padw)]
                    part[key] = np.pad(part[key], pad)
            parts.append(part)
            for nm in names:
                scene.remove(nm)
            del paths
    D = dict(scene_id=scene_id, band=band, fc=fc, seed=seed, dims=np.array(dims, dtype=np.float64), nodes=nodes,
             node_yaw=yaws, node_labels=labels, node_notes=notes, variant=variant, cams=np.asarray(cams),
             cam_types=types, user_reflectors=refl,
             paths={key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]})
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{scene_id}_{band}_s{seed}.npy"), D, allow_pickle=True)
    moved = [f"{labels[i]}:{n}" for i, n in enumerate(notes) if n]
    print(f"[RT-{nodeset} {variant}] {scene_id:13s} {band} s{seed} | {time.time() - t0:5.1f}s {' '.join(moved)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--root", default="/opt/detecam/sim/data_wall")
    ap.add_argument("--nodeset", default="wall")
    args = ap.parse_args()
    allsc = G.DEV_SCENES + G.EVAL_SCENES + F.FRESH_SCENES
    scenes = allsc if args.scenes == "all" else args.scenes.split(",")
    for var in args.variants.split(","):
        for sc in scenes:
            for bd in args.bands.split(","):
                for s in [int(x) for x in args.seeds.split(",")]:
                    od = os.path.join(args.root, var if args.nodeset == "wall" else f"{args.nodeset}_{var}")
                    if os.path.exists(os.path.join(od, f"{sc}_{bd}_s{s}.npy")):
                        continue
                    trace(sc, bd, s, var, od, args.nodeset)
