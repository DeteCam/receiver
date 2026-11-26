"""Extra ray-traced data for the review: same rooms / people / cameras as receiver/sim/v1_ring/gen_scenes.py
(build_scene is imported unchanged and the RNG is consumed in the same order), but with the receiver nodes moved.

node variants (tag):
  z075 / z120 / z150 ...   same corner xy, mounting height changed (R8 says "desk or shelf", V1 assumed 2.1-2.8 m)
  off010 / off100          distance from both corner walls changed (V1: 0.35-0.60 m)
  midwall                  four mid-wall positions instead of corners
Run with PYTHONHASHSEED=0 so that hash(scene_id) inside the repository code is reproducible.
"""
import argparse
import os
import sys
import tempfile
import time
import numpy as np
import mitsuba as mi
import sionna.rt as rt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
import gen_scenes as G  # noqa: E402  (repository code, unchanged)


def _free(p, boxes, margin=0.06):
    for lo, hi, mat, tag in boxes:
        if tag in ("floor", "ceil", "wx0", "wxl", "wy0", "wyw"):
            continue
        if np.all(p >= lo - margin) and np.all(p <= hi + margin):
            return False, hi[2]
    return True, None


# head-move offsets (cm, x = along one wall, y = the other horizontal axis, z = up); node index of offset k of
# receiver b = 4 + b * len(HEAD_OFFSETS_CM) + k
HEAD_OFFSETS_CM = ([(s * a, 0, 0) for a in (5, 10, 20) for s in (1, -1)] +
                   [(0, 0, s * a) for a in (5, 10, 20) for s in (1, -1)] +
                   [(0, s * a, 0) for a in (5, 10) for s in (1, -1)] +
                   [(sx * 10, 0, sz * 10) for sx in (1, -1) for sz in (1, -1)] +
                   [(sx * 3.536, sy * 3.536, 0) for sx in (1, -1) for sy in (1, -1)] +
                   [(sx * 7.071, sy * 7.071, 0) for sx in (1, -1) for sy in (1, -1)])


def move_nodes(nodes, dims, tag, boxes):
    L, W, H = dims
    nodes = np.array(nodes, dtype=np.float64)
    if tag.startswith("z"):
        nodes[:, 2] = float(tag[1:]) / 100.0
    elif tag.startswith("off"):
        d = float(tag[3:]) / 100.0
        nodes[:, 0] = [d, L - d, d, L - d]
        nodes[:, 1] = [d, d, W - d, W - d]
    elif tag == "midwall":
        d = float(np.min(nodes[:, :2]))
        z = nodes[0, 2]
        nodes = np.array([[L / 2, d, z], [L - d, W / 2, z], [d, W / 2, z], [L / 2, W - d, z]])
    elif tag == "head":  # dense set of small head moves around every receiver (see HEAD_OFFSETS_CM)
        base = nodes.copy()
        extra = [base[b] + np.array(o) / 100.0 for b in range(len(base)) for o in HEAD_OFFSETS_CM]
        nodes = np.vstack([base, np.array(extra)])
    elif tag.startswith("jit"):  # small motorised moves: each receiver also at +-d along the wall (x) and +-d vertically
        d = float(tag[3:]) / 100.0
        base = nodes.copy()
        extra = []
        for b in range(len(base)):
            for dx, dz in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)):
                extra.append(base[b] + np.array([dx, 0.0, dz]))
        nodes = np.vstack([base, np.array(extra)])
    elif tag == "ceilc":  # receiver 0 hung from the middle of the ceiling, the other three stay in their corners
        nodes[0] = [L / 2, W / 2, H - 0.12]
    elif tag != "base":
        raise ValueError(tag)
    notes = []
    for i in range(len(nodes)):
        for _ in range(4):
            ok, top = _free(nodes[i], boxes)
            if ok:
                break
            nodes[i, 2] = min(top + 0.15, H - 0.25)  # the receiver would simply be put on top of that object
            notes.append((i, round(float(nodes[i, 2]), 2)))
    return nodes, notes


def trace(scene_id, band, seed, out_dir, tag, builder=None):
    fc = 2.437e9 if band == "2g4" else 5.220e9
    rng = np.random.default_rng(2000 + seed * 101 + abs(hash(scene_id)) % 97)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="detecam_ext_") as mdir:
        b_boxes = {}
        if builder is None:
            orig_finalize = G.SceneBuilder.finalize

            def _fin(self):
                b_boxes["boxes"] = list(self.boxes)
                return orig_finalize(self)
            G.SceneBuilder.finalize = _fin
            try:
                scene, dims, nodes, cams, cam_types, user_reflectors = G.build_scene(scene_id, fc, rng, mdir)
            finally:
                G.SceneBuilder.finalize = orig_finalize
        else:
            scene, dims, nodes, cams, cam_types, user_reflectors, b_boxes["boxes"] = builder(scene_id, fc, rng, mdir)
        nodes, notes = move_nodes(nodes, dims, tag, b_boxes["boxes"])
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        for i, p in enumerate(cams):
            ori = rng.uniform(0, np.pi, 3)
            scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*map(float, ori))))
        ps = rt.PathSolver()
        chunk = 20  # transmitters per solver call (100+ at once runs out of memory)
        parts = []
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
            if k < G.KEEP_PATHS:  # pad so that every chunk has the same path axis
                padw = G.KEEP_PATHS - k
                for key in part:
                    pad = [(0, 0)] * (part[key].ndim - 1) + [(0, padw)]
                    part[key] = np.pad(part[key], pad)
            parts.append(part)
            for nm in names:
                scene.remove(nm)
            del paths
    D = dict(scene_id=scene_id, band=band, fc=fc, seed=seed, dims=np.array(dims, dtype=np.float64), nodes=nodes,
             cams=cams, cam_types=cam_types, user_reflectors=user_reflectors, node_tag=tag, node_notes=notes,
             paths={key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]})
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{scene_id}_{band}_s{seed}.npy"), D, allow_pickle=True)
    print(f"[RT-ext {tag}] {scene_id:13s} {band} s{seed} | {time.time()-t0:5.1f}s {notes if notes else ''}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="/opt/detecam/sim/data_ext")
    args = ap.parse_args()
    scenes = G.DEV_SCENES + G.EVAL_SCENES if args.scenes == "all" else args.scenes.split(",")
    for tag in args.tags.split(","):
        for sc in scenes:
            for b in args.bands.split(","):
                for s in [int(x) for x in args.seeds.split(",")]:
                    od = os.path.join(args.root, tag)
                    if os.path.exists(os.path.join(od, f"{sc}_{b}_s{s}.npy")):
                        continue
                    trace(sc, b, s, od, tag)
