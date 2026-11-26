"""Ray-traced data for the realistic rooms (plan sim/plan_20261126_real.md).
Receiver nodes = the hand-placed spots of mysim/real_scenes.py (wall hook / furniture top / table height / bathroom).
Same solver settings as mysim/gen_wall.py.  Variants: base (people as designed) | nop (nobody) | op1 (one operator).
Run: python3 mysim/gen_real.py --variants nop --scenes motel_a,... --bands 2g4,5g --seeds 0,1,2"""
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
sys.path.insert(0, HERE)
import gen_scenes as G  # noqa: E402
import real_scenes as R  # noqa: E402
from real_meta import scene_rng  # noqa: E402


def trace(scene_id, band, seed, variant, out_dir):
    fc = 2.437e9 if band == "2g4" else 5.220e9
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="detecam_real_") as mdir:
        rng = scene_rng(scene_id, seed)
        b, dims, spots, cams, types, refl, boxes, conceal = R.build_real(scene_id, fc, rng, mdir, variant)
        oris = [rng.uniform(0, np.pi, 3) for _ in range(len(cams))]
        scene = b.finalize()
        nodes = np.array([s[1:4] for s in spots], dtype=np.float64)
        yaws = np.radians([s[4] for s in spots])
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
            take = lambda arr: np.take_along_axis(arr, idx, axis=-1)  # noqa: E731
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
             node_yaw=yaws, node_labels=[s[0] for s in spots], node_types=[s[5] for s in spots], node_notes=[None] * len(spots),
             variant=variant, cams=np.asarray(cams), cam_types=types, conceal=conceal, user_reflectors=refl,
             paths={key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]})
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{scene_id}_{band}_s{seed}.npy"), D, allow_pickle=True)
    npath = float(np.mean((np.abs(D["paths"]["a"]) ** 2).sum(axis=2) > 1e-24) * G.KEEP_PATHS)
    print(f"[RT-real {variant}] {scene_id:13s} {band} s{seed} | {time.time() - t0:5.1f}s paths/link {npath:.0f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="nop")
    ap.add_argument("--scenes", default="A")
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="/opt/detecam/sim/data_real")
    args = ap.parse_args()
    scenes = {"A": R.SET_A, "B": R.SET_B, "all": R.REAL_SCENES}.get(args.scenes) or args.scenes.split(",")
    for var in args.variants.split(","):
        for bd in args.bands.split(","):
            for sc in scenes:
                for s in [int(x) for x in args.seeds.split(",")]:
                    od = os.path.join(args.root, var)
                    if os.path.exists(os.path.join(od, f"{sc}_{bd}_s{s}.npy")):
                        continue
                    trace(sc, bd, s, var, od)
    print("RT_REAL_DONE", args.variants, args.scenes, args.bands, args.seeds, flush=True)
