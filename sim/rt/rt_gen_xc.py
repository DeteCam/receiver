"""Lodging room: single-point dual-pol path data at four receiver spots away from the walls (same scene / cameras as rt_gen_xm.py):
a tripod in the free floor area, on the bed, high in the middle of the room, and a corner spot pulled 0.9 m out from both walls.
Usage: rt_gen_xc.py BAND SEED      env SPOTS='[[x,y,z],...]' OUT=prefix for other spots (e.g. several spots half a metre apart)"""
import sys, numpy as np, mitsuba as mi, sionna.rt as rt
from rtscene import motel
BAND, SEED = sys.argv[1], int(sys.argv[2]); FC = 2.437e9 if BAND == "2g4" else 5.22e9; KEEP = 70
def inside_any(p, boxes, margin=0.02):
    return any(np.all(p >= lo - margin) and np.all(p <= hi + margin) for lo, hi, mat, tag in boxes if not tag.startswith(("floor", "ceil", "w")))
rng = np.random.default_rng(1000 + SEED)
scene, info = motel(FC, rng); L, W, H = info["dims"]; cams, meta = [], []
while len(cams) < 40:
    u = rng.uniform()
    if u < 0.5:
        w = rng.integers(4); z = rng.uniform(0.2, 2.3)
        p = [[0.04, rng.uniform(0.1, W - 0.1), z], [L - 0.04, rng.uniform(0.1, W - 0.1), z], [rng.uniform(0.1, L - 0.1), 0.04, z], [rng.uniform(0.1, L - 0.1), W - 0.04, z]][w]; typ = "wall"
    elif u < 0.65:
        p = [rng.uniform(0.3, L - 0.3), rng.uniform(0.3, W - 0.3), H - 0.05]; typ = "ceiling"
    else:
        lo, hi, mat, tag = info["boxes"][rng.choice([i for i, b in enumerate(info["boxes"]) if b[3] in ("bed", "wardrobe", "tvconsole", "nightstand")])]
        p = [rng.uniform(lo[0] + 0.05, hi[0] - 0.05), rng.uniform(lo[1] + 0.05, hi[1] - 0.05), hi[2] + 0.04]; typ = "furniture:" + tag
    p = np.array(p, float)
    if inside_any(p, info["boxes"]) or p[2] > H - 0.03: continue
    cams.append(p); meta.append((len(cams), typ))
cams = np.array(cams, float)
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
for i, p in enumerate(cams):
    scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*[float(v) for v in rng.uniform(0, np.pi, 3)])))
ctr = np.array([L / 2, W / 2]); nodes = []
import os, json
SPOTS = json.loads(os.environ["SPOTS"]) if os.environ.get("SPOTS") else ((2.60, 1.10, 1.30), (1.30, 2.70, 0.90), (2.25, 1.90, 2.20), (0.90, 0.90, 1.50)); OUT = os.environ.get("OUT", "xc")
for pos in SPOTS:
    nodes.append(dict(pos=list(pos), bore=[1.0, 0.0, 0.0]))
for j, n in enumerate(nodes): scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, n["pos"]))))
paths = rt.PathSolver()(scene, synthetic_array=False, refraction=True, diffraction=True, edge_diffraction=True, max_depth=4, samples_per_src=100000)
a, tau = paths.cir(out_type="numpy", normalize_delays=False); a = a[:, 0, :, :, :, 0]; tau = tau[:, 0, :, 0, :]
th = paths.theta_t.numpy()[:, 0, :, 0, :]; ph = paths.phi_t.numpy()[:, 0, :, 0, :]
pw = (np.abs(a) ** 2).sum(2); idx = np.argsort(-pw, axis=-1)[..., :KEEP]; g = lambda x: np.take_along_axis(x, idx, -1)
out = dict(cams=cams, meta=np.array(meta, dtype=object), nodes=nodes, nano=dict(a=np.stack([g(a[:, :, 0]), g(a[:, :, 1])], 2).astype(np.complex64), tau=g(tau).astype(np.float64), th=g(th).astype(np.float32), ph=g(ph).astype(np.float32)))
np.save(f"data/{OUT}_{BAND}_{SEED}.npy", out, allow_pickle=True); print(OUT, BAND, SEED, "done", flush=True)
