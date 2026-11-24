"""Full per-element ray tracing of the 12 ideal Nano-84 element positions at the repository's two node positions (checks the single-point model).
Usage: rt_gen_nanofull.py classroom|motel BAND SEED
Same scene / cameras as rt_gen.py (identical RNG stream)."""
import os, sys, json, time, numpy as np, mitsuba as mi, sionna.rt as rt
sys.path.insert(0, os.path.join(os.environ.get("DETECAM_V6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "receiver-v6")), "sim"))
from raytrace_engine import NANO12_EDGE_M
from rtscene import classroom, motel
KIND, BAND, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); FC = 2.437e9 if BAND == "2g4" else 5.22e9
FRAME = np.load("frame12.npy"); KEEP = 70
def axes(bore):
    ey = np.asarray(bore, float); ey = ey / np.linalg.norm(ey); ex = np.cross(ey, [0, 0, 1.0]); ex /= np.linalg.norm(ex); return ex, ey, np.cross(ex, ey)
def inside_any(p, boxes, margin=0.02):
    return any(np.all(p >= lo - margin) and np.all(p <= hi + margin) for lo, hi, mat, tag in boxes if not tag.startswith(("floor", "ceil", "w")))
rng = np.random.default_rng(1000 + SEED)
scene, info = (classroom if KIND == "classroom" else motel)(FC, rng); L, W, H = info["dims"]; cams, meta = [], []
if KIND == "classroom":
    for i, (sx, sy) in enumerate(info["seats"]):
        typ = rng.choice(["chest", "chest", "glasses", "desk"]); lean = info["leans"][i]
        p = dict(chest=[sx + rng.uniform(-0.05, 0.05), sy - 0.11 - lean - 0.03, 1.0], glasses=[sx, sy - 0.20 - lean - 0.02, 1.24],
                 desk=[sx + rng.uniform(-0.2, 0.2), sy - 0.40 + rng.uniform(-0.1, 0.1), 0.77])[typ]
        cams.append(p); meta.append((i, typ))
else:
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
PS = rt.PathSolver()
def solve(src):
    for n in list(scene.transmitters.keys()): scene.remove(n)
    for j, p in enumerate(src): scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, p))))
    paths = PS(scene, synthetic_array=False, refraction=True, diffraction=True, edge_diffraction=True, max_depth=4, samples_per_src=100000)
    a, tau = paths.cir(out_type="numpy", normalize_delays=False); a = a[:, 0, :, :, :, 0]; tau = tau[:, 0, :, 0, :]
    th = paths.theta_t.numpy()[:, 0, :, 0, :]; ph = paths.phi_t.numpy()[:, 0, :, 0, :]
    pw = (np.abs(a) ** 2).sum(2); idx = np.argsort(-pw, axis=-1)[..., :KEEP]; g = lambda x: np.take_along_axis(x, idx, -1)
    return dict(a=np.stack([g(a[:, :, 0]), g(a[:, :, 1])], 2).astype(np.complex64), tau=g(tau).astype(np.float64), th=g(th).astype(np.float32), ph=g(ph).astype(np.float32))
nodes = json.load(open("nano_nodes.json"))["classroom_40" if KIND == "classroom" else "motel_std"]
out = dict(cams=cams, meta=np.array(meta, dtype=object), nodes=nodes, seats=info.get("seats")); t0 = time.time()
for k, n in enumerate(nodes):
    ex, ey, ez = axes(n["bore"]); src = np.array(n["pos"]) + NANO12_EDGE_M[:, 0:1] * ex + NANO12_EDGE_M[:, 2:3] * ez
    out[f"frm{k}"] = solve(src); out[f"frm{k}"]["src"] = src
np.save(f"data/nf_{KIND}_{BAND}_{SEED}.npy", out, allow_pickle=True); print("nf", KIND, BAND, SEED, "done in", round(time.time() - t0), "s", flush=True)
