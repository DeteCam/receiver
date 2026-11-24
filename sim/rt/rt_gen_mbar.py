"""Lodging room: four 16-element 47 cm bars (2 horizontal + 2 vertical) at room corners. Usage: rt_gen_mbar.py BAND SEED
Same scene / cameras as rt_gen.py motel (identical RNG stream)."""
import sys, time, numpy as np, mitsuba as mi, sionna.rt as rt
from rtscene import motel
BAND, SEED = sys.argv[1], int(sys.argv[2]); FC = 2.437e9 if BAND == "2g4" else 5.22e9
BAR = (np.arange(16) - 7.5) * 0.03125; KEEP = 70
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
PS = rt.PathSolver()
def solve(src):
    for n in list(scene.transmitters.keys()): scene.remove(n)
    for j, p in enumerate(src): scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, p))))
    paths = PS(scene, synthetic_array=False, refraction=True, diffraction=True, edge_diffraction=True, max_depth=4, samples_per_src=100000)
    a, tau = paths.cir(out_type="numpy", normalize_delays=False)
    a = a[:, 0, :, :, :, 0]; tau = tau[:, 0, :, 0, :]
    th = paths.theta_t.numpy()[:, 0, :, 0, :]; ph = paths.phi_t.numpy()[:, 0, :, 0, :]
    pw = (np.abs(a) ** 2).sum(2); idx = np.argsort(-pw, axis=-1)[..., :KEEP]; g = lambda x: np.take_along_axis(x, idx, -1)
    return dict(a=np.stack([g(a[:, :, 0]), g(a[:, :, 1])], 2).astype(np.complex64), tau=g(tau).astype(np.float64), th=g(th).astype(np.float32), ph=g(ph).astype(np.float32))
out = dict(cams=cams, meta=np.array(meta, dtype=object), dims=np.array(info["dims"]), fc=FC); t0 = time.time()
ctr = np.array([L / 2, W / 2])
#            position                 orientation
BARS = [((0.30, 0.30, 2.20), "h"), ((L - 0.30, 0.30, 1.40), "v"), ((0.30, W - 0.30, 1.40), "v"), ((3.50, W - 0.30, 2.20), "h")]
for k, (pos, ori) in enumerate(BARS):
    to_c = ctr - np.array(pos[:2]); to_c /= np.linalg.norm(to_c)
    along = np.array([-to_c[1], to_c[0], 0.0]) if ori == "h" else np.array([0.0, 0.0, 1.0])
    src = np.array(pos) + BAR[:, None] * along
    out[f"bar{k}"] = solve(src); out[f"bar{k}"]["src"] = src; out[f"bar{k}"]["bore"] = np.array([to_c[0], to_c[1], 0.0]); out[f"bar{k}"]["ori"] = ori
np.save(f"data/mbar_{BAND}_{SEED}.npy", out, allow_pickle=True); print("mbar", BAND, SEED, "done in", round(time.time() - t0), "s", flush=True)
