"""One-factor-at-a-time ladder: same cameras, environment changed step by step. Only the two repository node positions (single point each).
Usage: rt_gen_ladder.py KIND BAND SEED VAR DEPTH DIFFR      VAR = shell | furn | full ; DEPTH = max reflections ; DIFFR = 0/1"""
import os, sys, json, time, numpy as np, mitsuba as mi, sionna.rt as rt
import rtscene
from rtscene import classroom, motel, MOTEL_FURN
KIND, BAND, SEED, VAR, DEPTH, DIFFR = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6] == "1"
FC = 2.437e9 if BAND == "2g4" else 5.22e9; KEEP = 70
NODES = json.load(open("nano_nodes.json"))["classroom_40" if KIND == "classroom" else "motel_std"]
srng = np.random.default_rng(1000 + SEED); crng = np.random.default_rng(5000 + SEED)
furn = VAR in ("furn", "full"); ppl = VAR == "full"
ROI = VAR == "shellroi"
if KIND == "classroom":
    scene, info = classroom(FC, srng, people=ppl, furn=furn); L, W, H = info["dims"]; cams, meta = [], []
    for i, (sx, sy) in enumerate(info["seats"]):
        typ = crng.choice(["chest", "chest", "glasses", "desk"]); lean = info["leans"][i]
        p = dict(chest=[sx + crng.uniform(-0.05, 0.05), sy - 0.11 - lean - 0.03, 1.0], glasses=[sx, sy - 0.20 - lean - 0.02, 1.24],
                 desk=[sx + crng.uniform(-0.2, 0.2), sy - 0.40 + crng.uniform(-0.1, 0.1), 0.77])[typ]
        cams.append(p); meta.append((i, typ))
else:
    scene, info = motel(FC, srng, person=ppl, furn=furn); L, W, H = info["dims"]; cams, meta = [], []
    boxes = [(np.array(lo, float), np.array(hi, float), mat, tag) for lo, hi, mat, tag in MOTEL_FURN(L, W)]
    inside = lambda p: any(np.all(p >= lo - 0.02) and np.all(p <= hi + 0.02) for lo, hi, mat, tag in boxes)
    while len(cams) < 40:
        u = crng.uniform()
        if u < 0.5:
            w = crng.integers(4); z = crng.uniform(0.2, 2.3)
            p = [[0.04, crng.uniform(0.1, W - 0.1), z], [L - 0.04, crng.uniform(0.1, W - 0.1), z], [crng.uniform(0.1, L - 0.1), 0.04, z], [crng.uniform(0.1, L - 0.1), W - 0.04, z]][w]; typ = "wall"
        elif u < 0.65:
            p = [crng.uniform(0.3, L - 0.3), crng.uniform(0.3, W - 0.3), H - 0.05]; typ = "ceiling"
        else:
            lo, hi, mat, tag = boxes[crng.choice([i for i, b in enumerate(boxes) if b[3] in ("bed", "wardrobe", "tvconsole", "nightstand")])]
            p = [crng.uniform(lo[0] + 0.05, hi[0] - 0.05), crng.uniform(lo[1] + 0.05, hi[1] - 0.05), hi[2] + 0.04]; typ = "furniture:" + tag
        p = np.array(p, float)
        if inside(p) or p[2] > H - 0.03: continue
        cams.append(p); meta.append((len(cams), typ))
if ROI:                                                     # cameras on the repository's own test lattice (room interior, incl. mid-air points)
    sys.path.insert(0, os.path.join(os.environ.get("DETECAM_V6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "receiver-v6")), "sim")); from raytrace_engine import build_10_diverse_spaces
    roi = np.array(build_10_diverse_spaces(rng_seed=2026)["classroom_40" if KIND == "classroom" else "motel_std"]["roi"], float)
    cams = roi[crng.choice(len(roi), 40, replace=len(roi) < 40)]; meta = [(0, "roi")] * 40
cams = np.array(cams, float)
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
for i, p in enumerate(cams):
    scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*[float(v) for v in crng.uniform(0, np.pi, 3)])))
for j, n in enumerate(NODES): scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, n["pos"]))))
t0 = time.time()
paths = rt.PathSolver()(scene, synthetic_array=False, refraction=True, diffraction=DIFFR, edge_diffraction=DIFFR, max_depth=DEPTH, samples_per_src=200000)
a, tau = paths.cir(out_type="numpy", normalize_delays=False); a = a[:, 0, :, :, :, 0]; tau = tau[:, 0, :, 0, :]
th = paths.theta_t.numpy()[:, 0, :, 0, :]; ph = paths.phi_t.numpy()[:, 0, :, 0, :]
pw = (np.abs(a) ** 2).sum(2); idx = np.argsort(-pw, axis=-1)[..., :KEEP]; g = lambda x: np.take_along_axis(x, idx, -1)
out = dict(cams=cams, meta=np.array(meta, dtype=object), nodes=NODES, seats=info.get("seats"), var=VAR, depth=DEPTH, diffr=DIFFR,
           nano=dict(a=np.stack([g(a[:, :, 0]), g(a[:, :, 1])], 2).astype(np.complex64), tau=g(tau).astype(np.float64), th=g(th).astype(np.float32), ph=g(ph).astype(np.float32)))
np.save(f"data/lad_{KIND}_{BAND}_{SEED}_{VAR}_d{DEPTH}{'' if DIFFR else 'x'}.npy", out, allow_pickle=True)
print("lad", KIND, BAND, SEED, VAR, DEPTH, DIFFR, "done in", round(time.time() - t0), "s", flush=True)
