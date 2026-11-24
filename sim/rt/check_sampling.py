"""How much do the ray-traced per-element channels depend on the number of shot rays and on the random seed?
Classroom 2.4 GHz, seed 0, 12-element frame at one corner, 12 cameras."""
import sys, time, numpy as np, mitsuba as mi, sionna.rt as rt
from rtscene import classroom
FC = 2.437e9; FRAME = np.load("frame12.npy"); K = 32; DF = np.linspace(-8.75e6, 8.75e6, K)
def axes(bore):
    ey = np.asarray(bore, float); ey = ey / np.linalg.norm(ey); ex = np.cross(ey, [0, 0, 1.0]); ex /= np.linalg.norm(ex); return ex, ey, np.cross(ex, ey)
rng = np.random.default_rng(1000); scene, info = classroom(FC, rng); L, W, H = info["dims"]; cams = []
for i, (sx, sy) in enumerate(info["seats"]):
    typ = rng.choice(["chest", "chest", "glasses", "desk"]); lean = info["leans"][i]
    cams.append(dict(chest=[sx + rng.uniform(-0.05, 0.05), sy - 0.11 - lean - 0.03, 1.0], glasses=[sx, sy - 0.20 - lean - 0.02, 1.24], desk=[sx + rng.uniform(-0.2, 0.2), sy - 0.40 + rng.uniform(-0.1, 0.1), 0.77])[typ])
cams = np.array(cams, float)[[0, 3, 7, 9, 12, 14, 18, 21, 25, 30, 34, 39]]
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V"); scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
for i, p in enumerate(cams): scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*[float(v) for v in rng.uniform(0, np.pi, 3)])))
ctr = np.array([L / 2, W / 2]); to_c = ctr - 0.4; to_c /= np.linalg.norm(to_c); bore = [to_c[0] * np.cos(np.radians(15)), to_c[1] * np.cos(np.radians(15)), -np.sin(np.radians(15))]
ex, ey, ez = axes(bore); src = np.array([0.4, 0.4, 2.35]) + FRAME[:, 0:1] * ex + FRAME[:, 1:2] * ez
PS = rt.PathSolver()
def csi(seed, nsamp, chunk):
    out = []
    for c0 in range(0, 12, chunk):
        for n in list(scene.transmitters.keys()): scene.remove(n)
        for j, p in enumerate(src[c0:c0 + chunk]): scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, p))))
        paths = PS(scene, synthetic_array=False, refraction=True, diffraction=True, edge_diffraction=True, max_depth=4, samples_per_src=nsamp, seed=seed)
        a, tau = paths.cir(out_type="numpy", normalize_delays=False); a = a[:, 0, :, 0, :, 0]; tau = tau[:, 0, :, 0, :]      # (R, S, P) V-pol
        out.append(np.einsum("rsp,rspk->rsk", a, np.exp(-2j * np.pi * tau[..., None] * DF)))
    return np.concatenate(out, 1)                                                                                          # (R, 12, K)
def cmp(A, B):
    nm = np.sum(np.abs(A - B) ** 2, axis=(1, 2)) / np.sum(np.abs(A) ** 2, axis=(1, 2))
    co = np.abs(np.sum(A.conj() * B, axis=1)) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30)            # per camera, per subcarrier
    return 10 * np.log10(np.median(nm) + 1e-12), 10 * np.log10(np.max(nm) + 1e-12), float(np.median(co)), float(np.min(co.mean(1)))
runs = {}
for name, seed, ns, ch in (("100k s42", 42, 100000, 6), ("100k s7", 7, 100000, 6), ("400k s42", 42, 400000, 2), ("400k s7", 7, 400000, 2)):
    t0 = time.time(); runs[name] = csi(seed, ns, ch); print(name, "done", round(time.time() - t0), "s", flush=True)
for a, b in (("100k s42", "100k s7"), ("400k s42", "400k s7"), ("100k s42", "400k s42")):
    r = cmp(runs[a], runs[b]); print(f"{a:9s} vs {b:9s}: channel difference median {r[0]:6.1f} dB, worst camera {r[1]:6.1f} dB | array-vector correlation median {r[2]:.4f}, worst camera {r[3]:.4f}")
