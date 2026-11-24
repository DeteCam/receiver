"""Sanity: pure line-of-sight (no multipath) through the same synthesis + repository estimator. Errors should be ~grid size."""
import sys, json, numpy as np, torch
sys.argv = ["x", sys.argv[1], sys.argv[2]]
import rt_eval as E
from rt_eval import *
nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in json.load(open("nano_nodes.json"))["classroom_40" if KIND == "classroom" else "motel_std"]]
rng = np.random.default_rng(3); L_, W_, H_ = SPACE["dims"]
cams = np.stack([rng.uniform(0.5, L_ - 0.5, 60), rng.uniform(0.5, W_ - 0.5, 60), rng.uniform(0.6, 1.6, 60)], 1)
P = 1; R = len(cams); S = len(nodes)
d = dict(a=np.zeros((R, S, 2, P), complex), tau=np.zeros((R, S, P)), th=np.zeros((R, S, P)), ph=np.zeros((R, S, P)))
for r, c in enumerate(cams):
    for s, n in enumerate(nodes):
        v = c - n["pos"]; dist = np.linalg.norm(v); u = v / dist
        d["a"][r, s, 0, 0] = 1e-3 / dist; d["tau"][r, s, 0] = dist / C0; d["th"][r, s, 0] = np.arccos(u[2]); d["ph"][r, s, 0] = np.arctan2(u[1], u[0])
pre = []
for n in nodes:
    p0 = precompute_node_geometry(n, GRID, SPACE); p0["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], FC); p0["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], FC); pre.append(p0)
nano_A = [sph_steer(n["pos"] + NANO12_EDGE_M[:, 0:1] * axes(n["bore"])[0] + NANO12_EDGE_M[:, 2:3] * axes(n["bore"])[2]) for n in nodes]
e1, e2, e3 = [], [], []
for r in range(R):
    obs = []; sc = []
    for s, n in enumerate(nodes):
        pk, Rr, pw = observe(csi_nano(d, r, s, n, "ideal"), rng)
        fe = [extract_wideband_cov_features(h, BFLAG) for h in pk]; feat = np.mean([x[0] for x in fe], 0); Rall = np.mean([x[1] for x in fe], 0)
        with torch.no_grad(): Rai = model(torch.from_numpy(feat[None]), torch.tensor([BFLAG]))[0].numpy()
        amp = np.abs(pk[0]); cv = float(np.median(np.std(amp, 1) / (np.mean(amp, 1) + 1e-12)))
        obs.append(dict(R_all=Rall, R_ai=Rai, cv_f=cv, raw_pow=pw)); sc.append(capon_ll(Rr, nano_A[s]))
    e1.append(np.linalg.norm(localize_ma_pinn(obs, pre, GRID) - cams[r])); e2.append(np.linalg.norm(localize_ma_pinn([dict(o, R_ai=o["R_all"]) for o in obs], pre, GRID) - cams[r])); e3.append(np.linalg.norm(fuse(sc)[0] - cams[r]))
for nm, e in (("repo MA-PINN+AI", e1), ("repo MA-PINN no AI", e2), ("capon fusion", e3)):
    e = np.array(e); print(f"{KIND} {BAND} LoS-only  {nm:20s} median {np.median(e):.2f}  P90 {np.percentile(e, 90):.2f}  <=1m {100*np.mean(e<=1):.0f}%")
