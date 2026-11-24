"""Check the single-point model used for the Nano-84 board: ideal 12-element board at the repository's two node positions,
full per-element ray tracing (nf_*.npy) vs single-point path data + plane-wave model (the model used in rt_eval.py)."""
import sys, glob, numpy as np, torch
import rt_eval as E
from rt_eval import *
rng = np.random.default_rng(7); out = {}
for f in sorted(glob.glob(f"data/nf_{KIND}_{BAND}_?.npy")):
    Df = np.load(f, allow_pickle=True).item(); D = np.load(f.replace("nf_", ""), allow_pickle=True).item(); cams = D["cams"]
    assert np.allclose(cams, Df["cams"]); nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
    pre = []
    for n in nodes:
        p0 = precompute_node_geometry(n, GRID, SPACE); p0["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], FC); p0["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], FC); pre.append(p0)
    A = [sph_steer(Df[f"frm{n}"]["src"]) for n in range(2)]
    for r in range(len(cams)):
        for tag in ("full ray tracing", "single-point model"):
            obs, sc = [], []
            for n, node in enumerate(nodes):
                Hc = csi_discrete(Df[f"frm{n}"], r, node["bore"]) if tag[0] == "f" else csi_nano(D["nano"], r, n, node, "ideal")
                pk, Rr, pw = observe(Hc, rng); fe = [extract_wideband_cov_features(h, BFLAG) for h in pk]; feat = np.mean([x[0] for x in fe], 0); Rall = np.mean([x[1] for x in fe], 0)
                with torch.no_grad(): Rai = model(torch.from_numpy(feat[None]), torch.tensor([BFLAG]))[0].numpy()
                amp = np.abs(pk[0]); cv = float(np.median(np.std(amp, 1) / (np.mean(amp, 1) + 1e-12)))
                obs.append(dict(R_all=Rall, R_ai=Rai, cv_f=cv, raw_pow=pw)); sc.append(capon_ll(Rr, A[n]))
            out.setdefault(tag + " | repo MA-PINN+AI", []).append(np.linalg.norm(localize_ma_pinn(obs, pre, GRID) - cams[r]))
            out.setdefault(tag + " | capon", []).append(np.linalg.norm(fuse(sc)[0] - cams[r]))
for k, e in out.items():
    e = np.array(e); print(f"{KIND} {BAND} nano2 ideal | {k:38s} {np.median(e):.2f} m  {np.percentile(e, 90):.2f} m  {100*np.mean(e<=1):.1f}%  {100*np.mean(e<=0.5):.1f}%  n={len(e)}")
