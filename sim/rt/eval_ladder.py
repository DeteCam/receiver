"""One-factor-at-a-time ladder (2 Nano-84 nodes at the repository positions). Usage: eval_ladder.py KIND BAND [BW_Hz]"""
import sys, glob, numpy as np, torch
if len(sys.argv) > 3: sys.argv = sys.argv[:3] + [""] + sys.argv[3:]
import rt_eval as E
from rt_eval import (KIND, BAND, FC, BFLAG, GRID, SPACE, model, axes, csi_nano, observe, fuse, capon_ll, sph_steer,
                     precompute_node_geometry, steering_for_dirs, localize_ma_pinn, extract_wideband_cov_features, NANO12_EDGE_M)

def space_for(var):
    sp = dict(SPACE)
    if var in ("shell", "shellroi"): sp["obstacles"] = []
    elif var == "furn": sp["obstacles"] = [o for o in SPACE["obstacles"] if o.mat != "body_tissue"]
    return sp

STEPS = [("shellroi", "d1x", "ideal", "0  repository-like: walls, 1st order, cameras on its test lattice"),
         ("shell", "d1x", "ideal", "1  cameras moved to realistic places (seats / surfaces)"),
         ("shell", "d4", "ideal", "2  + reflections up to 4th order, diffraction"),
         ("furn", "d4", "ideal", "3  + fixed objects (board, desks, cabinet / furniture)"),
         ("full", "d4", "ideal", "4  + people"),
         ("full", "d4", "em", "5  + full-wave board response")]

def run():
    out = {}
    for var, dtag, mode, label in STEPS:
        files = sorted(glob.glob(f"data/lad_{KIND}_{BAND}_?_{var}_{dtag}.npy")); rng = np.random.default_rng(7); sp = space_for(var)
        res = {"repo": [], "capon": []}; seat = {"repo": [], "capon": []}
        G = GRID
        if var == "shellroi" and KIND == "classroom":
            Lr, Wr, Hr = SPACE["dims"]; G = np.array([[x, y, z] for x in np.arange(0.3, Lr - 0.2, 0.3) for y in np.arange(0.3, Wr - 0.2, 0.3) for z in np.arange(0.25, 2.7, 0.3)])
        E.GRID = G
        for f in files:
            D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; meta = D["meta"]; seats = D.get("seats")
            nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]; pre = []; A = []
            for n in nodes:
                p0 = precompute_node_geometry(n, G, sp)
                p0["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], FC); p0["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], FC); pre.append(p0)
                ex, ey, ez = axes(n["bore"]); A.append(sph_steer(n["pos"] + NANO12_EDGE_M[:, 0:1] * ex + NANO12_EDGE_M[:, 2:3] * ez))
            for r in range(len(cams)):
                obs, sc = [], []
                for n, node in enumerate(nodes):
                    Hc = csi_nano(D["nano"], r, n, node, mode)
                    if np.mean(np.abs(Hc) ** 2) < 1e-30: Hc = Hc + 1e-12
                    pk, Rr, pw = observe(Hc, rng)
                    fe = [extract_wideband_cov_features(h, BFLAG) for h in pk]; feat = np.mean([x[0] for x in fe], 0); Rall = np.mean([x[1] for x in fe], 0)
                    with torch.no_grad(): Rai = model(torch.from_numpy(feat[None]), torch.tensor([BFLAG]))[0].numpy()
                    amp = np.abs(pk[0]); cv = float(np.median(np.std(amp, 1) / (np.mean(amp, 1) + 1e-12)))
                    obs.append(dict(R_all=Rall, R_ai=Rai, cv_f=cv, raw_pow=pw)); sc.append(capon_ll(Rr, A[n]))
                for key, est in (("repo", localize_ma_pinn(obs, pre, G)), ("capon", fuse(sc)[0])):
                    res[key].append(float(np.linalg.norm(est - cams[r])))
                    if KIND == "classroom" and var != "shellroi":
                        i = int(meta[r][0]); j = int(np.argmin(np.hypot(seats[:, 0] - est[0], seats[:, 1] - 0.14 - est[1])))
                        seat[key].append((int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)))
        out[label] = (res, seat)
    np.save(f"res_ladder_{KIND}_{BAND}_{int(E.BW/1e6)}.npy", out, allow_pickle=True)
    print(f"\n=== ladder {KIND} {BAND} BW={E.BW/1e6:.1f} MHz   (median / P90 / <=1 m [ / seat / 3x3 ])   repository estimator + AI  ||  plain beamforming fusion")
    for label, (res, seat) in out.items():
        s = []
        for key in ("repo", "capon"):
            v = np.array(res[key]); t = f"{np.median(v):.2f} / {np.percentile(v, 90):.2f} / {100*np.mean(v<=1):.0f}%"
            if seat[key]: st = np.array(seat[key]).mean(0) * 100; t += f" / {st[0]:.0f}% / {st[1]:.0f}%"
            s.append(t)
        print(f"  {label:58s} n={len(res['repo']):3d}  {s[0]:34s} ||  {s[1]}")

if __name__ == "__main__":
    run()
