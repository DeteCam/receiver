"""Check the plane-wave approximation: 12-element frame at the repository's two node positions,
full per-element ray tracing (frm0/frm1) vs single-point path data + plane-wave model (nano)."""
import sys, glob, numpy as np
import rt_eval as E
from rt_eval import KIND, BAND, GRID, csi_discrete, observe, capon_ll, sph_steer
from rt_eval_frame import F12, csi_planar
from arm import arm_steer
rng = np.random.default_rng(7); et, ea = [], []
for f in sorted(glob.glob(f"data/{KIND}_{BAND}_?.npy")):
    D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
    At = [sph_steer(D[f"frm{n}"]["src"]) for n in range(2)]; Aa = [arm_steer(n, F12) for n in nodes]
    for r in range(len(cams)):
        st, sa = [], []
        for n, node in enumerate(nodes):
            pk, Rr, _ = observe(csi_discrete(D[f"frm{n}"], r, node["bore"]), rng); st.append(capon_ll(Rr, At[n]))
            pk, Rr, _ = observe(csi_planar(D["nano"], r, n, node, F12), rng); sa.append(capon_ll(Rr, Aa[n]))
        et.append(np.linalg.norm(GRID[int(np.argmax(sum(st)))] - cams[r])); ea.append(np.linalg.norm(GRID[int(np.argmax(sum(sa)))] - cams[r]))
for nm, e in (("full ray tracing", et), ("plane-wave approx", ea)):
    e = np.array(e); print(f"{KIND} {BAND} frame12 x2 capon | {nm:18s} {np.median(e):.2f} m  {np.percentile(e, 90):.2f} m  {100*np.mean(e<=1):.1f}%  {100*np.mean(e<=0.5):.1f}%  n={len(e)}")
