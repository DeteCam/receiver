"""Rotating arm (40 cm circle) at the repository's two node positions. Usage: rt_eval_arm.py KIND BAND"""
import sys, glob, numpy as np
import rt_eval as E
from rt_eval import KIND, BAND, GRID, KW, axes, observe, capon_ll, fuse, fov_grid, NANO12_EDGE_M, csi_nano, sph_steer
from arm import arm_local, csi_arm, arm_steer
def run():
    rng = np.random.default_rng(7); res = {}; ang = {}; ARM = arm_local(); Us, _, _ = fov_grid(75, 60, 2.0)
    for f in sorted(glob.glob(f"data/{KIND}_{BAND}_?.npy")):
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
        A = [arm_steer(n, ARM) for n in nodes]
        for r in range(len(cams)):
            sa = []
            for n, node in enumerate(nodes):
                pk, Rr, pw = observe(csi_arm(D["nano"], r, n, node, ARM), rng, n_pkt=2); sa.append(capon_ll(Rr, A[n]))
                v = cams[r] - node["pos"]; ul = np.array([v @ a for a in axes(node["bore"])]); ul /= np.linalg.norm(ul)
                St = np.exp(1j * KW * (Us @ ARM.T)) / 6.0; p = np.real(np.einsum("cm,mk,ck->c", St.conj(), Rr, St))
                ang.setdefault("arm40", []).append(float(np.degrees(np.arccos(np.clip(Us[int(np.argmax(p))] @ ul, -1, 1)))))
            res.setdefault("arm40cm x2|capon fusion (approx.)", []).append(float(np.linalg.norm(fuse(sa)[0] - cams[r])))
    for k, v in res.items():
        v = np.array(v); print(f"{KIND} {BAND} {k:36s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%  n={len(v)}")
    v = np.array(ang["arm40"]); print(f"   single-node bearing error: mean {v.mean():.1f}  median {np.median(v):.1f}  P90 {np.percentile(v, 90):.1f} deg")
if __name__ == "__main__":
    run()
