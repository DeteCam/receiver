"""Lodging room, four 47 cm bars (2 horizontal + 2 vertical). Usage: rt_eval_mbar.py motel BAND [BW]   (first arg must be 'motel')"""
import sys, glob, numpy as np
if len(sys.argv) > 3: sys.argv = sys.argv[:3] + [""] + sys.argv[3:]
import rt_eval as E
from rt_eval import BAND, GRID, SUBS, csi_discrete, observe, capon_ll, sph_steer, rt_model, K

def run():
    files = sorted(glob.glob(f"data/mbar_{BAND}_?.npy")); rng = np.random.default_rng(7); res = {}; typ = []
    fsel = np.arange(0, K, 5)
    combos = {"4 bars (2H+2V)": [0, 1, 2, 3], "3 bars (H+V+V)": [0, 1, 2], "2 bars (H+V)": [0, 1], "2 bars (H+H)": [0, 3]}
    for f in files:
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; bars = [D[f"bar{k}"] for k in range(4)]
        A16 = [sph_steer(b["src"]) for b in bars]; C16 = [rt_model(b["src"], fsel) for b in bars]
        ix = SUBS["bar8n"]; A8 = [sph_steer(b["src"][ix]) for b in bars]; C8 = [rt_model(b["src"][ix], fsel) for b in bars]
        for r in range(len(cams)):
            cam = cams[r]; typ.append(D["meta"][r][1]); cb, cm, cb8, cm8 = [], [], [], []
            for k, b in enumerate(bars):
                H = csi_discrete(b, r, b["bore"])
                pk, Rr, _ = observe(H, rng); cb.append(capon_ll(Rr, A16[k])); cm.append(C16[k].ll(np.array(pk)[:, :, fsel], "full"))
                pk, Rr, _ = observe(H[ix], rng); cb8.append(capon_ll(Rr, A8[k])); cm8.append(C8[k].ll(np.array(pk)[:, :, fsel], "full"))
            for name, idx in combos.items():
                for tag, sc in (("bar16|capon", cb), ("bar16|map-aware", cm), ("bar8n|capon", cb8), ("bar8n|map-aware", cm8)):
                    est = GRID[int(np.argmax(sum(sc[i] for i in idx)))]
                    res.setdefault(f"{name} {tag}", []).append(float(np.linalg.norm(est - cam)))
        print(f, "done", flush=True)
    np.save(f"res_mbar_{BAND}_{int(E.BW/1e6)}.npy", dict(res=res, typ=typ), allow_pickle=True)
    print(f"\n=== lodging room bars, {BAND} BW={E.BW/1e6:.1f} MHz n={len(typ)}  (3-D error median / P90 / <=1 m / <=0.5 m)")
    for k, v in res.items():
        v = np.array(v); print(f"  {k:38s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%")

if __name__ == "__main__":
    run()
