"""12-element 20 cm frame (and an 8-element subset) at four corners, full per-element ray tracing. Usage: rt_eval_frame4.py classroom|motel BAND [BW]"""
import sys, glob, numpy as np
if len(sys.argv) > 3: sys.argv = sys.argv[:3] + [""] + sys.argv[3:]
import rt_eval as E
from rt_eval import KIND, BAND, GRID, K, csi_discrete, observe, capon_ll, sph_steer, rt_model
from rt_eval_frame import maxmin_subset, F12

def run():
    files = sorted(glob.glob(f"data/f4_{KIND}_{BAND}_?.npy")); rng = np.random.default_rng(7); res = {}; seat = {}; typ = []
    fsel = np.arange(0, K, 5); i8 = maxmin_subset(F12, 8)
    sets = {"x4": [0, 1, 2, 3], "x3": [0, 1, 2], "x2 (diagonal)": [0, 3]} if KIND == "classroom" else {"x4": [0, 1, 2, 3], "x3": [0, 1, 2], "x2 (0,1)": [0, 1], "x2 (diagonal)": [0, 3]}
    for f in files:
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = D["nodes"]; seats = D.get("seats"); meta = D["meta"]
        A = {"frame12": [sph_steer(D[f"frm{k}"]["src"]) for k in range(4)], "frame8": [sph_steer(D[f"frm{k}"]["src"][i8]) for k in range(4)]}
        C = {"frame12": [rt_model(D[f"frm{k}"]["src"], fsel) for k in range(4)], "frame8": [rt_model(D[f"frm{k}"]["src"][i8], fsel) for k in range(4)]}
        for r in range(len(cams)):
            cam = cams[r]; typ.append(meta[r][1])
            for name, ix in (("frame12", np.arange(12)), ("frame8", i8)):
                sc, sm = [], []
                for k in range(4):
                    pk, Rr, _ = observe(csi_discrete(D[f"frm{k}"], r, np.array(nodes[k]["bore"]))[ix], rng)
                    sc.append(capon_ll(Rr, A[name][k])); sm.append(C[name][k].ll(np.array(pk)[:, :, fsel], "full"))
                for tag, s in (("capon", sc), ("map-aware", sm)):
                    for sn, ii in sets.items():
                        est = GRID[int(np.argmax(sum(s[i] for i in ii)))]; key = f"{name} {sn}|{tag}"
                        res.setdefault(key, []).append(float(np.linalg.norm(est - cam)))
                        if KIND == "classroom":
                            i = int(meta[r][0]); j = int(np.argmin(np.hypot(seats[:, 0] - est[0], seats[:, 1] - 0.14 - est[1])))
                            seat.setdefault(key, []).append((int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)))
        print(f, "done", flush=True)
    np.save(f"res_f4_{KIND}_{BAND}_{int(E.BW/1e6)}.npy", dict(res=res, seat=seat, typ=typ), allow_pickle=True)
    print(f"\n=== frames at 4 corners (full ray tracing), {KIND} {BAND} BW={E.BW/1e6:.1f} MHz n={len(typ)}  (median / P90 / <=1 m / <=0.5 m)")
    for k, v in sorted(res.items()):
        v = np.array(v); st = np.array(seat[k]).mean(0) * 100 if k in seat else (np.nan, np.nan)
        print(f"  {k:34s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%   seat {st[0]:4.0f}%  3x3 {st[1]:4.0f}%")
    if KIND == "classroom":
        typ_a = np.array(typ)
        for k in ("frame12 x4|map-aware", "frame8 x4|map-aware", "frame8 x4|capon"):
            v = np.array(res[k]); s = np.array(seat[k]); print("  by type", k, " | ".join(f"{t}: <=1m {100*np.mean(v[typ_a==t]<=1):.0f}% seat {100*s[typ_a==t,0].mean():.0f}% 3x3 {100*s[typ_a==t,1].mean():.0f}%" for t in ("chest", "glasses", "desk")))

if __name__ == "__main__":
    run()
