"""2-D arrays at the four corners, from single-point path data with a per-path plane-wave model (approximation).
Usage: rt_eval_frame.py classroom|motel BAND"""
import sys, glob, numpy as np
import rt_eval as E
from rt_eval import KIND, BAND, GRID, KW, DF, K, FRAME, SUBS, axes, udir, front, observe, capon_ll, rt_model
from arm import arm_local, arm_steer

def maxmin_subset(P, m):
    idx = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
    while len(idx) < m:
        d = np.min(np.linalg.norm(P[:, None] - P[idx][None], axis=2), axis=1); d[idx] = -1; idx.append(int(np.argmax(d)))
    return np.array(sorted(idx))

BAR = (np.arange(16) - 7.5) * 0.03125; B8 = BAR[SUBS["bar8n"]]; z0 = np.zeros
F12 = np.stack([FRAME[:, 0], z0(12), FRAME[:, 1]], 1); F8 = F12[maxmin_subset(F12, 8)]
ARR = {"frame12 (20 cm, 12 el.)": (F12, 4), "frame8 (20 cm, 8 el.)": (F8, 4), "bar16 horizontal (47 cm)": (np.stack([BAR, z0(16), z0(16)], 1), 4),
       "L 8+8 (47 cm arms)": (np.concatenate([np.stack([B8, z0(8), z0(8)], 1), np.stack([z0(8) + BAR[0], z0(8), B8 - BAR[0] + 0.03125], 1)]), 4),
       "arm 40 cm circle": (arm_local(), 2)}

def csi_planar(d, r, n, node, P):
    ex, ey, ez = axes(node["bore"]); u = udir(d["th"][r, n], d["ph"][r, n]); ul = np.stack([u @ ex, u @ ey, u @ ez], -1)
    return (np.exp(1j * KW * (P @ ul.T)) * (d["a"][r, n, 0] * front(u, ey))[None]) @ np.exp(-2j * np.pi * d["tau"][r, n][:, None] * DF)

def run():
    files = sorted(glob.glob(f"data/x_classroom_{BAND}_?.npy" if KIND == "classroom" else f"data/xm_{BAND}_?.npy")); rng = np.random.default_rng(7); res = {}; seat = {}
    fsel = np.arange(0, K, 5); sets = {"x4": [0, 1, 2, 3], "x2 (diagonal)": [0, 3]}
    for f in files:
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
        if KIND == "classroom": M0 = np.load(f.replace("x_", ""), allow_pickle=True).item(); seats = M0["seats"]; meta = M0["meta"]
        St = {a: [arm_steer(n, P) for n in nodes] for a, (P, _) in ARR.items()}
        Cm = {a: [rt_model(n["pos"] + ARR[a][0][:, 0:1] * axes(n["bore"])[0] + ARR[a][0][:, 2:3] * axes(n["bore"])[2], fsel) for n in nodes] for a in ("frame12 (20 cm, 12 el.)", "frame8 (20 cm, 8 el.)")}
        for r in range(len(cams)):
            cam = cams[r]
            def add(name, sc):
                for sn, ix in sets.items():
                    est = GRID[int(np.argmax(sum(sc[i] for i in ix)))]; key = f"{name} {sn}"
                    res.setdefault(key, []).append(float(np.linalg.norm(est - cam)))
                    if KIND == "classroom":
                        i = int(meta[r][0]); j = int(np.argmin(np.hypot(seats[:, 0] - est[0], seats[:, 1] - 0.14 - est[1])))
                        seat.setdefault(key, []).append((int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)))
            for a, (P, npk) in ARR.items():
                sc, sm = [], []
                for n, node in enumerate(nodes):
                    pk, Rr, _ = observe(csi_planar(D["nano"], r, n, node, P), rng, n_pkt=npk); sc.append(capon_ll(Rr, St[a][n]))
                    if a in Cm: sm.append(Cm[a][n].ll(np.array(pk)[:, :, fsel], "full"))
                add(f"{a}|capon", sc)
                if sm: add(f"{a}|map-aware", sm)
        print(f, "done", flush=True)
    np.save(f"res_frame_{KIND}_{BAND}.npy", dict(res=res, seat=seat), allow_pickle=True)
    print(f"\n=== 2-D arrays at 4 corners (plane-wave approx.), {KIND} {BAND} n={len(next(iter(res.values())))}  (median / P90 / <=1 m / <=0.5 m)")
    for k, v in res.items():
        v = np.array(v); st = np.array(seat[k]).mean(0) * 100 if k in seat else (np.nan, np.nan)
        print(f"  {k:46s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%   seat {st[0]:4.0f}%  3x3 {st[1]:4.0f}%")

if __name__ == "__main__":
    run()
