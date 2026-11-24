"""Palm-size receiver evaluation on single-point ray-traced path data.
Usage: pe.py SCENE BAND 'FILEGLOB' TAG [ARRAYS] [ESTIMATORS]
  SCENE = motel | classroom ; ARRAYS e.g. ringH45,ringH90,ringT45,plane84 ; ESTIMATORS e.g. bart,nomp,map1,map2
Env: REFL (json dict of per-plane reflectivity for map estimators), NPKT, CALDEG."""
import sys, os, glob, time, json, numpy as np
from palm import *

SCENE, BANDN, FGLOB, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ARRS = (sys.argv[5] if len(sys.argv) > 5 else "ringH45").split(","); ESTS = (sys.argv[6] if len(sys.argv) > 6 else "bart,nomp,map1").split(",")
B = Band(BANDN); rng = np.random.default_rng(11)
DIMS = (8.5, 7.5, 2.8) if SCENE == "classroom" else (4.5, 3.8, 2.6); L, W, H = DIMS
NANO = np.array([[-0.0252, 0.039], [-0.0045, 0.039], [0.0238, 0.039], [-0.0225, -0.039], [0.0008, -0.039], [0.0218, -0.039],
                 [-0.039, -0.0265], [-0.039, -0.0058], [-0.039, 0.0212], [0.039, -0.0272], [0.039, -0.003], [0.039, 0.0268]])
REFL = 0.5
if os.environ.get("REFL"):
    rd = json.loads(os.environ["REFL"]); REFL = {(0, 0.0): rd["x0"], (0, L): rd["xL"], (1, 0.0): rd["y0"], (1, W): rd["yW"], (2, H): rd["zH"], (2, 0.0): rd["z0"]}

def grid_points():
    if SCENE == "classroom": xs = np.arange(0.3, L - 0.2, 0.3); ys = np.arange(0.3, W - 0.2, 0.3); zs = np.array([0.75, 1.0, 1.25])
    else: xs = np.arange(0.05, L, 0.2); ys = np.arange(0.05, W, 0.2); zs = np.array([0.25, 0.55, 0.85, 1.15, 1.45, 1.75, 2.05, 2.35, 2.55])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])
GRID = grid_points()

def make_array(name, node):
    bore = np.array(node["bore"], float); bh = np.array([bore[0], bore[1], 0.0]); bh /= np.linalg.norm(bh)
    if name == "plane84":
        ex = np.cross(bh, [0, 0, 1.0]); return NANO[:, 0:1] * ex + NANO[:, 1:2] * np.array([0, 0, 1.0]), np.tile([0, 0, 1.0], (12, 1)), bh
    R = int(name[5:]) * 1e-3; n = 16 if R <= 0.06 else (24 if R <= 0.1 else 32)
    if name.startswith("ringH"): p, a = ring(R, n); return p, a, None
    if name.startswith("ringT"): p, a = ring(R, n, tilt_normal(bh, 30.0)); return p, a, None
    raise ValueError(name)

def load(f):
    D = np.load(f, allow_pickle=True).item(); meta = D.get("meta"); seats = D.get("seats")
    if meta is None or (SCENE == "classroom" and seats is None):
        M0 = np.load(f.replace("x_", ""), allow_pickle=True).item(); meta, seats = M0["meta"], M0["seats"]
    return D["nano"], D["cams"], [dict(pos=np.array(n["pos"], float), bore=np.array(n["bore"], float)) for n in D["nodes"]], meta, seats

def run():
    U, AZ, EL = dir_grid(); t0 = time.time(); files = sorted(glob.glob(FGLOB)); assert files, FGLOB
    dirres, posres = {}, {}
    for f in files:
        paths, cams, nodes, meta, seats = load(f); nn = len(nodes)
        subsets = {"x2": [0, 1]} if nn == 2 else {"x4": [0, 1, 2, 3], "x3": [0, 1, 2], "x2 adj": [0, 1], "x2 diag": [0, 3]}
        pre = {}
        for an in ARRS:
            for n, node in enumerate(nodes):
                pos, axis, front = make_array(an, node); mask = room_mask(U, node["pos"], DIMS)
                if front is not None: mask &= (U @ front) > 0.05
                A = steer(U, pos, axis, B); d = dict(pos=pos, axis=axis, mask=mask, A=A[mask], Um=U[mask], Ag=sph_steer_el(GRID, node["pos"], pos, axis, B))
                if "nomp" in ESTS: d["nomp"] = Nomp(pos, axis, B, U)
                for e in ESTS:
                    if e.startswith("map"): d[e] = MapModel(GRID, node["pos"], pos, axis, B, DIMS, refl=REFL, second=e.startswith("map2"))
                pre[(an, n)] = d
        for r in range(len(cams)):
            cam = cams[r]
            for an in ARRS:
                maps = {}
                for n, node in enumerate(nodes):
                    d = pre[(an, n)]; X = observe(synth(paths, r, n, d["pos"], d["axis"], B), rng)
                    v = cam - node["pos"]; ut = v / np.linalg.norm(v); _, lf = los_info(paths, r, n, cam, node["pos"])
                    def rec(est, u, conf=np.nan): dirres.setdefault((an, est), []).append((az_err(u, ut), abs(abs(np.degrees(np.arcsin(u[2]))) - abs(np.degrees(np.arcsin(ut[2])))), lf, conf))
                    if "bart" in ESTS:
                        rec("bart", d["Um"][int(np.argmax(bartlett_dir(X, d["A"])))]); maps.setdefault("bart", []).append(capon_ll(X, d["Ag"]))
                    if "nomp" in ESTS:
                        pl, resid, S, gam = d["nomp"].run(X, L=3)
                        ok = [p for p in pl if room_mask(p["u"][None], node["pos"], DIMS)[0]] or pl
                        i0 = pick_direct(ok); rec("nomp", ok[i0]["u"])
                    for e in ESTS:
                        if not e.startswith("map"): continue
                        ll = d[e].ll(X); maps.setdefault(e, []).append(ll)
                        g = GRID[int(np.argmax(ll))] - node["pos"]; rec(e, g / np.linalg.norm(g), az_spread(ll, GRID, node["pos"]))
                for est, ms in maps.items():
                    for sn, ix in subsets.items():
                        e = GRID[int(np.argmax(sum(ms[i] for i in ix)))]; row = [float(np.linalg.norm(e - cam)), np.nan, np.nan]
                        if seats is not None and SCENE == "classroom":
                            i = int(meta[r][0]); j = int(np.argmin(np.hypot(seats[:, 0] - e[0], seats[:, 1] - 0.14 - e[1])))
                            row[1:] = [int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)]
                        posres.setdefault((an, est, sn), []).append(row)
                        for i in ix:                                               # direction from each node after fusion
                            g = e - nodes[i]["pos"]; v = cam - nodes[i]["pos"]
                            dirres.setdefault((an, f"{est}+fuse {sn}"), []).append((az_err(g, v), 0.0, 1.0, np.nan))
        print(os.path.basename(f), "done", round(time.time() - t0), "s", flush=True)
    np.save(f"res_pe_{TAG}.npy", dict(dir=dirres, pos=posres), allow_pickle=True)
    print(f"\n=== {TAG}: {SCENE} {BANDN}  single-node azimuth error: mean / median / P90 / <=10deg | LoS-dominant: share, mean / median | most confident 80%: mean | elev. median")
    for k, v in dirres.items():
        v = np.array(v); lo = v[:, 2] >= 0.2; c = v[:, 3]; sel = ""
        if np.isfinite(c).all(): keep = c <= np.percentile(c, 80); sel = f"{v[keep, 0].mean():5.1f}"
        print(f"  {k[0]:8s} {k[1]:18s} n={len(v):4d}  {v[:,0].mean():5.1f} / {np.median(v[:,0]):5.1f} / {np.percentile(v[:,0],90):5.1f} / {100*np.mean(v[:,0]<=10):3.0f}% | {100*lo.mean():3.0f}%, {v[lo,0].mean():5.1f} / {np.median(v[lo,0]):5.1f} | {sel:>5s} | {np.median(v[:,1]):4.1f}")
    print(f"=== {TAG}: position error: median / P90 / <=1 m / <=0.5 m [/ seat / 3x3]")
    for k, v in posres.items():
        v = np.array(v); s = f"{np.median(v[:,0]):.2f} / {np.percentile(v[:,0],90):.2f} / {100*np.mean(v[:,0]<=1):.0f}% / {100*np.mean(v[:,0]<=0.5):.0f}%"
        if np.isfinite(v[:, 1]).all(): s += f" / {100*v[:,1].mean():.0f}% / {100*v[:,2].mean():.0f}%"
        print(f"  {k[0]:8s} {k[1]:6s} {k[2]:8s} n={len(v):4d}  {s}")

if __name__ == "__main__":
    run()
