"""Legacy plate moved freely (plan sim/plan_20261126_plate.md section 6): score maps at the wall middles m0..m3
(data_wall/nop) and five free-standing spots f0..f4 (data_wall/free_nop), then fused per pre-set spot plan.
One placement per spot measures every camera (one pose per spot), so the reference-curve templates of the room
grid are computed once per spot and frame.
Usage: python3 mysim/plate_pos.py <seeds> [scenes]"""
import os, sys, pickle, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import plate_eval as PE

R, K = PE.R, PE.K
SPOTS = [("nop", "m0"), ("nop", "m1"), ("nop", "m2"), ("nop", "m3"),
         ("free_nop", "f0"), ("free_nop", "f1"), ("free_nop", "f2"), ("free_nop", "f3"), ("free_nop", "f4")]
PLANS = {"F2": ["f0", "f2"], "F3": ["f0", "f2", "f4"], "F5": ["f0", "f1", "f2", "f3", "f4"],
         "F9": ["f0", "f1", "f2", "f3", "f4", "m0", "m1", "m2", "m3"], "W2": ["m0", "m2"]}


def templates(pat, Xdev, frames):
    out = []
    for yaw, tilt in frames:
        d, th_hat, ph_hat = R.sph_v(Xdev - R.antenna_pos(yaw, tilt))
        g = pat.world(yaw, R.frame(yaw, tilt), d, th_hat, ph_hat)
        g = 0.5 * (g + np.roll(g, -1, 0))
        T = [10 * np.log10(np.abs(a * g[:, :, 0] + b * g[:, :, 1]) ** 2 + 1e-30) for a, b in R.POLS]
        out.append(np.stack(T))                                   # [pol, 72, N]
    return out


def score(Ts, curves):
    sp = 0.0
    for T, (_, lv) in zip(Ts, curves):
        ok = np.isfinite(lv)
        if ok.sum() < 36:
            continue
        m = lv[ok] - lv[ok].mean()
        Tk = T[:, ok, :]; Tc = Tk - Tk.mean(1, keepdims=True)                     # [pol, B, N]
        al = np.clip(np.einsum("pbn,b->pn", Tc, m) / np.maximum((Tc ** 2).sum(1), 1e-9), 0.0, 1.2)
        gain = (m ** 2).sum() - ((m[None, :, None] - al[:, None, :] * Tc) ** 2).sum(1)
        sp = sp + gain
    return np.max(sp, 0) / 160.0                                  # best polarisation, same scaling as legacy TM


def run_scene(sc, seed, pat, frames):
    files = {v: np.load(f"/opt/detecam/sim/data_wall/{v}/{sc}_2g4_s{seed}.npy", allow_pickle=True).item()
             for v in ("nop", "free_nop")}
    Dn = files["nop"]; cams = Dn["cams"]; dims = np.asarray(Dn["dims"], float)
    assert np.allclose(files["free_nop"]["cams"], cams)
    grid = PE.make_room_grid(dims, max_xy_pts=260)
    n_cam = len(cams); maps = {}; az = {}; spot_xyz = {}
    for v, lab in SPOTS:
        D = files[v]; n = D["node_labels"].index(lab + "_A"); P = D["paths"]; node = D["nodes"][n]; spot_xyz[lab] = node
        rs = np.random.default_rng(PE.crc("platepos", sc, seed, lab))
        head = float(rs.uniform(0, 360)); o = PE.origin(frames); base = node - R.rotz(o, head)
        pose = (base + rs.normal(0, 0.05, 3) * np.array([1, 1, 0.5]), head + rs.normal(0, 3.0))
        Ts = templates(pat, K.to_dev(grid, pose), frames)
        M = np.zeros((n_cam, len(grid)), np.float32); A = np.zeros(n_cam, np.float32)
        for r in range(n_cam):
            rng = np.random.default_rng(PE.crc("platepos", sc, seed, lab, r))
            pr = (P["a"][r, n, 0], P["a"][r, n, 1], P["tau"][r, n], P["th"][r, n], P["ph"][r, n])
            w = PE.WallWorld(rng, pr, node, cams[r], D["fc"], frames, pat)
            w.head = head; w.base = base                           # the device stands still for every camera
            curves = []
            for y, t in frames:
                tt, C, rec = w.packets(y, t)
                curves.append(R.power_curve(tt, C, rec))
            s = score(Ts, curves); M[r] = s; A[r] = PE.az_err(grid[int(np.argmax(s))], node, cams[r])
        maps[lab] = M; az[lab] = A
    pos = {}
    for pn, labs in PLANS.items():
        lt = sum(maps[l] - maps[l].max(1, keepdims=True) for l in labs)
        pt = grid[np.argmax(lt, 1)]
        pos[pn] = np.linalg.norm(pt[:, :2] - cams[:, :2], axis=1).astype(np.float32)
    meta = dict(scene=sc, seed=seed, cam_types=list(Dn["cam_types"]), spots=spot_xyz)
    return meta, dict(az=az, pos=pos)


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    scenes = sys.argv[2].split(",") if len(sys.argv) > 2 else PE.OUTSIDE
    pat = R.Pattern(); frames = R.PLANS["tilt18"]
    outp = os.environ.get("PLATE_OUT", "/opt/detecam/sim/results/exp_plate_pos.pkl")
    res = pickle.load(open(outp, "rb")) if os.path.exists(outp) else {}
    t0 = time.time()
    for s in seeds:
        for sc in scenes:
            if (sc, s) in res:
                continue
            res[(sc, s)] = run_scene(sc, s, pat, frames)
            tmp = outp + ".tmp"; pickle.dump(res, open(tmp, "wb")); os.replace(tmp, outp)
            print(f"[{time.time() - t0:7.0f}s] {sc:12s} s{s}", flush=True)
    print("PLATEPOS_DONE", seeds, flush=True)


if __name__ == "__main__":
    main()
