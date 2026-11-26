"""Legacy rotating-plate receiver (receiver-legacy sim/rt, 'plate' = tilt18 + whole-curve matching against the
device reference curves, polarisation fitted) run on the same Sionna wall-scene paths as the ring
(plan sim/plan_20261126_plate.md). Single-receiver azimuth error, read as the azimuth of the best point of the
same room candidate grid the ring uses; the legacy 6 m direction-sphere reading is stored too.
Usage: python3 mysim/plate_eval.py <seeds> [node labels, default m0,m2] [scenes]"""
import os, sys, pickle, time, zlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

LEG = "/opt/detecam/receiver-legacy"
sys.path.insert(0, os.path.join(LEG, "sim", "rt"))
import rt_eval as R                       # noqa: E402  (legacy: Pattern, TM, RTWorld, PLANS, sph, rotz, SUB)
from fusion import origin                 # noqa: E402
import locate as K                        # noqa: E402
sys.path.insert(0, "/opt/detecam/sim/v1_ring")
from estimators import make_room_grid      # noqa: E402

C0 = 299792458.0
OUTSIDE = ["lodging_a", "restroom_a", "locker_a", "bathhouse_a", "bathhouse_b", "lodging_b", "restroom_b",
           "locker_b", "worst_metal", "lodging_c", "office_a", "restroom_c"]
SPHERE = os.environ.get("PLATE_SPHERE", "0") == "1"
LAB = ["c0", "c1", "c2", "c3", "m0", "m1", "m2", "m3"]


def grid_dirs(step=3.0):
    az = np.radians(np.arange(0, 360, step)); el = np.radians(np.arange(-66, 31, step)); A, L = np.meshgrid(az, el)
    return np.stack([np.cos(L) * np.cos(A), np.cos(L) * np.sin(A), np.sin(L)], -1).reshape(-1, 3)


def crc(*a):
    return zlib.crc32(repr(a).encode()) & 0xFFFFFFFF


class WallWorld(R.RTWorld):
    """RTWorld fed from our path arrays: a[2 pol (theta, phi) at the receiver], tau, th/ph (direction to the source)."""
    def __init__(self, rng, paths_rn, node, cam, fc, frames, pat, rate=100.0, snr=30.0):
        aV, aH, tau, th, ph = paths_rn
        pw = np.abs(aV) ** 2 + np.abs(aH) ** 2; ok = pw > 1e-24
        aV, aH, tau, th, ph, pw = aV[ok], aH[ok], tau[ok].astype(float), th[ok].astype(float), ph[ok].astype(float), pw[ok]
        self.tau = tau; self.body = np.zeros(len(tau), bool)
        self.d, self.th_hat, self.ph_hat = R.sph(th, ph)
        self.cj = np.stack([aV, aH], -1).astype(complex)
        self.fk = float(fc) + R.SUB
        self.phase = np.exp(-2j * np.pi * self.tau[:, None] * self.fk[None, :])
        self.rng = rng; self.pat = pat; self.rate = rate; self.tv = 0.0; self.snr = snr
        self.src = np.asarray(node, float); self.cam = np.asarray(cam, float); self.head = float(rng.uniform(0, 360))
        self.o = origin(frames); self.base = self.src - R.rotz(self.o, self.head)
        v = self.cam - self.src; self.c_true = v / np.linalg.norm(v)
        isl = np.abs(tau - np.linalg.norm(v) / C0) < 0.08e-9; self.los = bool(isl.any())
        self.k_db = float(10 * np.log10(pw[isl].sum() / max(pw.sum(), 1e-30) + 1e-9)) if self.los else -99.0


def az_err(p, node, cam):
    a = np.arctan2(p[1] - node[1], p[0] - node[0]); t = np.arctan2(cam[1] - node[1], cam[0] - node[0])
    return float(abs((np.degrees(a - t) + 180.0) % 360.0 - 180.0))


def run_file(path, labels, pat, tm, G, plan="tilt18"):
    D = np.load(path, allow_pickle=True).item()
    frames = R.PLANS[plan]; P = D["paths"]; cams = D["cams"]; nodes = D["nodes"]; dims = np.asarray(D["dims"], float)
    grid = make_room_grid(dims, max_xy_pts=260)
    idx = [D["node_labels"].index(l + "_A") for l in labels]
    n_cam = len(cams)
    out = dict(az_grid=np.zeros((n_cam, len(idx)), np.float32), az_sph=np.zeros((n_cam, len(idx)), np.float32),
               los=np.zeros((n_cam, len(idx)), bool), kdb=np.zeros((n_cam, len(idx)), np.float32))
    for r in range(n_cam):
        for j, n in enumerate(idx):
            rng = np.random.default_rng(crc("plate", D["scene_id"], D["band"], int(D["seed"]), r, n))
            pr = (P["a"][r, n, 0], P["a"][r, n, 1], P["tau"][r, n], P["th"][r, n], P["ph"][r, n])
            w = WallWorld(rng, pr, nodes[n], cams[r], D["fc"], frames, pat)
            pc = [w.scan(y, t)[1] for y, t in frames]
            pose = (w.base + rng.normal(0, 0.05, 3) * np.array([1, 1, 0.5]), w.head + rng.normal(0, 3.0))
            _, b = tm.score(K.to_dev(grid, pose), [pc], frames)
            out["az_grid"][r, j] = az_err(grid[int(np.argmax(b[0]))], nodes[n], cams[r])
            if SPHERE:
                Xg = nodes[n] + 6.0 * G
                _, bg = tm.score(K.to_dev(Xg, pose), [pc], frames)
                out["az_sph"][r, j] = az_err(Xg[int(np.argmax(bg[0]))], nodes[n], cams[r])
            else:
                out["az_sph"][r, j] = np.nan
            out["los"][r, j] = w.los; out["kdb"][r, j] = w.k_db
    meta = dict(scene=D["scene_id"], band=D["band"], seed=int(D["seed"]), labels=labels, cam_types=list(D["cam_types"]))
    return meta, out


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else ["m0", "m2"]
    scenes = sys.argv[3].split(",") if len(sys.argv) > 3 else OUTSIDE
    pat = R.Pattern(); tm = R.TM(pat, 160.0); G = grid_dirs()
    outp = os.environ.get("PLATE_OUT", "/opt/detecam/sim/results/exp_bearing_plate.pkl")
    res = pickle.load(open(outp, "rb")) if os.path.exists(outp) else {}
    t0 = time.time()
    for s in seeds:
        for sc in scenes:
            key = (sc, "2g4", s, tuple(labels))
            if key in res:
                continue
            meta, o = run_file(f"/opt/detecam/sim/data_wall/nop/{sc}_2g4_s{s}.npy", labels, pat, tm, G)
            res[key] = (meta, o)
            tmp = outp + ".tmp"; pickle.dump(res, open(tmp, "wb")); os.replace(tmp, outp)
            print(f"[{time.time() - t0:7.0f}s] {sc:12s} s{s} {labels} | grid mean {o['az_grid'].mean():.1f} med {np.median(o['az_grid']):.1f}", flush=True)
    print("PLATE_DONE", seeds, labels, flush=True)


if __name__ == "__main__":
    main()
