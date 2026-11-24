"""Palm-size rotating receiver: sensitivity, in-situ calibration and feature export (map-aware estimator only).
Usage: pe2.py SCENE BAND 'FILEGLOB' TAG          SCENE = motel | classroom
Env (all optional):
  ARR=ringH45   ring radius in mm            NPOS=16   arm positions per turn        BW=17.5e6  used bandwidth (Hz)
  DIMERR=0      1-sigma error (m) of the room length/width given to the estimator (height: half of it)
  POSERR=0      1-sigma error (m) of the node position given to the estimator
  HEADERR=0     1-sigma error (deg) of the node heading given to the estimator
  ANGJIT=0      1-sigma error (deg) of the arm angle at each position
  CALDEG=4.5    per-position phase residual (deg)        RNOISE=-30  relative noise (dB)
  TVAR=f,s      a fraction f of the non-direct paths changes phase randomly between arm positions (1-sigma s deg): moving people
  NSWEEP=1      with TVAR: number of turns of the arm that are averaged (the fixed reference antenna keeps the turns phase-aligned)
  CALIB=0       number of known-position beacons used for in-situ calibration      BEACONERR=0.10 (m)      REFLCAL=0|1
  MARK=CALIB    leave this many beacon positions out of the statistics even when CALIB=0 (same test set with / without calibration)
  SEED=0        seed of the geometry errors / beacon choice
  MAPFURN=1     lodging room only: add the faces of the wardrobe, bed and TV console to the map (mapx.py); their positions get the DIMERR error too
  EM=free|box[,dz[,gap]]  use the full-wave (NEC2) response of the two-dipole front end instead of ideal dipoles: the moving dipole and the
                fixed reference dipole raised dz mm (default 65) on the axis; free standing, or with the metal body of the host gap mm
                (default 15) below the moving dipole.
                The estimator keeps the ideal model.
Saves res_pe2_TAG.npy = dict(rows=[...per camera-node sample...], pos=[...per camera...], cfg=...)."""
import sys, os, glob, time, numpy as np
from scipy.optimize import minimize
from palm import *
from mapx import MapModelX, MOTEL_FACES

SCENE, BANDN, FGLOB, TAG = sys.argv[1:5]
E = os.environ.get
RAD = int(E("ARR", "ringH45")[5:]) * 1e-3; NPOS = int(E("NPOS", "16")); BW = float(E("BW", "17.5e6"))
DIMERR, POSERR, HEADERR, ANGJIT = float(E("DIMERR", "0")), float(E("POSERR", "0")), float(E("HEADERR", "0")), float(E("ANGJIT", "0"))
CALDEG, RNOISE = float(E("CALDEG", "4.5")), float(E("RNOISE", "-30"))
TVAR = tuple(float(x) for x in E("TVAR").split(",")) if E("TVAR") else None
CALIB, BEACONERR, REFLCAL, SEED = int(E("CALIB", "0")), float(E("BEACONERR", "0.10")), int(E("REFLCAL", "0")), int(E("SEED", "0"))
FURN = int(E("MAPFURN", "0")); NSWEEP = int(E("NSWEEP", "1"))
KSUB = max(16, int(round(16 * BW / 17.5e6)))
B = Band(BANDN, bw=BW, k=min(KSUB, 64)); rng = np.random.default_rng(11); grng = np.random.default_rng(100 + SEED); brng = np.random.default_rng(500 + SEED)
MARK = int(E("MARK", str(CALIB)))                                        # beacons left out of the statistics (also without calibration)
DIMS = (8.5, 7.5, 2.8) if SCENE == "classroom" else (4.5, 3.8, 2.6); L, W, H = DIMS
PLANES = lambda d: [(0, 0.0), (0, d[0]), (1, 0.0), (1, d[1]), (2, d[2]), (2, 0.0)]
ORDER = dict(classroom=[0, 1, 2, 3], motel=[0, 3, 1, 2])                 # order in which one receiver is carried around (front corners first / diagonal first)

def grid_points():
    if SCENE == "classroom": xs = np.arange(0.3, L - 0.2, 0.3); ys = np.arange(0.3, W - 0.2, 0.3); zs = np.array([0.75, 1.0, 1.25])
    else: xs = np.arange(0.05, L, 0.2); ys = np.arange(0.05, W, 0.2); zs = np.array([0.25, 0.55, 0.85, 1.15, 1.45, 1.75, 2.05, 2.35, 2.55])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])
GRID = grid_points()

def arm_angles(extra=0.0): return 2 * np.pi * np.arange(NPOS) / NPOS + extra
def ring_pos(extra=0.0):
    a = arm_angles(extra)
    return RAD * np.stack([np.cos(a), np.sin(a), 0 * a], 1)
AXIS = np.tile([0, 0, 1.0], (NPOS, 1))

class EMPattern:
    """Full-wave far-field patterns of the reference dipole (on the axis, raised by dz mm) and the moving dipole (44 mm arm), 5-degree grid,
    from ../em/puck3.py.  spec = "free" | "box" [",dz[,gap[,flip]]"]: reference raised dz mm (default 65), metal body gap mm below the moving
    dipole (default 15), flip = the receiver hangs upside down."""
    def __init__(self, spec):
        f = 2440 if BANDN == "2g4" else 5220; q = spec.split(","); kind = q[0]; dz = q[1] if len(q) > 1 else "65"; gap = q[2] if len(q) > 2 else "15"
        self.flip = len(q) > 3 and q[3] == "flip"                             # receiver hung upside down (antennas below the body)
        long_arm = RAD > 0.07; r0 = 0.090 if long_arm else 0.044                 # the antenna model exists for a 44 mm and a 90 mm arm
        r = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "em", "puck3_r90_results.npy" if long_arm else "puck3_results.npy"),
                    allow_pickle=True).item()[(f, kind == "box", float(dz), float(gap))]
        th = np.radians(np.arange(0, 181, 5)); ph = np.radians(np.arange(0, 360, 5)); k = 2 * np.pi * f * 1e6 / C0
        nom = np.exp(1j * k * r0 * np.sin(th)[None] * np.cos(ph)[:, None]); s = np.abs(r[9001]["eth"][:, 18]).mean()
        self.mov = (r[9002]["eth"] / nom / s, r[9002]["eph"] / nom / s); self.ref = (r[9001]["eth"] / s, r[9001]["eph"] / s)   # geometric phase removed before interpolation
    @staticmethod
    def bil(F, th, ph):
        a = (np.degrees(np.asarray(ph, np.float64)) % 360) / 5.0; i0 = np.floor(a).astype(int) % 72; fa = a - np.floor(a); i1 = (i0 + 1) % 72
        b = np.clip(np.degrees(np.asarray(th, np.float64)) / 5.0, 0, 36 - 1e-6); j0 = np.floor(b).astype(int); fb = b - j0
        return (F[i0, j0] * (1 - fa) + F[i1, j0] * fa) * (1 - fb) + (F[i0, j0 + 1] * (1 - fa) + F[i1, j0 + 1] * fa) * fb
    def resp(self, aV, aH, th, ph, alpha):
        """Responses (M, P) of the moving and the reference dipole to every path when the arm is at angles alpha (M,)."""
        pl = ph[None] - alpha[:, None]; T = np.broadcast_to(th[None], pl.shape); Td, pd = (np.pi - T, -pl) if self.flip else (T, pl)   # direction in the device frame
        mov = (aV[None] * self.bil(self.mov[0], Td, pd) + aH[None] * self.bil(self.mov[1], Td, pd)) * np.exp(1j * B.k0 * RAD * np.sin(T) * np.cos(pl))
        return mov, aV[None] * self.bil(self.ref[0], Td, pd) + aH[None] * self.bil(self.ref[1], Td, pd)
EMP = EMPattern(E("EM")) if E("EM") else None

def synth_tv(paths, r, n, pos, cam, node_pos):
    aV, aH = paths["a"][r, n, 0], paths["a"][r, n, 1]; th, ph, tau = paths["th"][r, n], paths["ph"][r, n], paths["tau"][r, n]
    if EMP is not None:                                                   # what the receiver really forms: moving / reference, times the mean reference
        mov, ref = EMP.resp(aV, aH, th, ph, np.arctan2(pos[:, 1], pos[:, 0])); Fq = np.exp(-2j * np.pi * tau[:, None] * B.df[None])
        Hm, Hr = mov @ Fq, ref @ Fq
        return Hm / (Hr + 1e-30) * Hr.mean(0, keepdims=True)
    st = np.exp(1j * B.k0 * (pos @ udir(th, ph).T)) * element_gain(aV, aH, th, ph, AXIS); Fq = np.exp(-2j * np.pi * tau[:, None] * B.df[None])
    if TVAR:
        nd = np.abs(tau - np.linalg.norm(cam - node_pos) / C0) > 0.06e-9; sel = nd & (rng.random(len(tau)) < TVAR[0]); H = 0
        for _ in range(NSWEEP):                                           # several turns of the arm, averaged with the help of the fixed reference antenna
            s2 = st.copy(); s2[:, sel] = s2[:, sel] * np.exp(1j * np.radians(rng.normal(0, TVAR[1], (len(pos), int(sel.sum()))))); H = H + s2 @ Fq
        return H / NSWEEP
    return st @ Fq

def load(f):
    D = np.load(f, allow_pickle=True).item(); meta = D.get("meta"); seats = D.get("seats")
    if meta is None or (SCENE == "classroom" and seats is None):
        M0 = np.load(f.replace("x_", ""), allow_pickle=True).item(); meta, seats = M0["meta"], M0["seats"]
    return D["nano"], D["cams"], [np.array(n["pos"], float) for n in D["nodes"]], meta, seats

def ll_full(mm, X, n_eff=10.0, w_vis=0.7):
    s1, g = mm.score(X, return_gain=True); s0 = mm.score(X, use=np.arange(1, mm.S.shape[1]))
    l1 = -n_eff * np.log(1 - s1 + 2e-3); l0 = -n_eff * np.log(1 - s0 + 2e-3)
    return np.logaddexp(np.log(w_vis) + l1, np.log(1 - w_vis) + l0), s1, s0, g

def modes(ll, az_c, k=3, sep=15.0, nb=120):
    """Top-k azimuth modes of the likelihood map seen from the node (candidate index of each)."""
    b = ((np.degrees(az_c) % 360) / (360 / nb)).astype(int) % nb; best = np.full(nb, -np.inf); arg = np.full(nb, -1)
    o = np.argsort(ll); best[b[o]] = ll[o]; arg[b[o]] = o
    out = []
    for i in np.argsort(-best):
        if not np.isfinite(best[i]): break
        if all(min(abs(i - j), nb - abs(i - j)) * (360 / nb) >= sep for j, _ in out): out.append((i, arg[i]))
        if len(out) == k: break
    return [c for _, c in out]

PRIOR = np.array([np.radians(10.0), 0.15, 0.15, 0.15, 0.15])              # how wrong the hand-measured pose / room size may be (1-sigma)
def calibrate(Xb, pb, pos_e, node_e, dims_e):
    """In-situ calibration of one node with known-position beacons: heading, node x/y, room length/width."""
    def model(t):
        c, s = np.cos(t[0]), np.sin(t[0]); p = pos_e @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
        return MapModel(pb, node_e + np.array([t[1], t[2], 0]), p, AXIS, B, (dims_e[0] + t[3], dims_e[1] + t[4], dims_e[2]), second=True)
    def cost(t):
        t = np.clip(t, [-0.5, -0.6, -0.6, -0.6, -0.6], [0.5, 0.6, 0.6, 0.6, 0.6]); mm = model(t)
        return float(10.0 * sum(np.log(1 - mm.score(X)[i] + 2e-3) for i, X in enumerate(Xb)) + 0.5 * np.sum((t / PRIOR) ** 2))
    best = None
    for h0 in np.radians([0.0, -8.0, 8.0]):                                # a few heading starts: the cost is multi-modal in heading
        r = minimize(cost, [h0, 0, 0, 0, 0], method="Nelder-Mead",
                     options=dict(initial_simplex=np.array([[h0, 0, 0, 0, 0]] + [[h0 + (i == 0) * 0.05, (i == 1) * 0.08, (i == 2) * 0.08, (i == 3) * 0.08, (i == 4) * 0.08] for i in range(5)]),
                                  maxiter=160, xatol=2e-3, fatol=1e-3))
        if best is None or r.fun < best.fun: best = r
    t = np.clip(best.x, [-0.5, -0.6, -0.6, -0.6, -0.6], [0.5, 0.6, 0.6, 0.6, 0.6]); mm = model(t); refl = None
    if REFLCAL:                                                           # per-plane amplitude reflectivity from the fitted path gains
        rat = []
        for i, X in enumerate(Xb):
            s, g = mm.score(X, return_gain=True)
            if s[i] > 0.6 and abs(g[i, 0]) > 0: rat.append(np.abs(g[i, 1:7]) / abs(g[i, 0]) / np.sqrt(mm.w[i, 1:7] / 0.25))   # = rho_i (w was built with rho = 0.5)
        if len(rat) >= 3: refl = np.clip(np.median(np.array(rat), axis=0), 0.1, 0.95)
    return t, refl

def run():
    t0 = time.time(); files = sorted(glob.glob(FGLOB)); assert files, FGLOB; rows, posr, adap = [], [], []
    for fi, f in enumerate(files):
        paths, cams, nodes, meta, seats = load(f); nn = len(nodes)
        subsets = {"x2": [0, 1]} if nn == 2 else {"x4": [0, 1, 2, 3], "x3": [0, 1, 2], "x2 adj": [0, 1], "x2 diag": [0, 3]}
        dims_e = (L + grng.normal(0, DIMERR), W + grng.normal(0, DIMERR), H + grng.normal(0, DIMERR / 2))
        beacons = sorted(brng.choice(len(cams), MARK, replace=False).tolist()) if MARK else []
        pre = []
        for n in range(nn):
            pos_t = ring_pos(np.radians(grng.normal(0, ANGJIT, NPOS)) if ANGJIT else 0.0)          # where the arm really is
            node_e = nodes[n] + grng.normal(0, POSERR, 3) * np.array([1, 1, 0.5]); herr = np.radians(grng.normal(0, HEADERR))
            pos_e = ring_pos(herr); de = dims_e; refl = 0.5; cal = None
            if CALIB:
                pb = cams[beacons] + brng.normal(0, BEACONERR, (len(beacons), 3)) * np.array([1, 1, 0.5])
                Xb = [observe(synth_tv(paths, r, n, pos_t, cams[r], nodes[n]), brng, CALDEG, 0.4, RNOISE) for r in beacons]
                t, rf = calibrate(Xb, pb, pos_e, node_e, dims_e); cal = t
                c, s = np.cos(t[0]), np.sin(t[0]); pos_e = pos_e @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
                node_e = node_e + np.array([t[1], t[2], 0]); de = (dims_e[0] + t[3], dims_e[1] + t[4], dims_e[2])
                if rf is not None: refl = {p: float(v) for p, v in zip(PLANES(de), rf)}
            if FURN and SCENE == "motel":
                fc = [(ax, val + grng.normal(0, DIMERR), lo, hi, rf) for ax, val, lo, hi, rf in MOTEL_FACES]
                mm = MapModelX(GRID, node_e, pos_e, AXIS, B, de, faces=fc, refl=refl, second=True)
            else: mm = MapModel(GRID, node_e, pos_e, AXIS, B, de, refl=refl, second=True)
            v = GRID - node_e; azc = np.arctan2(v[:, 1], v[:, 0]); dc = np.linalg.norm(v, axis=1)
            Ub, _, _ = dir_grid(); mk = room_mask(Ub, np.clip(node_e, 0.06, np.array(de) - 0.06), de)        # the receiver itself is inside the room
            if not mk.any(): mk[:] = True
            Ab = steer(Ub, pos_e, AXIS, B)[mk]; Ubm = Ub[mk]
            pre.append(dict(pos_t=pos_t, node_e=node_e, mm=mm, azc=azc, dc=dc, Ab=Ab, Ubm=Ubm, herr=np.degrees(herr), cal=cal))
            if CALIB: print(f"  node {n}: heading error {np.degrees(herr):+.1f} deg -> residual {np.degrees(herr + t[0]):+.1f} deg, shifts {np.round(t[1:], 2)}", flush=True)
        for r in range(len(cams)):
            cam = cams[r]; lls = []
            for n in range(nn):
                d = pre[n]; X = observe(synth_tv(paths, r, n, d["pos_t"], cam, nodes[n]), rng, CALDEG, 0.4, RNOISE)
                ll, s1, s0, g = ll_full(d["mm"], X); lls.append(ll)
                vt = cam - nodes[n]; azt = np.arctan2(vt[1], vt[0]); _, lf = los_info(paths, r, n, cam, nodes[n])
                pb = bartlett_dir(X, d["Ab"]); ub = d["Ubm"][int(np.argmax(pb))]; azb = np.arctan2(ub[1], ub[0])
                aerr = lambda a: abs((np.degrees(a - azt) + 180) % 360 - 180)
                ms = modes(ll, d["azc"]); feats = []
                for c in ms:
                    gg = np.abs(g[c]) ** 2 * np.real(np.einsum("rr->r", d["mm"].G[c])); u = (GRID[c] - d["node_e"]) / d["dc"][c]
                    ib = int(np.argmax(d["Ubm"] @ u))
                    feats.append([ll[c] - ll.max(), s1[c], s0[c], gg[0] / (gg.sum() + 1e-30), d["dc"][c], pb[ib] / pb.max(),
                                  abs((np.degrees(d["azc"][c] - azb) + 180) % 360 - 180), aerr(d["azc"][c])])
                while len(feats) < 3: feats.append([-50, 0, 0, 0, 0, 0, 180, 180])
                Xn = X / np.sqrt(np.mean(np.abs(X) ** 2)); cb = int(np.argmax(ll)); K = X.shape[1]
                coh = lambda g: float(np.abs(np.sum(Xn[:, :-g] * Xn[:, g:].conj())) / np.sqrt(np.sum(np.abs(Xn[:, :-g]) ** 2) * np.sum(np.abs(Xn[:, g:]) ** 2)))
                xf = [coh(K // 4), coh(K // 2), float(np.std(np.abs(Xn)) / np.mean(np.abs(Xn))), float(pb.max() / np.sum(np.abs(X) ** 2)), float(s1[cb]), float(s0[cb])]
                rows.append(dict(f=fi, r=r, n=n, beacon=r in beacons, err=aerr(d["azc"][cb]), err_b=aerr(azb), lf=lf, xf=np.array(xf),   # xf: signal-only cues of a visible direct path
                                 spread=az_spread(ll, GRID, d["node_e"]), feats=np.array(feats), dist=float(np.linalg.norm(vt))))
            for sn, ix in subsets.items():
                e = GRID[int(np.argmax(sum(lls[i] for i in ix)))]; seat = [np.nan, np.nan]
                if seats is not None and SCENE == "classroom":
                    i = int(meta[r][0]); j = int(np.argmin(np.hypot(seats[:, 0] - e[0], seats[:, 1] - 0.14 - e[1])))
                    seat = [int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)]
                fe = [float(abs((np.degrees(np.arctan2(e[1] - nodes[i][1], e[0] - nodes[i][0]) - np.arctan2(cam[1] - nodes[i][1], cam[0] - nodes[i][0])) + 180) % 360 - 180)) for i in ix]
                posr.append(dict(f=fi, r=r, sub=sn, beacon=r in beacons, err=float(np.linalg.norm(e - cam)), seat=seat, fuse_az=fe))
            if nn == 4:                                                   # one receiver carried around: add its spots one at a time
                order = ORDER[SCENE]; acc = np.zeros(len(GRID)); seq = []; n0 = nodes[order[0]]
                for i in order:
                    acc = acc + lls[i]; p = np.exp(acc - acc.max()); p /= p.sum(); mu = p @ GRID; e = GRID[int(np.argmax(acc))]
                    aze = abs((np.degrees(np.arctan2(e[1] - n0[1], e[0] - n0[0]) - np.arctan2(cam[1] - n0[1], cam[0] - n0[0])) + 180) % 360 - 180)
                    seq.append((float(np.sqrt(p @ np.sum((GRID - mu) ** 2, axis=1))), float(aze), float(np.linalg.norm(e - cam))))
                adap.append(dict(f=fi, r=r, beacon=r in beacons, seq=seq))
        print(os.path.basename(f), "done", round(time.time() - t0), "s", flush=True)
    cfg = {k: E(k) for k in ("ARR", "EM", "MAPFURN", "NSWEEP", "NPOS", "BW", "DIMERR", "POSERR", "HEADERR", "ANGJIT", "CALDEG", "RNOISE", "TVAR", "CALIB", "MARK", "BEACONERR", "REFLCAL", "SEED") if E(k)}
    np.save(f"res_pe2_{TAG}.npy", dict(rows=rows, pos=posr, adap=adap, cfg=cfg, scene=SCENE, band=BANDN), allow_pickle=True)
    summary(rows, posr, cfg, adap)

def summary(rows, posr, cfg, adap=()):
    use = [x for x in rows if not x["beacon"]]; e = np.array([x["err"] for x in use]); sp = np.array([x["spread"] for x in use]); keep = sp <= np.percentile(sp, 80)
    print(f"\n=== {TAG} {SCENE} {BANDN} {cfg}")
    print(f"  one node, one spot   n={len(e):4d}  mean {e.mean():5.1f}  median {np.median(e):5.1f}  P90 {np.percentile(e, 90):5.1f}  P95 {np.percentile(e, 95):5.1f}  <=10deg {100 * np.mean(e <= 10):3.0f}%  | most confident 80%: mean {e[keep].mean():5.1f}  P95 {np.percentile(e[keep], 95):5.1f}")
    for sn in dict.fromkeys(x["sub"] for x in posr):
        P = [x for x in posr if x["sub"] == sn and not x["beacon"]]; pe_ = np.array([x["err"] for x in P]); fa = np.concatenate([x["fuse_az"] for x in P])
        s = f"  fused {sn:8s}      n={len(P):4d}  direction: mean {fa.mean():5.1f}  median {np.median(fa):5.1f}  P90 {np.percentile(fa, 90):5.1f}  P95 {np.percentile(fa, 95):5.1f}  | position: median {np.median(pe_):.2f} m  <=1 m {100 * np.mean(pe_ <= 1):3.0f}%  <=0.5 m {100 * np.mean(pe_ <= 0.5):3.0f}%"
        st = np.array([x["seat"] for x in P], float)
        if np.isfinite(st).all(): s += f"  seat {100 * st[:, 0].mean():3.0f}%  3x3 {100 * st[:, 1].mean():3.0f}%"
        print(s)
    A = [x for x in adap if not x["beacon"]]
    if A:
        S = np.array([x["seq"] for x in A])                                  # (N, 4 spots, [spread m, direction error seen from the first spot, position error])
        for k in range(4): print(f"  always {k + 1} spot(s)            direction from the first spot: mean {S[:, k, 1].mean():5.1f}  median {np.median(S[:, k, 1]):5.1f}  P90 {np.percentile(S[:, k, 1], 90):5.1f}  P95 {np.percentile(S[:, k, 1], 95):5.1f}  | position <=1 m {100 * np.mean(S[:, k, 2] <= 1):3.0f}%")

if __name__ == "__main__":
    run()
