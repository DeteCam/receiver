"""Evaluate DeteCam array/placement options on Sionna-RT path data. Usage: rt_eval.py KIND BAND [suffix]"""
import sys, json, glob, numpy as np, torch
import os
HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.environ.get("DETECAM_V6", os.path.join(HERE, "..", "..", "..", "receiver-v6"))   # checkout of DeteCam/receiver at tag v6-nano2
EMDIR = os.path.join(HERE, "..", "em")
sys.path.insert(0, os.path.join(V6, "sim")); sys.path.insert(0, EMDIR)
from raytrace_engine import build_10_diverse_spaces, NANO12_EDGE_M, C0
from benchmark_10spaces_ai import precompute_node_geometry, steering_for_dirs, localize_ma_pinn
from train_ai_models import ReconCovNet, extract_wideband_cov_features
from em_manifold import Manifold, fov_grid
from mapcache import Cache

KIND, BAND = sys.argv[1], sys.argv[2]; SUF = sys.argv[3] if len(sys.argv) > 3 else ""
BW = float(sys.argv[4]) if len(sys.argv) > 4 else 17.5e6
FC = 2.437e9 if BAND == "2g4" else 5.22e9; BFLAG = 0.0 if BAND == "2g4" else 1.0
K = 32; DF = np.linspace(-BW / 2, BW / 2, K); LAM = C0 / FC; KW = 2 * np.pi / LAM
SPACE = build_10_diverse_spaces(rng_seed=2026)["classroom_40" if KIND == "classroom" else "motel_std"]
L, W, H = SPACE["dims"]
EMRES = np.load(os.path.join(EMDIR, "em_results.npy"), allow_pickle=True).item()
MAN = Manifold(EMRES["sh_2g4"][2440] if BAND == "2g4" else EMRES["sh_5g"][5220], FC)
FRAME = np.load("frame12.npy"); FRAME3 = np.stack([FRAME[:, 0], np.zeros(12), FRAME[:, 1]], 1)
model = ReconCovNet(n_ant=12, in_ch=6, hidden=192, rank=3)
model.load_state_dict(torch.load(os.path.join(V6, "firmware", "core", "weights", "recon_covnet.pt"), map_location="cpu", weights_only=True)); model.eval()

def axes(bore):
    ey = np.asarray(bore, float); ey = ey / np.linalg.norm(ey); ex = np.cross(ey, [0, 0, 1.0]); ex /= np.linalg.norm(ex); return ex, ey, np.cross(ex, ey)
def udir(th, ph): return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
def front(u, bore):
    off = np.degrees(np.arccos(np.clip(u @ bore, -1, 1))); return 10 ** (-np.minimum(12 * (off / 65.0) ** 2, 30) / 20)

def grid_points():
    if KIND == "classroom": xs = np.arange(0.3, L - 0.2, 0.3); ys = np.arange(0.3, W - 0.2, 0.3); zs = np.array([0.75, 1.0, 1.25])
    else: xs = np.arange(0.05, L, 0.2); ys = np.arange(0.05, W, 0.2); zs = np.array([0.25, 0.55, 0.85, 1.15, 1.45, 1.75, 2.05, 2.35, 2.55])
    pts = [[x, y, z] for x in xs for y in ys for z in zs if not any(ob.b[0] <= x <= ob.b[1] and ob.b[2] <= y <= ob.b[3] and ob.b[4] <= z <= ob.b[5] and ob.mat in ("concrete", "steel_pec", "wet_tile", "partition") for ob in SPACE["obstacles"])]
    return np.array(pts)
GRID = grid_points()
# sub-arrays of the 16-element 47 cm bar: 8 non-uniform elements (3 SP4T modules) and a 4-chain no-switch array (router with 4 coherent chains, 12 packets)
SUBS = {"bar8n": np.array([0, 1, 3, 6, 9, 12, 14, 15]), "chain4": (np.array([0, 2, 8, 12]) if BAND == "2g4" else np.array([0, 1, 4, 6]))}

def csi_discrete(d, r, bore):
    """V-pol co-polarised discrete elements with a directive element pattern. -> H (S, K)"""
    u = udir(d["th"][r], d["ph"][r]); w = front(u, bore)
    return np.einsum("sp,spk->sk", d["a"][r, :, 0, :] * w, np.exp(-2j * np.pi * d["tau"][r][..., None] * DF))

def csi_nano(d, r, n, node, mode):
    """Nano-84 board at node n from single-point dual-pol path data. mode 'ideal' = repository assumption, 'em' = full-wave board."""
    ex, ey, ez = axes(node["bore"]); th, ph = d["th"][r, n], d["ph"][r, n]; u = udir(th, ph)
    ul = np.stack([u @ ex, u @ ey, u @ ez], -1); aV, aH = d["a"][r, n, 0], d["a"][r, n, 1]
    ph_f = np.exp(-2j * np.pi * d["tau"][r, n][:, None] * DF)                                    # (P, K)
    if mode == "ideal":
        st = np.exp(1j * KW * (NANO12_EDGE_M @ ul.T))                                           # (12, P)
        return (st * (aV * front(u, ey))[None]) @ ph_f
    tg = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1); pg = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
    thl = np.arccos(np.clip(ul[:, 2], -1, 1)); phl = np.arctan2(ul[:, 1], ul[:, 0])
    tl = np.stack([np.cos(thl) * np.cos(phl), np.cos(thl) * np.sin(phl), -np.sin(thl)], -1); pl = np.stack([-np.sin(phl), np.cos(phl), np.zeros_like(phl)], -1)
    R = np.stack([ex, ey, ez], 1)                                                                # local -> global
    tlg, plg = tl @ R.T, pl @ R.T
    E = aV[:, None] * tg + aH[:, None] * pg
    pol = np.stack([np.sum(E * tlg, 1), np.sum(E * plg, 1)], 1)
    resp = MAN.resp(ul, pol)[:12] * (ul[:, 1] > 0.0)[None]                                       # rear hemisphere zeroed (generous)
    return resp @ ph_f

EM_NORM = None; CUR = {}
def observe(Hclean, rng, n_pkt=4, cal_deg=4.5, cal_db=0.4):
    """Calibration residual + noise; returns list of packet CSI and covariance."""
    M = Hclean.shape[0]
    c = 10 ** (rng.normal(0, cal_db, M) / 20) * np.exp(1j * np.radians(rng.normal(0, cal_deg, M)))
    p = np.mean(np.abs(Hclean) ** 2); sig2 = max(10 ** (-109 / 10), p * 1e-3)
    pk = [c[:, None] * Hclean + np.sqrt(sig2 / 2) * (rng.standard_normal(Hclean.shape) + 1j * rng.standard_normal(Hclean.shape)) for _ in range(n_pkt)]
    R = sum(h @ h.conj().T for h in pk) / (n_pkt * K)
    return pk, R, p

def fuse(scores):
    tot = sum(scores); return GRID[int(np.argmax(tot))], tot

def bartlett_ll(R, A):
    """A (C, M) unit-norm steering -> log normalised Bartlett."""
    p = np.real(np.einsum("cm,mk,ck->c", A.conj(), R, A)); return np.log(p / p.max() + 1e-6)
def capon_ll(R, A, load=0.08):
    M = R.shape[0]; Ri = np.linalg.inv(R + load * np.real(np.trace(R)) / M * np.eye(M))
    p = 1.0 / np.maximum(np.real(np.einsum("cm,mk,ck->c", A.conj(), Ri, A)), 1e-12); return np.log(p / p.max() + 1e-6)
def sph_steer(E):
    d = np.linalg.norm(GRID[:, None, :] - E[None], axis=2); A = np.exp(-1j * KW * d); return A / np.sqrt(E.shape[0])

def rt_model(E, fsel):
    """Map-aware model (direct + 4 walls + ceiling images), exact spherical steering. Returns Cache."""
    kf = 2 * np.pi * (FC + DF[fsel]) / C0; C = len(GRID); M = E.shape[0]
    A = np.zeros((M, C, 6, len(fsel)), np.complex64); P = np.zeros((C, 6))
    imgs = [GRID.copy()]
    for ax, val in ((0, 0.0), (0, L), (1, 0.0), (1, W), (2, H)):
        g = GRID.copy(); g[:, ax] = 2 * val - g[:, ax]; imgs.append(g)
    for k, g in enumerate(imgs):
        d = np.linalg.norm(g[:, None, :] - E[None], axis=2)                                      # (C, M)
        A[:, :, k, :] = np.exp(-1j * d.T[:, :, None] * kf[None, None, :]); P[:, k] = (1.0 if k == 0 else 0.25) / np.mean(d, 1) ** 2
    return Cache(A, P)

def bearing_err(R, st_fn, u_true_local, Us):
    A = st_fn(Us); A = A / np.linalg.norm(A, axis=1, keepdims=True)
    p = np.real(np.einsum("cm,mk,ck->c", A.conj(), R, A)); uh = Us[int(np.argmax(p))]
    return float(np.degrees(np.arccos(np.clip(uh @ u_true_local, -1, 1))))

def run():
    files = sorted(glob.glob(f"data/{KIND}_{BAND}_?{SUF}.npy")); rng = np.random.default_rng(7)
    res = {}; ang = {}; meta = []
    seat = {}
    def add(name, est, cam):
        res.setdefault(name, []).append(float(np.linalg.norm(est - cam)))
        if KIND == "classroom":
            j = int(np.argmin(np.hypot(CUR["seats"][:, 0] - est[0], CUR["seats"][:, 1] - 0.14 - est[1]))); i = CUR["i"]
            seat.setdefault(name, []).append((int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)))
    Us, _, _ = fov_grid(75, 60, 2.0)
    fsel = np.arange(0, K, 5)
    # EM dual-pol calibrated manifold on the plane-wave search grid (for bearing) is built per node below
    for f in files:
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
        pre = []
        for n in nodes:
            p0 = precompute_node_geometry(n, GRID, SPACE)
            p0["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], FC); p0["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], FC); pre.append(p0)
        # dual-pol EM manifold for every grid point (direction from node, local frame)
        em_B = []
        for n in nodes:
            ex, ey, ez = axes(n["bore"]); v = GRID - n["pos"]; v /= np.linalg.norm(v, axis=1, keepdims=True); ul = np.stack([v @ ex, v @ ey, v @ ez], 1)
            one = np.ones(len(ul)); zero = np.zeros(len(ul))
            em_B.append((MAN.resp(ul, np.stack([one, zero], 1).astype(complex))[:12], MAN.resp(ul, np.stack([zero, one], 1).astype(complex))[:12], ul[:, 1] > 0.05))
        frm_E = [n["pos"] + FRAME[:, 0:1] * axes(n["bore"])[0] + FRAME[:, 1:2] * axes(n["bore"])[2] for n in nodes]
        frm_A = [sph_steer(E) for E in frm_E]; frm_C = [rt_model(E, fsel) for E in frm_E]
        nano_E = [n["pos"] + NANO12_EDGE_M[:, 0:1] * axes(n["bore"])[0] + NANO12_EDGE_M[:, 2:3] * axes(n["bore"])[2] for n in nodes]
        nano_A = [sph_steer(E) for E in nano_E]
        bars = [k for k in ("bar0", "bar1", "bar2", "bar3") if k in D]
        bar_A = [sph_steer(D[b]["src"]) for b in bars]; bar_C = [rt_model(D[b]["src"], fsel) for b in bars]
        sub_A = {sn: [sph_steer(D[b]["src"][ix]) for b in bars] for sn, ix in SUBS.items()}; sub_C = {sn: [rt_model(D[b]["src"][ix], fsel) for b in bars] for sn, ix in SUBS.items()}
        for r in range(len(cams)):
            cam = cams[r]; meta.append(D["meta"][r][1]); CUR["seats"] = D.get("seats"); CUR["i"] = int(D["meta"][r][0]) if KIND == "classroom" else 0
            # ---------- Nano-84 (repository hardware), 2 nodes
            for mode in ("ideal", "em"):
                obs = []; sc_b = []; sc_emcal = []
                for n, node in enumerate(nodes):
                    Hc = csi_nano(D["nano"], r, n, node, mode)
                    if np.mean(np.abs(Hc) ** 2) < 1e-30: Hc = Hc + 1e-12
                    pk, Rr, pw = observe(Hc, rng)
                    feat = np.mean([extract_wideband_cov_features(h, BFLAG)[0] for h in pk], axis=0); Rall = np.mean([extract_wideband_cov_features(h, BFLAG)[1] for h in pk], axis=0)
                    with torch.no_grad(): Rai = model(torch.from_numpy(feat[None]), torch.tensor([BFLAG]))[0].numpy()
                    amp = np.abs(pk[0]); cv = float(np.median(np.std(amp, 1) / (np.mean(amp, 1) + 1e-12)))
                    obs.append(dict(R_all=Rall, R_ai=Rai, cv_f=cv, raw_pow=pw)); sc_b.append(capon_ll(Rr, nano_A[n]))
                    v = cam - node["pos"]; ul = np.array([v @ a for a in axes(node["bore"])]); ul /= np.linalg.norm(ul)
                    ang.setdefault(f"nano_{mode}", []).append(bearing_err(Rr, lambda U: np.exp(1j * KW * (U @ NANO12_EDGE_M.T)), ul, Us))
                    if mode == "em":
                        Bt, Bp, okf = em_B[n]
                        y1 = np.einsum("mc,mk->ck", Bt.conj(), Rr); y2 = np.einsum("mc,mk->ck", Bp.conj(), Rr)
                        a_ = np.sum(np.abs(Bt) ** 2, 0); b_ = np.sum(Bt.conj() * Bp, 0); d_ = np.sum(np.abs(Bp) ** 2, 0); det = np.maximum(a_ * d_ - np.abs(b_) ** 2, 1e-9 * a_ * d_ + 1e-30)
                        q11 = np.real(np.sum(y1 * Bt.T, 1)); q22 = np.real(np.sum(y2 * Bp.T, 1)); q12 = np.sum(y1 * Bp.T, 1)
                        pwr = (d_ * q11 - 2 * np.real(b_ * q12.conj()) + a_ * q22) / det
                        pwr = np.where(okf, np.maximum(pwr, 1e-30), 1e-30); sc_emcal.append(np.log(pwr / pwr.max() + 1e-6))
                add(f"nano2_{mode}|repo MA-PINN+AI", localize_ma_pinn(obs, pre, GRID), cam)
                o2 = [dict(o, R_ai=o["R_all"]) for o in obs]; add(f"nano2_{mode}|repo MA-PINN (no AI)", localize_ma_pinn(o2, pre, GRID), cam)
                add(f"nano2_{mode}|capon fusion", fuse(sc_b)[0], cam)
                if mode == "em": add("nano2_em|full dual-pol chamber cal", fuse(sc_emcal)[0], cam)
            # ---------- frame12 (module build, same 2 node positions)
            sb, sr = [], []
            for n, node in enumerate(nodes):
                pk, Rr, pw = observe(csi_discrete(D[f"frm{n}"], r, node["bore"]), rng)
                sb.append(capon_ll(Rr, frm_A[n])); sr.append(frm_C[n].ll(np.array(pk)[:, :, fsel], "full"))
                v = cam - node["pos"]; ul = np.array([v @ a for a in axes(node["bore"])]); ul /= np.linalg.norm(ul)
                ang.setdefault("frame12", []).append(bearing_err(Rr, lambda U: np.exp(1j * KW * (U @ FRAME3.T)), ul, Us))
            add("frame12 x2|capon fusion", fuse(sb)[0], cam); add("frame12 x2|map-aware likelihood", fuse(sr)[0], cam)
            # ---------- bar16 at 4 corners (classroom)
            if bars:
                sb, sr = [], []
                for k, b in enumerate(bars):
                    pk, Rr, pw = observe(csi_discrete(D[b], r, D[b]["bore"]), rng)
                    sb.append(capon_ll(Rr, bar_A[k])); sr.append(bar_C[k].ll(np.array(pk)[:, :, fsel], "full"))
                    al = D[b]["src"][15] - D[b]["src"][0]; al = al / np.linalg.norm(al); ctr = D[b]["src"].mean(0); v = cam - ctr; sin_t = float(v @ al / np.linalg.norm(v))
                    sg = np.linspace(-1, 1, 721); st = np.exp(1j * KW * np.outer(sg, (np.arange(16) - 7.5) * 0.03125)) / 4.0
                    pb = np.real(np.einsum("cm,mk,ck->c", st.conj(), Rr, st)); ang.setdefault("bar16(cone)", []).append(abs(float(np.degrees(np.arcsin(sg[int(np.argmax(pb))]) - np.arcsin(sin_t)))))
                add("bar16 x4|capon fusion", fuse(sb)[0], cam); add("bar16 x4|map-aware likelihood", fuse(sr)[0], cam)
                add("bar16 x2(diag)|map-aware likelihood", fuse([sr[0], sr[3]])[0], cam); add("bar16 x2(diag)|capon fusion", fuse([sb[0], sb[3]])[0], cam)
                for sname, idx in SUBS.items():
                    s1, s2 = [], []
                    for k, b in enumerate(bars):
                        Hc = csi_discrete(D[b], r, D[b]["bore"])[idx]; pk, Rr, pw = observe(Hc, rng, n_pkt=(4 if sname.startswith("bar") else 12))
                        s1.append(capon_ll(Rr, sub_A[sname][k])); s2.append(sub_C[sname][k].ll(np.array(pk)[:, :, fsel], "full"))
                        if sname.startswith("bar16") is False and k == 0:
                            pass
                    add(f"{sname} x4|capon fusion", fuse(s1)[0], cam); add(f"{sname} x4|map-aware likelihood", fuse(s2)[0], cam)
        print(f, "done", flush=True)
    np.save(f"res_{KIND}_{BAND}{SUF}_{int(BW/1e6)}.npy", dict(res=res, ang=ang, meta=meta, seat=seat), allow_pickle=True)
    print(f"\n=== {KIND} {BAND} BW={BW/1e6:.1f} MHz  n={len(meta)}  (3-D error: median / P90 / within 1 m / within 0.5 m)")
    for k, v in res.items():
        v = np.array(v); st = np.array(seat[k]).mean(0) * 100 if k in seat else (np.nan, np.nan)
        print(f"  {k:42s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%   seat {st[0]:4.0f}%  3x3 {st[1]:4.0f}%")
    print("  single-node bearing error (deg) median / P90:")
    for k, v in ang.items():
        v = np.array(v); print(f"    {k:14s} {np.median(v):5.1f}  {np.percentile(v, 90):5.1f}")

if __name__ == "__main__":
    run()
