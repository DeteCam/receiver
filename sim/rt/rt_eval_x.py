"""Nano-84 board at the 4 classroom corners (x_*.npy from rt_gen_extra.py). Usage: rt_eval_x.py classroom BAND"""
import sys, glob, numpy as np, torch
import rt_eval as E
from arm import arm_local, csi_arm, arm_steer
from rt_eval import (KIND, BAND, FC, BFLAG, KW, GRID, SPACE, MAN, model, axes, csi_nano, observe, fuse, capon_ll, sph_steer,
                     precompute_node_geometry, steering_for_dirs, localize_ma_pinn, extract_wideband_cov_features, NANO12_EDGE_M)

def emcal_ll(Rr, Bt, Bp, okf):
    y1 = np.einsum("mc,mk->ck", Bt.conj(), Rr); y2 = np.einsum("mc,mk->ck", Bp.conj(), Rr)
    a_ = np.sum(np.abs(Bt) ** 2, 0); b_ = np.sum(Bt.conj() * Bp, 0); d_ = np.sum(np.abs(Bp) ** 2, 0); det = np.maximum(a_ * d_ - np.abs(b_) ** 2, 1e-9 * a_ * d_ + 1e-30)
    q11 = np.real(np.sum(y1 * Bt.T, 1)); q22 = np.real(np.sum(y2 * Bp.T, 1)); q12 = np.sum(y1 * Bp.T, 1)
    pwr = (d_ * q11 - 2 * np.real(b_ * q12.conj()) + a_ * q22) / det
    pwr = np.where(okf, np.maximum(pwr, 1e-30), 1e-30); return np.log(pwr / pwr.max() + 1e-6)

def run():
    files = sorted(glob.glob(f"data/x_{KIND}_{BAND}_?.npy")); rng = np.random.default_rng(7); res = {}; seat = {}
    for f in files:
        D = np.load(f, allow_pickle=True).item(); cams = D["cams"]; nodes = [dict(pos=np.array(n["pos"]), bore=np.array(n["bore"])) for n in D["nodes"]]
        M0 = np.load(f.replace("x_", ""), allow_pickle=True).item(); seats = M0["seats"]; meta = M0["meta"]
        def add(name, est, cam, i):
            res.setdefault(name, []).append(float(np.linalg.norm(est - cam)))
            j = int(np.argmin(np.hypot(seats[:, 0] - est[0], seats[:, 1] - 0.14 - est[1])))
            seat.setdefault(name, []).append((int(j == i), int(abs(j // 8 - i // 8) <= 1 and abs(j % 8 - i % 8) <= 1)))
        pre = []; em_B = []; nano_A = []
        for n in nodes:
            p0 = precompute_node_geometry(n, GRID, SPACE)
            p0["A_dir"] = steering_for_dirs(p0["u_dir"], NANO12_EDGE_M, n["bore"], FC); p0["A_mir"] = steering_for_dirs(p0["best_mir_u"], NANO12_EDGE_M, n["bore"], FC); pre.append(p0)
            ex, ey, ez = axes(n["bore"]); v = GRID - n["pos"]; v /= np.linalg.norm(v, axis=1, keepdims=True); ul = np.stack([v @ ex, v @ ey, v @ ez], 1)
            one = np.ones(len(ul)); zero = np.zeros(len(ul))
            em_B.append((MAN.resp(ul, np.stack([one, zero], 1).astype(complex))[:12], MAN.resp(ul, np.stack([zero, one], 1).astype(complex))[:12], ul[:, 1] > 0.05))
            nano_A.append(sph_steer(n["pos"] + NANO12_EDGE_M[:, 0:1] * ex + NANO12_EDGE_M[:, 2:3] * ez))
        ARM = arm_local(); arm_A = [arm_steer(n, ARM) for n in nodes]
        for r in range(len(cams)):
            cam = cams[r]; i = int(meta[r][0])
            sa = []
            for n, node in enumerate(nodes):
                pk, Rr, pw = observe(csi_arm(D["nano"], r, n, node, ARM), rng, n_pkt=2); sa.append(capon_ll(Rr, arm_A[n]))
            add("arm40cm x4|capon fusion (approx.)", fuse(sa)[0], cam, i)
            for mode in ("ideal", "em"):
                obs = []; sc_b = []; sc_em = []
                for n, node in enumerate(nodes):
                    Hc = csi_nano(D["nano"], r, n, node, mode)
                    if np.mean(np.abs(Hc) ** 2) < 1e-30: Hc = Hc + 1e-12
                    pk, Rr, pw = observe(Hc, rng)
                    fe = [extract_wideband_cov_features(h, BFLAG) for h in pk]; feat = np.mean([x[0] for x in fe], axis=0); Rall = np.mean([x[1] for x in fe], axis=0)
                    with torch.no_grad(): Rai = model(torch.from_numpy(feat[None]), torch.tensor([BFLAG]))[0].numpy()
                    amp = np.abs(pk[0]); cv = float(np.median(np.std(amp, 1) / (np.mean(amp, 1) + 1e-12)))
                    obs.append(dict(R_all=Rall, R_ai=Rai, cv_f=cv, raw_pow=pw)); sc_b.append(capon_ll(Rr, nano_A[n]))
                    if mode == "em": sc_em.append(emcal_ll(Rr, *em_B[n]))
                add(f"nano4_{mode}|repo MA-PINN+AI", localize_ma_pinn(obs, pre, GRID), cam, i)
                add(f"nano4_{mode}|capon fusion", fuse(sc_b)[0], cam, i)
                if mode == "em": add("nano4_em|full dual-pol chamber cal", fuse(sc_em)[0], cam, i)
        print(f, "done", flush=True)
    np.save(f"res_x_{KIND}_{BAND}_{int(E.BW/1e6)}.npy", dict(res=res, seat=seat), allow_pickle=True)
    print(f"\n=== 4-corner Nano-84, {KIND} {BAND} BW={E.BW/1e6:.1f} MHz n={len(next(iter(res.values())))}  (median / P90 / <=1 m / <=0.5 m)")
    for k, v in res.items():
        v = np.array(v); st = np.array(seat[k]).mean(0) * 100
        print(f"  {k:42s} {np.median(v):5.2f} m  {np.percentile(v, 90):5.2f} m  {100*np.mean(v<=1.0):5.1f}%  {100*np.mean(v<=0.5):5.1f}%   seat {st[0]:4.0f}%  3x3 {st[1]:4.0f}%")

if __name__ == "__main__":
    run()
