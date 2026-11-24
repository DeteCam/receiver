"""Numbers quoted in the report (reads res_pe2_*.npy).  Usage: report_numbers.py"""
import os, numpy as np
def L(tag):
    f = f"res_pe2_{tag}.npy"; return np.load(f, allow_pickle=True).item() if os.path.exists(f) else None
q = lambda e: f"{np.mean(e):4.1f} / {np.median(e):4.1f} / {np.percentile(e, 90):4.1f}"
def single(d, nodes=None, conf=None, bart=False, files=None):
    R = [x for x in d["rows"] if not x["beacon"] and (nodes is None or x["n"] in nodes) and (files is None or x["f"] in files)]
    e = np.array([x["err_b" if bart else "err"] for x in R])
    if conf: sp = np.array([x["spread"] for x in R]); e = e[sp <= np.percentile(sp, conf)]
    return e
def fused(d, sub):
    P = [x for x in d["pos"] if x["sub"] == sub and not x["beacon"]]; fa = np.concatenate([x["fuse_az"] for x in P]); pe = np.array([x["err"] for x in P]); st = np.array([x["seat"] for x in P], float)
    s = f"dir {q(fa)} | <=1 m {100 * np.mean(pe <= 1):3.0f}% | median {np.median(pe):.2f} m"
    if np.isfinite(st).all(): s += f" | seat {100 * st[:, 0].mean():.0f}% 3x3 {100 * st[:, 1].mean():.0f}%"
    return s
def seq(d):
    S = np.array([x["seq"] for x in d["adap"] if not x["beacon"]]); return " ; ".join(f"{k + 1} spot(s): {q(S[:, k, 1])} P95 {np.percentile(S[:, k, 1], 95):4.1f} <=1m {100 * np.mean(S[:, k, 2] <= 1):.0f}%" for k in range(S.shape[1]))
def show(tag, label, good, pair):
    d = L(tag)
    if d is None: print(f"{label:46s} (not run)"); return
    bad = [n for n in range(4) if n not in good]
    print(f"{label:46s} all {q(single(d))} | good spots {q(single(d, good))} conf80 {single(d, good, 80).mean():4.1f} | other spots {q(single(d, bad))} | all conf80 {single(d, None, 80).mean():4.1f} | plain beamforming all {q(single(d, bart=True))}")
    print(f"{'':46s} fused {pair}: {fused(d, pair)}   ||   fused x4: {fused(d, 'x4')}")
    print(f"{'':46s} carried around: {seq(d)}")
print("mean / median / P90 of the direction error (deg)\n")
for tag, label in (("f_c24", "classroom 2.4 ideal"), ("s_em100", "classroom 2.4 real antenna"), ("f_c5", "classroom 5 ideal"), ("s5_em100", "classroom 5 real antenna"), ("c_r90em", "classroom 2.4 real antenna, 9 cm arm"),
                   ("xl24", "classroom 2.4 ideal, low spots"), ("xl24em", "classroom 2.4 real, low spots"), ("xl5em", "classroom 5 real, low spots"), ("xe24", "classroom 2.4 ideal, 4 spots 0.5 m apart")):
    show(tag, label, (0, 1), "x2 adj")
for tag, label in (("f_m24", "motel 2.4 ideal"), ("m_em100", "motel 2.4 real antenna"), ("m_r90em", "motel 2.4 real antenna, 9 cm arm"), ("f_m5", "motel 5 ideal"), ("m5_em100", "motel 5 real antenna"),
                   ("xc24", "motel 2.4 ideal, away from walls"), ("xc24em", "motel 2.4 real, away from walls"), ("xd24", "motel 2.4 ideal, 4 spots 0.5 m apart (corner 0)"),
                   ("xd24em", "motel 2.4 real, 4 spots 0.5 m apart (corner 0)"), ("xf24em", "motel 2.4 real, 4 spots 0.5 m apart (corner 1)"), ("w_conc", "motel 2.4 ideal, concrete walls"), ("w_metal", "motel 2.4 ideal, metal walls")):
    show(tag, label, (0, 2), "x2 diag")
d = L("f_m24")
if d is not None: print(f"\nmotel 2.4 ideal, files 0-1 only (same cameras as the wall-material test): map-aware {q(single(d, files=(0, 1)))} | plain beamforming {q(single(d, bart=True, files=(0, 1)))}")
for tag in ("w_conc", "w_metal"):
    d = L(tag)
    if d is not None: print(f"{tag}: map-aware {q(single(d))} | plain beamforming {q(single(d, bart=True))}")
