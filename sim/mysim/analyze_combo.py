"""Summaries for exp_combo.pkl (jit10 data: +-10 cm along the wall (x) / vertical (z), 2 bands, seeds 0-1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S

r = S.load("/opt/detecam/sim/results/exp_combo.pkl")
cfgs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["d85", "d120_16", "d85_seat", "d120_16_seat", "d85_seat_ref",
                                                          "d120_16_seat_ref"]
for cfg in cfgs:
    if not any(cfg in o for _, o in r.values()):
        continue
    print("cfg", cfg)
    for mv in ("s", "x", "z", "xz"):
        for band in (("2g4", "5g"), ("2g4",), ("5g",)):
            t = S.triple(r, cfg, bands=band, grp=lambda p, mv=mv: f"{mv}|{p}")
            line = f"  {mv:3s} {'+'.join(band):8s} {S.fmt(t)}"
            if cfg.endswith("_ref"):
                tr = S.triple(r, cfg, bands=band, grp=lambda p, mv=mv: f"{mv}|{p}") if False else None
                kb = lambda s, mv=mv: f"{mv}|{S.BOARD[s]}_ref"
                e1 = S.stat(S.errs(r, cfg, S.EXAM, kb, band)); e2 = S.stat(S.errs(r, cfg, S.OTHER, lambda s, mv=mv: f"{mv}|diag_ref", band))
                e3 = S.stat(S.errs(r, cfg, S.OTHER, lambda s, mv=mv: f"{mv}|all4_ref", band))
                line += f"   refined: {S.fmt((e1, e2, e3))}"
            print(line)
    # seat metrics for exam rooms
    s1, s3 = [], []
    for k, (m, o) in r.items():
        if k[0] in S.EXAM and cfg in o and "seat1" in o[cfg]:
            s1.append(o[cfg]["seat1"].get(f"xz|{S.BOARD[k[0]]}")); s3.append(o[cfg]["seat3"].get(f"xz|{S.BOARD[k[0]]}"))
    s1 = [a for a in s1 if a is not None]; s3 = [a for a in s3 if a is not None]
    if s1:
        print(f"  seat (xz moves, board pair): top1={np.mean(np.concatenate(s1)):.2f} top3={np.mean(np.concatenate(s3)):.2f}")
    s1, s3 = [], []
    for k, (m, o) in r.items():
        if k[0] in S.EXAM and cfg in o and "seat1" in o[cfg]:
            s1.append(o[cfg]["seat1"].get(f"s|{S.BOARD[k[0]]}")); s3.append(o[cfg]["seat3"].get(f"s|{S.BOARD[k[0]]}"))
    s1 = [a for a in s1 if a is not None]; s3 = [a for a in s3 if a is not None]
    if s1:
        print(f"  seat (no move, board pair): top1={np.mean(np.concatenate(s1)):.2f} top3={np.mean(np.concatenate(s3)):.2f}")
