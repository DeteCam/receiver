"""Summaries for exp_shapes.pkl, exp_motion2.pkl, exp_chain.pkl, exp_qc.pkl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S

R = "/opt/detecam/sim/results/"
KF = ((S.EXAM, lambda s: S.board_key(s)), (S.OTHER, lambda s: "diag_equal"), (S.OTHER, lambda s: "all4_equal"))


def table(res, cfgs, ref, label):
    print(f"== {label} (exam board pair | other diag | other all4; mean (share <= 0.5 m); delta vs {ref} with 95% room CI)")
    for c in cfgs:
        if not any(c in o for _, o in res.values()):
            continue
        t = S.triple(res, c)
        ds = []
        for scs, kf in KF:
            d, ci = S.paired_delta(res, ref, c, scs, kf, n_boot=500) if c != ref else (0.0, (0.0, 0.0))
            ds.append(f"{d:+.2f}[{ci[0]:+.2f},{ci[1]:+.2f}]")
        by = " / ".join(S.fmt(S.triple(res, c, bands=(b,))) for b in ("2g4", "5g"))
        print(f"  {c:14s} {S.fmt(t)}   d: {' '.join(ds)}")
        print(f"  {'':14s}   2.4 | 5 GHz: {by}")


def groups(res, cfg, names, label):
    print(f"== {label}: {cfg}")
    for g in names:
        t = S.triple(res, cfg, grp=lambda p, g=g: f"{g}|{p}")
        print(f"  {g:8s} {S.fmt(t)}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "shapes"
    if what == "shapes":
        r = S.load(R + "exp_shapes.pkl")
        import shapes as SH
        table(r, ["d120_16", "d160_16"] + list(SH.GEOM), "d120_16", "head shapes")
    elif what == "motion2":
        r = S.load(R + "exp_motion2.pkl")
        groups(r, "nodfold", ["one", "two", "three", "unf", "fold2"], "nod / fold looks (two=+20 nod, three=+-20)")
        groups(r, "repeat", ["one", "two", "three", "unf", "fold2"], "control: same looks repeated")
    elif what == "qc":
        r = S.load(R + "exp_qc.pkl")
        table(r, ["clean", "mix", "mix_rule", "mix_ai"], "mix", "packet quality filter")
