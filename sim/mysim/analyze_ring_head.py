"""Summaries for exp_ring.pkl (head structure) and exp_head.pkl (head motion)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S

R = "/opt/detecam/sim/results/"


def ring():
    r = S.load(R + "exp_ring.pkl")
    print("== head structure (exp_ring) : exam board pair | other diag | other all4   mean (share <=0.5 m)")
    for band in (("2g4",), ("5g",), ("2g4", "5g")):
        print(" band", "+".join(band))
        for cfg in ["d85", "d85_pan_inc", "d85_pan_coh", "d120_8", "d120_8_pan_coh", "d160_8", "d160_8_pan_inc",
                    "d160_8_pan_coh", "d120_16", "d160_16"]:
            one = S.triple(r, cfg, bands=band)
            two = S.triple(r, cfg, bands=band, grp=lambda p: "x2_" + p)
            print(f"  {cfg:16s} 1 look: {S.fmt(one)}   2 looks: {S.fmt(two)}")
    for cfg in ["d85_pan_coh", "d160_8", "d160_8_pan_coh", "d160_16"]:
        for nm, scs, kf in (("exam", S.EXAM, lambda s: S.board_key(s)), ("diag", S.OTHER, lambda s: "diag_equal"),
                            ("all4", S.OTHER, lambda s: "all4_equal")):
            d, ci = S.paired_delta(r, "d85", cfg, scs, kf)
            print(f"  delta {cfg:16s} vs d85 {nm}: {d:+.3f} m  95% room-bootstrap [{ci[0]:+.3f}, {ci[1]:+.3f}]")


def head(cfgs=("d85",)):
    r = S.load(R + "exp_head.pkl")
    import exp_head as H
    print("== head motion (exp_head, 2.4 GHz seed 0): exam board pair | other diag | other all4")
    for cfg in cfgs:
        if not any(cfg in o for _, o in r.values()):
            continue
        print(" cfg", cfg)
        for mn, ks in H.MOTIONS.items():
            t = S.triple(r, cfg, bands=("2g4",), grp=lambda p, mn=mn: f"{mn}|{p}")
            print(f"  {mn:9s} looks/receiver={1 + len(ks):2d}  {S.fmt(t)}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "ring"
    if what == "ring":
        ring()
    else:
        head(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("d85",))
