"""One-at-a-time ablation of the "real" condition (plan sim/plan_20261126_real.md, section 3) for P0.
Each variant removes ONE item of the bundle (or changes one counter-measure); same rooms, cameras and random draws.
Usage: python3 mysim/exp_real_abl.py <scenes> <bands> <seeds> [variant names]"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E
import exp_final as F
import exp_real as X
import real_scenes as R


def variants(band, exam):
    alpha, g, snr = F.ANCHOR["H4"][band]
    full = X.cond_cfg("real", band, exam, "d120_16_H4")
    kw = full["synth_kw"]
    def mod(**ch):
        c = dict(full)
        c.update({k: v for k, v in ch.items() if k != "kw"})
        if "kw" in ch:
            k2 = dict(kw)
            for a, b in ch["kw"].items():
                if b is None:
                    k2.pop(a, None)
                else:
                    k2[a] = b
            c["synth_kw"] = k2
        return c
    V = {
        "R_full": full,
        "R_noswapfix": mod(kw=dict(swap_fix=False)),                      # swapped packets left to the trimmed mean
        "R_noswap": mod(kw=dict(p_swap=0.0, swap_fix=False)),
        "R_noagc": mod(kw=dict(agc=None)),
        "R_nomis": mod(kw=dict(p_mis=0.0, mis_guard=False)),
        "R_noleak": mod(kw=dict(leak_db=None)),
        "R_acal3": mod(kw=dict(alpha_err_deg=3.0, alpha_err_db=0.25)),
        "R_mean": mod(kw=dict(agg="mean")),
        "R_snr": mod(snr_db=snr),
        "R_cal": mod(cal_phase_deg=4.5, cal_amp_db=0.4),
        "R_pose": mod(sig_xy=0.05, sig_z=0.025, sig_head=1.5),
        "R_n10": mod(n_sweep=10),
        "R_dyn0": mod(dyn_frac=0.0),
    }
    return V


def main():
    scenes, bands, seeds = sys.argv[1].split(","), sys.argv[2].split(","), [int(x) for x in sys.argv[3].split(",")]
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    out = os.path.join(X.ROOT, "results", "real_abl.pkl")
    for band in bands:
        for sc in scenes:
            V = variants(band, sc in R.EXAM)
            X.setup_scene(sc, "nod", "d120_16_H4")
            for nm, cfg in V.items():
                if names and nm not in names:
                    continue
                E.run({nm: cfg}, [sc], [band], seeds, X.data_dir(sc, "real"), X.NEC_REAL, out, store_zone=False, verbose=True)
    print("ABL_DONE", flush=True)


if __name__ == "__main__":
    main()
