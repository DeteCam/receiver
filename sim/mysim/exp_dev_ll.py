"""Likelihood maps + evidence values of P0 on the OLD rooms (development set), for tuning reliability rules / models
and fusion rules without touching the realistic rooms.  Eight wall spots at 2.0 m, three nod looks each.
Usage: python3 mysim/exp_dev_ll.py <seeds> [band] [design name]"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E
import exp_final as F
import exp_wall as W
import exp_real as X
import est_variants

LAB = W.LAB


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    band = sys.argv[2] if len(sys.argv) > 2 else "2g4"
    nm = sys.argv[3] if len(sys.argv) > 3 else "P0F"
    base, looks, extra = X.DESIGNS[nm]
    pat = f"{base}_H4"
    n = 16
    E.STORE_LL_ALL = True
    E.NODE_YAW = {i + 8 * k: W.YAW8[i] for k in range(2) for i in range(8)}
    E.AUG_NODES = list(range(n)) * 2
    E.NODE_PAT = {**{n + i: f"{pat}_t+20" for i in range(n)}, **{2 * n + i: f"{pat}_t-20" for i in range(n)}}
    E.POSE_SHARE = {**{n + i: i for i in range(n)}, **{2 * n + i: i for i in range(n)}}
    for i in range(n):
        E.NODE_YAW[n + i] = E.NODE_YAW[i]
        E.NODE_YAW[2 * n + i] = E.NODE_YAW[i]
    E.ONLY_NODES = [i + k * n for k in range(3) for i in range(8)]          # height A only
    E.SINGLE_GROUPS = {LAB[i]: [i, n + i, 2 * n + i] for i in range(8)}
    E.EXTRA_GROUPS = {"quadm": tuple(j for i in (4, 5, 6, 7) for j in (i, n + i, 2 * n + i)),
                      "quadc": tuple(j for i in (0, 1, 2, 3) for j in (i, n + i, 2 * n + i)),
                      "mopp": tuple(j for i in (4, 6) for j in (i, n + i, 2 * n + i))}
    cfg = dict(X.cond_cfg("design", band, False, pat), **extra)
    cfg["est_cls"] = getattr(est_variants, cfg["est_cls"]) if isinstance(cfg.get("est_cls"), str) else cfg.get("est_cls")
    out = os.path.join(X.ROOT, "results", f"dev_ll_{nm}_{band}.pkl")
    E.run({nm: cfg}, F.OUTSIDE, [band], seeds, os.path.join(X.ROOT, "data_wall", "nop"), X.NEC_REAL, out, store_zone=False, verbose=True)
    print("DEV_LL_DONE", flush=True)


if __name__ == "__main__":
    main()
