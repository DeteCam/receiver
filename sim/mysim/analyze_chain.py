"""Summary of exp_chain.pkl: position error per chain-order handling, plus packet-level order statistics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import summ as S
from analyze_shapes import table

R = "/opt/detecam/sim/results/"
CFGS = ["base", "naive", "stronger", "drop3", "drop6", "pad6", "pad6_drop3", "pad10", "a2_drop3", "a5", "a5_drop3",
        "agcraw_n5", "drop3_agcraw_n5"]

if __name__ == "__main__":
    r = S.load(R + "exp_chain.pkl")
    table(r, CFGS, "base", "chain order handling (D120-16 head)")
    print("== packet statistics (mean over cameras x nodes x rooms x bands x seeds)")
    for c in CFGS:
        acc = {}
        for k, (meta, o) in r.items():
            if c not in o or "extra" not in o[c]:
                continue
            for ek, ev in o[c]["extra"].items():
                acc.setdefault(ek, []).append(np.asarray(ev, dtype=float).ravel())
        if acc:
            print(f"  {c:16s} " + "  ".join(f"{ek}={100 * np.mean(np.concatenate(v)):.3f}%" for ek, v in sorted(acc.items())))
