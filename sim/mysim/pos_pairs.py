"""Paired count of cameras found within 0.5 m (fused position, four spots) against a baseline: lost / won per room and an
exact sign test.  [simulation]  Usage: python3 mysim/pos_pairs.py <base pkl>:<cfg> <pkl>:<cfg> [...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import molex_compare as M
base = M.table(*sys.argv[1].split(':'))
for arg in sys.argv[2:]:
    T = M.table(*arg.split(':'))
    for sc in sorted({k[0] for k in base[2]}):
        keys = [k for k in base[2] if k[0] == sc and k in T[2]]
        b = np.array([base[2][k] for k in keys]); t = np.array([T[2][k] for k in keys])
        lost = int(np.sum((b <= 0.5) & (t > 0.5))); won = int(np.sum((b > 0.5) & (t <= 0.5)))
        per_seed = {s: (100*np.mean([base[2][k] <= 0.5 for k in keys if k[1] == s]), 100*np.mean([T[2][k] <= 0.5 for k in keys if k[1] == s])) for s in sorted({k[1] for k in keys})}
        # exact sign test (two-sided) on discordant pairs
        from math import comb
        n = lost + won; kk = min(lost, won)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n) if n else 1.0
        print(f"{arg.split(':')[1]:6s} {sc:13s} base {100*np.mean(b<=0.5):5.1f} -> {100*np.mean(t<=0.5):5.1f}  lost {lost:2d} won {won:2d} p={p:.3f}  med {np.median(b):.2f}->{np.median(t):.2f}  <=1m {100*np.mean(b<=1):5.1f}->{100*np.mean(t<=1):5.1f}  seeds " + " ".join(f"s{s}:{a:.0f}->{c:.0f}" for s,(a,c) in per_seed.items()))
