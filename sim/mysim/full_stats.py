"""Full statistics of one configuration (AGENTS.md 5.1-4): mean / median / 90 % / 95 % of the single-spot bearing error M1
and of the fused bearing N1, the worst room, and the fused position within 0.5 m and 1 m per room.  [simulation]
Usage: python3 mysim/full_stats.py <pkl>:<cfg> [<pkl>:<cfg> ...]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import molex_compare as M  # noqa: E402


def q(v):
    v = np.asarray(list(v), dtype=np.float64)
    return f"{v.mean():5.2f} / {np.median(v):5.2f} / {np.percentile(v, 90):5.1f} / {np.percentile(v, 95):5.1f}"


def main():
    print("# cfg        M1 mean / median / p90 / p95 [deg]   worst room (mean M1)     N1 mean / median / p90 / p95 [deg]"
          "   pos <=0.5 m / <=1 m per room [%]")
    for arg in sys.argv[1:]:
        p, c = arg.split(":")
        m1, n1, pos = M.table(p, c)
        rooms = sorted({k[0] for k in pos})
        by = {r: np.mean([v for k, v in m1.items() if k[0] == r]) for r in rooms}
        worst = max(by, key=by.get)
        pr = "  ".join(f"{r} {100 * np.mean([pos[k] <= 0.5 for k in pos if k[0] == r]):.1f}/"
                       f"{100 * np.mean([pos[k] <= 1.0 for k in pos if k[0] == r]):.1f}" for r in rooms)
        print(f"  {c:8s} {q(m1.values())}   {worst} {by[worst]:.1f}   {q(n1.values())}   {pr}   n(M1)={len(m1)} n(cam)={len(pos)}")


if __name__ == "__main__":
    main()
