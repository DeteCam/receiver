"""Paired comparison of the box-distance / cable heads with P0F (plan sim/plan_20261126_molex.md).  [simulation]
M1 = every spot's own bearing (outside rooms, four spots), N1 = bearing to the 4-spot fused position from the spot with
the smallest spread, pos = fused position within 0.5 m (per room).  Differences are paired on the same camera, spot
and seed; 95 % intervals by bootstrap over (room, seed) blocks.
Usage: python3 mysim/molex_compare.py <base pkl>:<base cfg> <pkl>:<cfg> [<pkl>:<cfg> ...]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S  # noqa: E402
import real_scenes as R  # noqa: E402


def table(pkl, cfg):
    res = S.load(os.path.join(S.ROOT, "results", pkl))
    m1, n1, pos = {}, {}, {}
    for key in sorted(res):
        sc, band, seed = key
        mt, outs = res[key]
        if cfg not in outs or sc in R.EXAM or "ll_all" not in outs[cfg]:
            continue
        o = outs[cfg]
        LL = o["ll_all"].astype(np.float32)
        grid = o["grid"].astype(np.float64)
        npos = o["node_pos_est"].astype(np.float64)
        cams, nodes = mt["cams"], mt["nodes"]
        base = R.COMBO[sc]["four"]
        for r in range(len(cams)):
            maps = [np.sum([LL[r, j] for j in S.real_looks(sc, i)], axis=0) for i in base]
            pt = grid[int(np.argmax(np.sum(maps, axis=0)))]
            ef, es, sp = [], [], []
            for i, mp in zip(base, maps):
                v = grid - npos[i][None]
                az = np.arctan2(v[:, 1], v[:, 0])
                vt = cams[r] - nodes[i]
                az_true = np.arctan2(vt[1], vt[0])
                es.append(float(S.azerr(az[int(np.argmax(mp))], az_true)))
                w = pt - npos[i]
                ef.append(float(S.azerr(np.arctan2(w[1], w[0]), az_true)))
                sp.append(S.circ_spread(mp, az))
                m1[(sc, seed, r, i)] = es[-1]
            n1[(sc, seed, r)] = ef[int(np.argmin(sp))]
            pos[(sc, seed, r)] = float(np.linalg.norm(pt[:2] - cams[r][:2]))
    return m1, n1, pos


def boot(diff_by_block, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = list(diff_by_block)
    sums = np.array([np.sum(diff_by_block[b]) for b in blocks])
    cnts = np.array([len(diff_by_block[b]) for b in blocks])
    idx = rng.integers(0, len(blocks), (n, len(blocks)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(est, [2.5, 97.5])


def st(v):
    v = np.asarray(list(v), dtype=np.float64)
    return f"{v.mean():6.2f} / {np.percentile(v, 95):5.1f}"


def main():
    bp, bc = sys.argv[1].split(":")
    B = table(bp, bc)
    rooms = sorted({k[0] for k in B[2]})
    print(f"# base {bc} ({bp})   [simulation, 2.4 GHz, set A outside rooms, seeds 0-2]")
    print("# cfg      M1 mean/p95   dM1 (95% CI)          rel    N1 mean/p95   pos0.5 per room (" + ", ".join(rooms) + ")")
    for arg in [sys.argv[1]] + sys.argv[2:]:
        p, c = arg.split(":")
        T = B if arg == sys.argv[1] else table(p, c)
        keys = sorted(set(T[0]) & set(B[0]))
        d = {}
        for k in keys:
            d.setdefault((k[0], k[1]), []).append(T[0][k] - B[0][k])
        dm = np.mean([T[0][k] - B[0][k] for k in keys])
        lo, hi = boot(d)
        rel = np.mean([T[0][k] for k in keys]) / np.mean([B[0][k] for k in keys]) - 1.0
        pr = [100.0 * np.mean([T[2][k] <= 0.5 for k in T[2] if k[0] == sc]) for sc in rooms]
        print(f"  {c:7s} {st(T[0].values())}  {dm:+6.2f} ({lo:+5.2f},{hi:+5.2f})  {100 * rel:+6.1f}%  {st(T[1].values())}   "
              + "  ".join(f"{x:5.1f}" for x in pr))
        per_seed = {}
        for k, v in T[1].items():
            per_seed.setdefault(k[1], []).append(v)
        print("           N1 by seed: " + "  ".join(f"s{s} {np.mean(v):5.2f}" for s, v in sorted(per_seed.items())))


if __name__ == "__main__":
    main()
