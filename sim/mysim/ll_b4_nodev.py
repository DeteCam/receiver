"""B4 and fused-position bearing without the old-room model (for bands that have no old-room maps).  [simulation]
Usage: python3 mysim/ll_b4_nodev.py <cfg> <real pkl name> [group]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S
import real_scenes as R

cfg = sys.argv[1]
res = S.load(os.path.join(S.ROOT, "results", sys.argv[2]))
grp = sys.argv[3] if len(sys.argv) > 3 else "four"
rows = S.spot_table(res, cfg, S.real_spots("all"), S.real_looks)
row = {(r["scene"], r["seed"], r["cam"], r["node"]): r for r in rows}


def st(e):
    e = np.asarray(e, dtype=np.float64)
    return f"{e.mean():6.2f} {np.median(e):5.1f} {np.percentile(e, 90):5.1f} {np.percentile(e, 95):5.1f}  {100 * np.mean(e > 20):4.1f}%  n={len(e)}"


out = dict(single=[], b4=[], fused_b4=[], fused_all=[])
for key in sorted(res):
    sc, bd, seed = key
    mt, outs = res[key]
    if cfg not in outs or sc in R.EXAM or "ll_all" not in outs[cfg]:
        continue
    o = outs[cfg]
    LL = o["ll_all"].astype(np.float32); grid = o["grid"].astype(np.float64)
    npos = o["node_pos_est"].astype(np.float64); cams, nodes = mt["cams"], mt["nodes"]
    base = R.COMBO[sc][grp]
    for r in range(len(cams)):
        maps = [np.sum([LL[r, j] for j in S.real_looks(sc, i)], axis=0) for i in base]
        pt = grid[int(np.argmax(np.sum(maps, axis=0)))]
        es = np.array([row[(sc, seed, r, i)]["err"] for i in base])
        sp = np.array([row[(sc, seed, r, i)]["spread"] for i in base])
        ef = []
        for i in base:
            vt = cams[r] - nodes[i]; v = pt - npos[i]
            ef.append(float(S.azerr(np.arctan2(v[1], v[0]), np.arctan2(vt[1], vt[0]))))
        k = int(np.argmin(sp))
        out["single"] += list(es); out["b4"].append(es[k]); out["fused_b4"].append(ef[k]); out["fused_all"] += ef
print(f"# {sys.argv[2]} cfg={cfg} group={grp}: mean median p90 p95 | gross>20 | n   [simulation]")
for k, nm in (("single", "every spot, its own bearing (M1)"), ("b4", "most reliable spot (spread), its own bearing (B4)"),
              ("fused_b4", "most reliable spot (spread), bearing to the fused position"), ("fused_all", "every spot, bearing to the fused position")):
    print(f"  {nm:60s} {st(out[k])}")
