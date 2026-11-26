"""Bearing read off the fused position (offline, stored maps).  [simulation]
After k spots are fused into one position estimate, each spot's bearing is taken as the direction from that spot to
the fused estimate.  Compared with the single-spot bearing (plan metric) and with 'most reliable spot' (B4).
Usage: python3 mysim/ll_fbear.py <band> <cfg> <real pkl name> [group]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ll_study as S
import real_scenes as R

band, cfg = sys.argv[1], sys.argv[2]
realp = os.path.join(S.ROOT, "results", sys.argv[3])
grp = sys.argv[4] if len(sys.argv) > 4 else "four"
devcfg = os.environ.get("DEV_CFG", cfg)     # learned-reliability model from another cfg's development rooms
devp = os.path.join(S.ROOT, "results", f"dev_ll_{devcfg}_{band}.pkl")
dev_rows = S.spot_table(S.load(devp), devcfg, S.dev_spots, S.dev_looks)
model = S.train_model(dev_rows)
res = S.load(realp)
rows = S.spot_table(res, cfg, S.real_spots("all"), S.real_looks)
pb = model.predict_proba(S.xmat(rows))[:, 1]
row = {(r["scene"], r["seed"], r["cam"], r["node"]): (r, p) for r, p in zip(rows, pb)}


def st(e):
    e = np.asarray(e, dtype=np.float64)
    return f"{e.mean():6.2f} {np.median(e):5.1f} {np.percentile(e, 90):5.1f} {np.percentile(e, 95):5.1f}  {100 * np.mean(e > 20):4.1f}%  n={len(e)}"


out = {k: [] for k in ("single", "fused_all", "b4_spread", "b4_learned", "fused_at_b4s", "fused_at_b4l", "fused_first",
                       "single_first", "blend_s", "blend_l")}
per_kind = {}
per_room, per_seed = {}, {}
for key in sorted(res):
    sc, bd, seed = key
    mt, outs = res[key]
    if cfg not in outs or sc in R.EXAM or "ll_all" not in outs[cfg]:
        continue
    o = outs[cfg]
    LL = o["ll_all"].astype(np.float32); grid = o["grid"].astype(np.float64)
    npos = o["node_pos_est"].astype(np.float64); cams, nodes = mt["cams"], mt["nodes"]
    base = R.COMBO[sc][grp]
    if isinstance(base, int):
        continue
    for r in range(len(cams)):
        maps = [np.sum([LL[r, j] for j in S.real_looks(sc, i)], axis=0) for i in base]
        c = int(np.argmax(np.sum(maps, axis=0)))
        pt = grid[c]
        ef, es, sp, pl = [], [], [], []
        for i in base:
            vt = cams[r] - nodes[i]; az_true = np.arctan2(vt[1], vt[0])
            v = pt - npos[i]
            ef.append(float(S.azerr(np.arctan2(v[1], v[0]), az_true)))
            rw, p = row[(sc, seed, r, i)]
            es.append(rw["err"]); sp.append(rw["spread"]); pl.append(p)
        ef, es, sp, pl = map(np.array, (ef, es, sp, pl))
        ks, kl = int(np.argmin(sp)), int(np.argmin(pl))
        out["single"] += list(es); out["fused_all"] += list(ef)
        out["b4_spread"].append(es[ks]); out["b4_learned"].append(es[kl])
        out["fused_at_b4s"].append(ef[ks]); out["fused_at_b4l"].append(ef[kl])
        out["fused_first"].append(ef[0]); out["single_first"].append(es[0])
        # per spot: keep the single-spot bearing when that spot looks reliable, else the fused one
        out["blend_s"] += [es[k] if sp[k] <= 13.87 else ef[k] for k in range(len(base))]
        out["blend_l"] += [es[k] if pl[k] <= 0.2 else ef[k] for k in range(len(base))]
        per_kind.setdefault(R.KIND[sc], []).extend(ef)
        per_room.setdefault(sc, dict(m1=[], n1=[], pos=[]))
        per_room[sc]["m1"] += list(es); per_room[sc]["n1"].append(ef[ks])
        per_room[sc]["pos"].append(float(np.linalg.norm(pt[:2] - cams[r][:2])))
        per_seed.setdefault(seed, []).append(ef[ks])

print(f"# bearing read off the fused position, group '{grp}', {os.path.basename(realp)} cfg={cfg} {band}   [simulation]")
print("# columns: mean median p90 p95 | gross>20 | n      (lines: mean <= 8.92, p95 < 27.53)")
names = dict(single="every spot, its own bearing (plan metric M1)",
             fused_all="every spot, bearing to the fused position",
             single_first="first spot, its own bearing", fused_first="first spot, bearing to the fused position",
             b4_spread="most reliable spot (spread), its own bearing (B4)",
             fused_at_b4s="most reliable spot (spread), bearing to the fused position",
             b4_learned="most reliable spot (learned), its own bearing",
             fused_at_b4l="most reliable spot (learned), bearing to the fused position",
             blend_s="every spot: own bearing if spread <= 13.87, else fused",
             blend_l="every spot: own bearing if learned p <= 0.2, else fused")
for k, v in out.items():
    print(f"  {names[k]:62s} {st(v)}")
for k, v in sorted(per_kind.items()):
    print(f"  fused, every spot, kind {k:9s}                                   {st(v)}")
print("# per room: M1 (every spot, own bearing) | N1 (most reliable spot by spread, bearing to the fused position) | fused position within 0.5 m")
for sc, d in per_room.items():
    print(f"  {sc:13s} M1 {st(d['m1'])} | N1 {st(d['n1'])} | pos0.5 {100 * np.mean(np.array(d['pos']) <= 0.5):5.1f}%")
for sd, v in sorted(per_seed.items()):
    print(f"  seed {sd}: N1 {st(v)}")
