"""Summary tables for the realistic rooms (plan sim/plan_20261126_real.md, section 4).
Usage: python3 mysim/analyze_real.py <tag> [cfg] [--vs other_cfg[@other_tag]] [--bands 2g4,5g]
Reads results/real_<tag>.pkl.  All numbers are simulation values."""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_scenes as R  # noqa: E402
import real_meta as M  # noqa: E402

ROOT = "/opt/detecam/sim"
BEHIND = {"bath_vent", "bath_shelf", "shower_shelf", "stall_tp", "stall_hook", "stall_vent", "stall_floor"}
_meta = {}


def meta(sc, seed):
    if (sc, seed) not in _meta:
        _meta[(sc, seed)] = M.meta_for(sc, seed, "base")
    return _meta[(sc, seed)]


def load(tag):
    with open(os.path.join(ROOT, "results", f"real_{tag}.pkl"), "rb") as f:
        return pickle.load(f)


def stat(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    if len(v) == 0:
        return "    -     -     -     -      0"
    return f"{v.mean():5.1f} {np.median(v):5.1f} {np.percentile(v, 90):5.1f} {np.percentile(v, 95):5.1f} {len(v):6d}"


def bearing_rows(res, cfg, band, spots="four"):
    """list of dict(scene, seed, spot, type, err (n_cam,), conceal, behind)"""
    rows = []
    for (sc, bd, seed), (mt, out) in sorted(res.items()):
        if bd != band or cfg not in out:
            continue
        ex = out[cfg].get("extra", {})
        m = meta(sc, seed)
        conceal = np.array([c != "open" for c in m["conceal"]])
        behind = np.array([t in BEHIND for t in m["cam_types"]])
        sp = R.SPOTS[sc]
        idx = range(len(sp)) if spots == "all" else R.COMBO[sc][spots]
        for i in idx:
            key = "sg_" + sp[i][0]
            if key not in ex:
                continue
            rows.append(dict(scene=sc, seed=seed, spot=sp[i][0], type=sp[i][5], err=ex[key][:, 0].astype(np.float64),
                             conceal=conceal, behind=behind))
    return rows


def pos_rows(res, cfg, band, group):
    rows = []
    for (sc, bd, seed), (mt, out) in sorted(res.items()):
        if bd != band or cfg not in out:
            continue
        pe = out[cfg]["pos_err"].get(f"{group}_equal")
        if pe is None:
            continue
        o = out[cfg]
        rows.append(dict(scene=sc, seed=seed, err=pe.astype(np.float64),
                         seat1=o.get("seat1", {}).get(group), seat3=o.get("seat3", {}).get(group)))
    return rows


def cat(rows, key="err", f=None):
    v = [r[key] if f is None else r[key][f(r)] for r in rows]
    return np.concatenate(v) if v else np.zeros(0)


def room_boot(rows_a, rows_b, n_boot=4000, seed=1):
    """Paired difference a - b per (scene, seed, spot); bootstrap over rooms (plan) and over (room, seed) clusters."""
    ka = {(r["scene"], r["seed"], r.get("spot")): r["err"] for r in rows_a}
    kb = {(r["scene"], r["seed"], r.get("spot")): r["err"] for r in rows_b}
    keys = sorted(set(ka) & set(kb), key=repr)
    if not keys:
        return None
    d = {k: ka[k] - kb[k] for k in keys}
    mean_b = float(np.mean(np.concatenate([kb[k] for k in keys])))
    mean_d = float(np.mean(np.concatenate([d[k] for k in keys])))
    rng = np.random.default_rng(seed)
    out = {}
    for name, unit in (("room", lambda k: k[0]), ("room_seed", lambda k: (k[0], k[1]))):
        units = sorted({unit(k) for k in keys}, key=repr)
        per = {u: np.concatenate([d[k] for k in keys if unit(k) == u]) for u in units}
        bs = []
        for _ in range(n_boot):
            pick = rng.integers(len(units), size=len(units))
            bs.append(np.mean(np.concatenate([per[units[i]] for i in pick])))
        out[name] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return dict(mean_d=mean_d, mean_b=mean_b, rel=mean_d / mean_b, ci=out, n=len(keys),
                per_room={sc: float(np.mean(np.concatenate([d[k] for k in keys if k[0] == sc]))) for sc in sorted({k[0] for k in keys})})


def report(tag, cfg, bands=("2g4", "5g"), vs=None):
    res = load(tag)
    L = []
    P = L.append
    cfgs = sorted({c for (_k, (_m, o)) in res.items() for c in o})
    P(f"# real_{tag}.pkl  cfg={cfg}  (configs in file: {', '.join(cfgs)})   [simulation, not measured]")
    for band in bands:
        rows = bearing_rows(res, cfg, band, "four")
        if not rows:
            continue
        scenes = sorted({r["scene"] for r in rows}, key=R.REAL_SCENES.index)
        outside = [s for s in scenes if s not in R.EXAM]
        P(f"\n## {band}  single-spot bearing error (deg), four-combo spots, columns: mean median p90 p95 n")
        for sc in scenes:
            P(f"  {sc:13s} {stat(cat([r for r in rows if r['scene'] == sc]))}")
        ro = [r for r in rows if r["scene"] in outside]
        if ro:
            P(f"  {'OUTSIDE pooled':13s} {stat(cat(ro))}   <- G2 proxy (line: mean <= 8.92, p95 < 27.53)")
            for s in sorted({r['seed'] for r in ro}):
                P(f"     seed {s}      {stat(cat([r for r in ro if r['seed'] == s]))}")
            for kind in sorted({R.KIND[s] for s in outside}):
                P(f"     {kind:10s}  {stat(cat([r for r in ro if R.KIND[r['scene']] == kind]))}")
            P(f"     open cams   {stat(cat(ro, f=lambda r: ~r['conceal']))}")
            P(f"     hidden cams {stat(cat(ro, f=lambda r: r['conceal']))}")
            P(f"     same zone   {stat(cat(ro, f=lambda r: ~r['behind']))}")
            P(f"     behind part.{stat(cat(ro, f=lambda r: r['behind']))}")
        re_ = [r for r in rows if r["scene"] in R.EXAM]
        if re_:
            P(f"  {'EXAM pooled':13s} {stat(cat(re_))}")
        P(f"  {'ALL pooled':13s} {stat(cat(rows))}")
        ra = bearing_rows(res, cfg, band, "all")
        P("  by spot type (all spots, outside rooms):")
        for t in "WFTB":
            P(f"     {t}           {stat(cat([r for r in ra if r['type'] == t and r['scene'] in outside]))}")
        P(f"\n## {band}  position error (m): within 0.5 m / within 1 m / median   (line: >= 80 % within 0.5 m per kind)")
        for grp in ("four", "pair", "low", "sub"):
            pr = pos_rows(res, cfg, band, grp)
            if not pr:
                continue
            P(f"  group {grp}")
            for sc in scenes:
                v = cat([r for r in pr if r["scene"] == sc])
                if len(v):
                    P(f"     {sc:13s} {100 * np.mean(v <= 0.5):5.1f} % {100 * np.mean(v <= 1.0):5.1f} %  {np.median(v):5.2f} m  n={len(v)}")
            for kind in sorted({R.KIND[s] for s in scenes}):
                v = cat([r for r in pr if R.KIND[r["scene"]] == kind])
                if len(v):
                    P(f"     kind {kind:8s} {100 * np.mean(v <= 0.5):5.1f} % {100 * np.mean(v <= 1.0):5.1f} %  {np.median(v):5.2f} m  n={len(v)}")
        ex = [r for r in pos_rows(res, cfg, band, "pair") if r["seat3"] is not None]
        if ex:
            P(f"\n## {band}  exam rooms: suspect desk top-1 / top-3 (line: top-3 >= 80 %)")
            for grp in ("one", "pair", "four", "low"):
                for sc in [s for s in scenes if s in R.EXAM]:
                    rr = [r for r in pos_rows(res, cfg, band, grp) if r["scene"] == sc and r["seat3"] is not None]
                    if rr:
                        s1 = np.concatenate([r["seat1"] for r in rr]); s3 = np.concatenate([r["seat3"] for r in rr])
                        P(f"     {grp:5s} {sc:13s} top1 {100 * s1.mean():5.1f} %  top3 {100 * s3.mean():5.1f} %  n={len(s1)}")
        if vs:
            ocfg, otag = (vs.split("@") + [tag])[:2]
            ores = load(otag) if otag != tag else res
            P(f"\n## {band}  paired difference {cfg} - {ocfg}@{otag} (bearing, four-combo spots)")
            for name, sel in (("outside", outside), ("all", scenes)):
                b = room_boot([r for r in rows if r["scene"] in sel],
                              [r for r in bearing_rows(ores, ocfg, band, "four") if r["scene"] in sel])
                if b:
                    P(f"  {name:8s} mean diff {b['mean_d']:+6.2f} deg ({100 * b['rel']:+5.1f} % of {b['mean_b']:.2f}); room 95 % CI "
                      f"[{b['ci']['room'][0]:+.2f}, {b['ci']['room'][1]:+.2f}]; room x seed CI [{b['ci']['room_seed'][0]:+.2f}, {b['ci']['room_seed'][1]:+.2f}]"
                      f"; per room {', '.join(f'{k} {v:+.1f}' for k, v in b['per_room'].items())}")
    return "\n".join(L)


if __name__ == "__main__":
    tag = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "P0"
    vs = sys.argv[sys.argv.index("--vs") + 1] if "--vs" in sys.argv else None
    bands = tuple(sys.argv[sys.argv.index("--bands") + 1].split(",")) if "--bands" in sys.argv else ("2g4", "5g")
    txt = report(tag, cfg, bands, vs)
    print(txt)
    with open(os.path.join(ROOT, "results", f"real_summary_{tag}_{cfg}.txt"), "w") as f:
        f.write(txt + "\n")
