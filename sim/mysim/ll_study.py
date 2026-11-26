"""Offline software studies on stored likelihood maps (no new signal synthesis).
  reliability of one measurement (rules vs a learned model), "move when unsure" procedure, robust fusion of spots.
Development data: results/dev_ll_<cfg>_<band>.pkl (old rooms, eight wall spots).  Check data: results/real_<tag>.pkl with
maps (realistic rooms).  Nothing is tuned on the realistic rooms: thresholds and models come from the old rooms.
All numbers are simulation values."""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_scenes as R  # noqa: E402
import exp_wall as W  # noqa: E402

ROOT = "/opt/detecam/sim"
FEAT_NAMES = ["s1max", "s0max", "sdmax", "s1pk", "s0pk", "sdpk", "dl10", "fsel", "ringspread", "ringlvl"]


def azerr(a, b):
    d = np.degrees(a - b)
    return np.abs((d + 180.0) % 360.0 - 180.0)


def circ_spread(ll, az):
    p = np.exp(ll - ll.max())
    p /= p.sum()
    Rm = np.abs(np.sum(p * np.exp(1j * az)))
    return float(np.degrees(np.sqrt(max(-2.0 * np.log(max(Rm, 1e-9)), 0.0))))


def az_profile(ll, az, nb=72):
    """best log-likelihood per 5-degree azimuth bin"""
    b = ((np.degrees(az) % 360.0) / (360.0 / nb)).astype(int) % nb
    prof = np.full(nb, -1e9)
    np.maximum.at(prof, b, ll)
    return prof


def spot_table(res, cfg, spots_of, looks_of):
    """One row per (file, camera, spot): error of the summed-looks bearing + features.
    spots_of(scene) -> list of (label, base node index); looks_of(scene, i) -> node indices of that spot's looks."""
    rows = []
    for key in sorted(res):
        sc, band, seed = key
        mt, outs = res[key]
        if cfg not in outs or "ll_all" not in outs[cfg]:
            continue
        o = outs[cfg]
        LL = o["ll_all"].astype(np.float32)
        grid = o["grid"].astype(np.float64)
        npos = o["node_pos_est"].astype(np.float64)
        cams, nodes = mt["cams"], mt["nodes"]
        feat = o.get("extra", {}).get("feat")
        for lab, i in spots_of(sc):
            idx = looks_of(sc, i)
            v = grid - npos[i][None]
            az = np.arctan2(v[:, 1], v[:, 0])
            for r in range(len(cams)):
                lls = [LL[r, j] for j in idx]
                lt = np.sum(lls, axis=0)
                c = int(np.argmax(lt))
                vt = cams[r] - nodes[i]
                az_true = np.arctan2(vt[1], vt[0])
                err = float(azerr(az[c], az_true))
                az_l = np.array([az[int(np.argmax(l))] for l in lls])
                dis = float(max(azerr(az_l[a], az_l[b]) for a in range(len(idx)) for b in range(a + 1, len(idx)))) if len(idx) > 1 else 0.0
                dis_sum = float(np.max(azerr(az_l, az[c])))
                prof = az_profile(lt, az)
                kb = int(np.argmax(prof))
                far = np.ones(72, bool)
                for d in range(-4, 5):
                    far[(kb + d) % 72] = False
                gap = float(prof[kb] - prof[far].max())           # how far the best rival direction (> 20 deg away) falls short
                f = dict(scene=sc, seed=seed, cam=r, spot=lab, node=i, err=err, az=float(az[c]), est=grid[c],
                         spread=circ_spread(lt, az), spread_look=float(np.mean([o["spread"][r, j] for j in idx])),
                         dis=dis, dis_sum=dis_sum, gap=gap, llmax=float(np.mean([o["ll_max"][r, j] for j in idx])),
                         los=float(o["los_frac"][r, i]), dist=float(np.linalg.norm(vt)))
                if feat is not None:
                    F = feat[r, idx]                       # (looks, n_feat)
                    for k, nm in enumerate(FEAT_NAMES):
                        f[nm + "_mean"] = float(F[:, k].mean())
                        f[nm + "_min"] = float(F[:, k].min())
                rows.append(f)
    return rows


def dev_spots(sc):
    return [(W.LAB[i], i) for i in range(8)]


def dev_looks(sc, i):
    return [i, 16 + i, 32 + i]


def real_spots(which="four"):
    def f(sc):
        sp = R.SPOTS[sc]
        idx = range(len(sp)) if which == "all" else R.COMBO[sc][which]
        return [(sp[i][0], i) for i in idx]
    return f


def real_looks(sc, i):
    n = len(R.SPOTS[sc])
    return [i, n + i, 2 * n + i]


X_COLS = ["spread", "spread_look", "dis", "dis_sum", "gap", "llmax"] + [f"{n}_{s}" for n in FEAT_NAMES for s in ("mean", "min")]


def xmat(rows, cols=X_COLS):
    return np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)


def auc(score, bad):
    """probability that a bad measurement gets a higher (worse) score than a good one"""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(bad, score)) if 0 < bad.sum() < len(bad) else float("nan")


def keep_curve(score, err, fracs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    order = np.argsort(score)
    out = []
    for f in fracs:
        k = order[: max(1, int(round(f * len(err))))]
        out.append((f, float(err[k].mean()), float(np.percentile(err[k], 95)), float(np.mean(err[k] > 20))))
    return out


def move_policy(rows, score, thr, order_of, max_spots=4):
    """Per camera: measure at the first spot; accept if score <= thr, else move to the next spot (up to max_spots);
    if none is accepted take the lowest score.  Returns final errors and number of spots used."""
    by = {}
    for r, s in zip(rows, score):
        by.setdefault((r["scene"], r["seed"], r["cam"]), {})[r["spot"]] = (s, r["err"])
    errs, used = [], []
    for (sc, seed, cam), d in by.items():
        labs = [l for l in order_of(sc) if l in d][:max_spots]
        done = False
        for k, l in enumerate(labs):
            if d[l][0] <= thr:
                errs.append(d[l][1]); used.append(k + 1); done = True
                break
        if not done:
            best = min(labs, key=lambda l: d[l][0])
            errs.append(d[best][1]); used.append(len(labs))
    return np.array(errs), np.array(used)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_model(rows, bad_thr=20.0, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = xmat(rows)
    y = np.array([r["err"] > bad_thr for r in rows])
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, l2_regularization=1.0, random_state=seed)
    m.fit(X, y)
    return m


def loro_scores(rows, bad_thr=20.0):
    """leave-one-room-out scores on the development rooms (honest estimate inside the development set)"""
    sc = np.array([r["scene"] for r in rows])
    out = np.zeros(len(rows))
    for s in sorted(set(sc)):
        tr = [r for r in rows if r["scene"] != s]
        m = train_model(tr, bad_thr)
        k = np.where(sc == s)[0]
        out[k] = m.predict_proba(xmat([rows[i] for i in k]))[:, 1]
    return out


def report_reliability(dev_path, real_path, cfg="P0F", real_spot_set="four", bad_thr=20.0):
    L = []
    P = L.append
    dev = spot_table(load(dev_path), cfg, dev_spots, dev_looks)
    real = spot_table(load(real_path), cfg, real_spots(real_spot_set), real_looks)
    real_out = [r for r in real if r["scene"] not in R.EXAM]
    P(f"# reliability of one measurement (gross = error > {bad_thr:.0f} deg)  [simulation]")
    P(f"dev rows {len(dev)} (old rooms, 8 wall spots) gross {100*np.mean([r['err']>bad_thr for r in dev]):.1f} %, mean {np.mean([r['err'] for r in dev]):.2f}")
    P(f"real rows {len(real_out)} (realistic outside rooms, {real_spot_set} spots) gross {100*np.mean([r['err']>bad_thr for r in real_out]):.1f} %, mean {np.mean([r['err'] for r in real_out]):.2f}")
    e_dev = np.array([r["err"] for r in dev]); b_dev = e_dev > bad_thr
    e_re = np.array([r["err"] for r in real_out]); b_re = e_re > bad_thr
    model = train_model(dev, bad_thr)
    scores = {
        "rule: spread": (np.array([r["spread"] for r in dev]), np.array([r["spread"] for r in real_out])),
        "rule: looks disagree": (np.array([r["dis"] for r in dev]), np.array([r["dis"] for r in real_out])),
        "rule: -rival gap": (-np.array([r["gap"] for r in dev]), -np.array([r["gap"] for r in real_out])),
        "rule: -direct evidence": (-np.array([r["dl10_mean"] for r in dev]), -np.array([r["dl10_mean"] for r in real_out])),
        "rule: -s1 at peak": (-np.array([r["s1pk_mean"] for r in dev]), -np.array([r["s1pk_mean"] for r in real_out])),
        "learned (boosting, dev rooms)": (loro_scores(dev, bad_thr), model.predict_proba(xmat(real_out))[:, 1]),
    }
    P("\n## how well each score separates gross errors (AUC; 0.5 = useless)   dev (leave-one-room-out for the model) | realistic rooms")
    for nm, (sd, sr) in scores.items():
        P(f"  {nm:32s} {auc(sd, b_dev):.3f} | {auc(sr, b_re):.3f}")
    P("\n## keep the most reliable share of measurements: mean error / p95 / gross share (realistic rooms)")
    for nm, (sd, sr) in scores.items():
        P(f"  {nm:32s} " + "  ".join(f"{int(100*f)}%: {m:4.1f}/{p:4.1f}/{100*g:4.1f}%" for f, m, p, g in keep_curve(sr, e_re)))
    P("\n## 'move when unsure' procedure on the realistic rooms (threshold fixed on the dev rooms so that X % of dev measurements are accepted)")
    P("   columns: accepted on dev | final mean / median / p90 / p95 error | gross share | mean number of spots used")
    order_of = lambda sc: [R.SPOTS[sc][i][0] for i in R.COMBO[sc]["four"]]  # noqa: E731
    base = move_policy(real_out, np.zeros(len(real_out)), 1.0, order_of)
    P(f"  {'always first spot':32s}  -    {base[0].mean():5.2f} {np.median(base[0]):5.1f} {np.percentile(base[0],90):5.1f} {np.percentile(base[0],95):5.1f}  {100*np.mean(base[0]>20):4.1f}%  {base[1].mean():.2f}")
    for nm, (sd, sr) in scores.items():
        for acc in (0.9, 0.8, 0.7, 0.6):
            thr = np.quantile(sd, acc)
            e, u = move_policy(real_out, sr, thr, order_of)
            P(f"  {nm:32s} {int(100*acc):3d}%  {e.mean():5.2f} {np.median(e):5.1f} {np.percentile(e,90):5.1f} {np.percentile(e,95):5.1f}  {100*np.mean(e>20):4.1f}%  {u.mean():.2f}")
    # oracle bound: best of the four spots
    by = {}
    for r in real_out:
        by.setdefault((r["scene"], r["seed"], r["cam"]), []).append(r["err"])
    P(f"  {'(bound) best of the 4 spots':32s}  -    {np.mean([min(v) for v in by.values()]):5.2f}")
    imp = None
    try:
        from sklearn.inspection import permutation_importance
        Xr = xmat(real_out)
        pi = permutation_importance(model, xmat(dev), b_dev, n_repeats=3, random_state=0, scoring="roc_auc")
        imp = sorted(zip(X_COLS, pi.importances_mean), key=lambda t: -t[1])[:8]
    except Exception as ex:  # noqa: BLE001
        P(f"(importance failed: {ex})")
    if imp:
        P("\n## what the learned model looks at (permutation importance on dev rooms, top 8): " + ", ".join(f"{k} {v:.3f}" for k, v in imp))
    return "\n".join(L), (dev, real, model, scores)


def fuse_eval(res, cfg, groups_of, looks_of, clips=(None, 4.0, 3.0, 2.0, 1.5, 1.0), spot_clips=(None, 9.0, 6.0, 4.0, 3.0)):
    """Position error of fused spots for several robust fusion rules.  groups_of(scene) -> {name: [base node indices]}"""
    out = {}
    for key in sorted(res):
        sc, band, seed = key
        mt, outs = res[key]
        if cfg not in outs or "ll_all" not in outs[cfg]:
            continue
        o = outs[cfg]
        LL = o["ll_all"].astype(np.float32)
        grid = o["grid"].astype(np.float64)
        cams = mt["cams"]
        lp = o.get("logprior")
        for gname, base in groups_of(sc).items():
            for r in range(len(cams)):
                look = [[LL[r, j] for j in looks_of(sc, i)] for i in base]
                for c in clips:
                    lt = sum(np.maximum(l, -c) if c else l for sp in look for l in sp)
                    if lp is not None:
                        lt = lt + lp
                    pt = grid[int(np.argmax(lt))]
                    out.setdefault((gname, f"look_clip_{c}"), []).append((sc, seed, float(np.linalg.norm(pt[:2] - cams[r][:2]))))
                for c in spot_clips[1:]:
                    lt = sum(np.maximum(np.sum(sp, axis=0) - np.max(np.sum(sp, axis=0)), -c) for sp in look)
                    if lp is not None:
                        lt = lt + lp
                    pt = grid[int(np.argmax(lt))]
                    out.setdefault((gname, f"spot_clip_{c}"), []).append((sc, seed, float(np.linalg.norm(pt[:2] - cams[r][:2]))))
    return out


def report_fusion(dev_path, real_path, cfg="P0F"):
    L = []
    P = L.append
    dev = fuse_eval(load(dev_path), cfg, lambda sc: {"quadm": [4, 5, 6, 7], "quadc": [0, 1, 2, 3], "mopp": [4, 6]}, dev_looks)
    real = fuse_eval(load(real_path), cfg, lambda sc: {k: list(v) for k, v in R.COMBO[sc].items() if k in ("four", "pair")}, real_looks)
    P("# robust fusion of spots: share within 0.5 m / within 1 m / median (m)   [simulation]")
    for name, d, grp in (("dev old rooms quadm (4 wall-centre spots)", dev, "quadm"), ("dev old rooms quadc (4 corner spots)", dev, "quadc"),
                         ("dev old rooms mopp (2 spots)", dev, "mopp"),
                         ("realistic outside rooms four", real, "four"), ("realistic outside rooms pair", real, "pair")):
        P(f"\n## {name}")
        for (g, rule), v in d.items():
            if g != grp:
                continue
            e = np.array([x[2] for x in v if x[0] not in R.EXAM])
            if len(e) == 0:
                continue
            P(f"  {rule:16s} {100*np.mean(e<=0.5):5.1f} %  {100*np.mean(e<=1.0):5.1f} %  {np.median(e):5.2f} m   n={len(e)}")
    return "\n".join(L), (dev, real)


if __name__ == "__main__":
    what = sys.argv[1]
    band = sys.argv[2] if len(sys.argv) > 2 else "2g4"
    cfg = sys.argv[3] if len(sys.argv) > 3 else "P0F"
    devp = os.path.join(ROOT, "results", f"dev_ll_{cfg}_{band}.pkl")
    realp = os.path.join(ROOT, "results", sys.argv[4] if len(sys.argv) > 4 else "real_A_design_ll.pkl")
    if what == "rel":
        txt, _ = report_reliability(devp, realp, cfg)
    else:
        txt, _ = report_fusion(devp, realp, cfg)
    print(txt)
    with open(os.path.join(ROOT, "results", f"ll_{what}_{cfg}_{band}_{os.path.basename(realp)[5:-4]}.txt"), "w") as f:
        f.write(txt + "\n")
