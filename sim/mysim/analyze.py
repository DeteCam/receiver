"""Summaries and paired bootstrap confidence intervals for results produced by my_eval.py."""
import pickle
import sys
import numpy as np

DEV_SCENES = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
EVAL_SCENES = ["lecture_a", "lecture_b", "bathhouse_a", "bathhouse_b", "classroom_b", "lodging_b",
               "restroom_b", "locker_b", "worst_metal"]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect(res, cfg, band, scenes, metric, seeds=None):
    """Return (E, scene_idx) where E is (units, items): one row per camera sample.
    metric: 'one' | 'adj_spread' | 'adj_equal' | 'diag_spread' | 'diag_equal' | 'pos_diag_equal' ..."""
    rows, sc_idx = [], []
    for si, sc in enumerate(scenes):
        for key in sorted(k for k in res if k[0] == sc and k[1] == band and (seeds is None or k[2] in seeds)):
            o = res[key][1].get(cfg)
            if o is None:
                continue
            if metric == "one":
                e = o["one"]
            elif metric.startswith("pos_"):
                e = o["pos_err"][metric[4:]][:, None]
            else:
                e = o["fused"][metric]
            rows.append(np.asarray(e, dtype=np.float64))
            sc_idx.append(np.full(len(e), si))
    if not rows:
        return None, None
    return np.vstack(rows), np.concatenate(sc_idx)


def stat4(e):
    e = np.asarray(e).ravel()
    return float(np.mean(e)), float(np.median(e)), float(np.percentile(e, 90)), float(np.percentile(e, 95))


def fmt4(e):
    m, md, p90, p95 = stat4(e)
    return f"{m:5.2f} / {md:4.1f} / {p90:4.1f} / {p95:4.1f}"


def summary_table(res, cfgs, band, scenes, weighting="spread"):
    lines = []
    for c in cfgs:
        one, sci = collect(res, c, band, scenes, "one")
        if one is None:
            continue
        adj, _ = collect(res, c, band, scenes, f"adj_{weighting}")
        diag, sci2 = collect(res, c, band, scenes, f"diag_{weighting}")
        worst = max(range(len(scenes)), key=lambda i: diag[sci2 == i].mean() if np.any(sci2 == i) else -1)
        wd = diag[sci2 == worst]
        node_means = one.mean(axis=0)
        lines.append(f"| `{c}` | {fmt4(one)} | {fmt4(adj)} | {fmt4(diag)} | `{scenes[worst]}` {wd.mean():.1f}° ({np.percentile(wd,95):.1f}°) | "
                     f"자리{int(np.argmax(node_means))} {node_means.max():.1f}° |")
    return "\n".join(lines)


def paired_boot(res, cfg_a, cfg_b, band, scenes, metric, n_boot=2000, seed=1, level="camera"):
    """Difference (A - B) in mean and P95 with a paired bootstrap.
    level='camera': resample camera samples (rooms treated as fixed).
    level='scene' : resample whole rooms (9 clusters) -> how well the difference generalises to new rooms."""
    A, sci = collect(res, cfg_a, band, scenes, metric)
    B, _ = collect(res, cfg_b, band, scenes, metric)
    if A is None or B is None or A.shape != B.shape:
        return None
    rng = np.random.default_rng(seed)
    U = len(A)
    d_mean, d_p95 = [], []
    by_scene = [np.where(sci == i)[0] for i in range(len(scenes))]
    by_scene = [b for b in by_scene if len(b)]
    for _ in range(n_boot):
        if level == "camera":
            idx = rng.integers(0, U, U)
        else:
            pick = rng.integers(0, len(by_scene), len(by_scene))
            idx = np.concatenate([by_scene[p] for p in pick])
        a, b = A[idx].ravel(), B[idx].ravel()
        d_mean.append(a.mean() - b.mean())
        d_p95.append(np.percentile(a, 95) - np.percentile(b, 95))
    d_mean, d_p95 = np.array(d_mean), np.array(d_p95)
    per_scene = [(A[b].mean() - B[b].mean()) for b in by_scene]
    return dict(
        mean_a=float(A.mean()), mean_b=float(B.mean()), d_mean=float(A.mean() - B.mean()),
        d_mean_ci=(float(np.percentile(d_mean, 2.5)), float(np.percentile(d_mean, 97.5))),
        p95_a=float(np.percentile(A, 95)), p95_b=float(np.percentile(B, 95)),
        d_p95=float(np.percentile(A, 95) - np.percentile(B, 95)),
        d_p95_ci=(float(np.percentile(d_p95, 2.5)), float(np.percentile(d_p95, 97.5))),
        scenes_a_better=int(np.sum(np.array(per_scene) < 0)), n_scenes=len(per_scene),
        per_scene=[round(float(x), 2) for x in per_scene],
    )


def fmt_cmp(r):
    sig_m = "" if (r["d_mean_ci"][0] <= 0 <= r["d_mean_ci"][1]) else " *"
    sig_p = "" if (r["d_p95_ci"][0] <= 0 <= r["d_p95_ci"][1]) else " *"
    return (f"평균 {r['mean_a']:.2f} vs {r['mean_b']:.2f} (차 {r['d_mean']:+.2f}°, 95% CI {r['d_mean_ci'][0]:+.2f}~{r['d_mean_ci'][1]:+.2f}{sig_m}) | "
            f"P95 {r['p95_a']:.1f} vs {r['p95_b']:.1f} (차 {r['d_p95']:+.1f}°, CI {r['d_p95_ci'][0]:+.1f}~{r['d_p95_ci'][1]:+.1f}{sig_p}) | "
            f"A가 나은 방 {r['scenes_a_better']}/{r['n_scenes']}")


if __name__ == "__main__":
    res = load(sys.argv[1] if len(sys.argv) > 1 else "/opt/detecam/sim/results/rev2.pkl")
    cfgs = ["BASE_D100_Conj", "EXACT_D100_MAP", "BASE_D85_Conj", "EXACT_D85_MAP", "D75_EXACT_MAP", "S5CAL_D85_MAP", "S5CAL_D100_MAP"]
    for band in ("2g4", "5g"):
        print(f"\n### {band} eval 9 (fusion weighting 1/spread)")
        print(summary_table(res, cfgs, band, EVAL_SCENES, "spread"))
