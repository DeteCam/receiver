"""Full V1 switch-ring evaluation harness executing pre-registered receiver/sim/plan_20261125.md.
Strictly enforces:
  - Generator (gen_scenes.py) vs Estimator (estimators.py) separation
  - Dev split (classroom_a, lodging_a, restroom_a, locker_a) frozen BEFORE Held-Out Eval split
  - All 6 mandatory spaces + worst_metal reported individually AND overall (Mean/Median/P90/P95/Worst)
  - Apples-to-apples comparison: Bartlett, DiffLoc-Proxy, SPEC-002 (NaiveFlip & NoFlip), SPEC-002+, SPEC-002++
  - L0 -> L5 ablation ladder and failed/no-gain ablations (conj avg, gated-70% avg, reflective OPEN, ideal ant)
  - 5 GHz bandwidth comparison (20 / 40 / 80 MHz)
  - Incremental checkpoint saving per part so long evaluations resume cleanly

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py --part all
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py --part dev
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py --part eval_2g4
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py --part eval_5g
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py --part extra
"""
import argparse
import json
import os
import sys
import time
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from channel_sim import IdealDipoleArrayPattern, NEC2ArrayPattern, simulate_diffloc_proxy, synthesize_ring_csi
from estimators import RingMapEstimator, az_err_deg, bartlett_azimuth, fuse_two_nodes, make_room_grid

DEV_SCENES = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
EVAL_SCENES = [
    "lecture_a", "lecture_b",
    "bathhouse_a", "bathhouse_b",
    "classroom_b", "lodging_b", "restroom_b", "locker_b",
    "worst_metal",
]


def make_df(bw_hz):
    # Matches existing pe2.py subcarrier sampling (16 @ 20 MHz, 24 @ 40 MHz, 32 @ 80 MHz)
    k = 16 if bw_hz <= 20e6 else (24 if bw_hz <= 40e6 else 32)
    return np.linspace(-0.45 * bw_hz, 0.45 * bw_hz, k)


def stats_dict(errs):
    a = np.asarray(errs, dtype=np.float64)
    if len(a) == 0:
        return dict(n=0, mean=np.nan, med=np.nan, p90=np.nan, p95=np.nan, le10=np.nan)
    return dict(
        n=int(len(a)),
        mean=round(float(np.mean(a)), 2),
        med=round(float(np.median(a)), 2),
        p90=round(float(np.percentile(a, 90)), 2),
        p95=round(float(np.percentile(a, 95)), 2),
        le10=round(float(100.0 * np.mean(a <= 10.0)), 1),
    )


def evaluate_scene_file(
    npy_path,
    nec_db,
    methods,
    bw_hz=20e6,
    pose_err=True,
    snr_db=20.0,
    cal_phase_deg=4.5,
    cal_amp_db=0.4,
    dyn_frac=0.35,
    n_sweep=10,
    huber_delta=0.35,
):
    D = np.load(npy_path, allow_pickle=True).item()
    scene_id = D["scene_id"]
    band = D["band"]
    fc = float(D["fc"])
    seed = int(D["seed"])
    dims_true = np.asarray(D["dims"], dtype=np.float64)
    nodes_true = np.asarray(D["nodes"], dtype=np.float64)
    cams_true = np.asarray(D["cams"], dtype=np.float64)
    refl_true = D["user_reflectors"]
    paths = D["paths"]

    df_hz = make_df(bw_hz)
    dyn_std = 45.0 if band == "2g4" else 90.0
    rng_pose = np.random.default_rng(7000 + seed * 31 + abs(hash(scene_id)) % 89)
    rng_sig = np.random.default_rng(9000 + seed * 53 + abs(hash(scene_id)) % 97)

    if pose_err:
        dims_est = np.maximum(dims_true + rng_pose.normal(0.0, [0.15, 0.15, 0.075]), [2.0, 2.0, 1.8])
        node_pos_est = nodes_true + rng_pose.normal(0.0, [0.15, 0.15, 0.075], nodes_true.shape)
        heading_est = np.radians(rng_pose.normal(0.0, 5.0, len(nodes_true)))
        refl_est = []
        for ax, val, lo, hi, rf in refl_true:
            dv = float(rng_pose.normal(0.0, 0.15))
            refl_est.append((ax, float(val + dv), lo, hi, rf))
    else:
        dims_est = dims_true.copy()
        node_pos_est = nodes_true.copy()
        heading_est = np.zeros(len(nodes_true), dtype=np.float64)
        refl_est = list(refl_true)

    grid = make_room_grid(dims_est, max_xy_pts=260)

    pat_cache = {}
    def get_pat(key):
        if key not in pat_cache:
            pat_cache[key] = IdealDipoleArrayPattern(8, 100.0, 0.0, fc) if key == "ideal_8_dz0" else NEC2ArrayPattern(nec_db[(key, band)], fc)
        return pat_cache[key]

    # Share identical RingMapEstimator instances across methods (e.g. NaiveFlip & NoFlip, or Ratio/Conj/Gated70)
    map_est_pool = {}
    est_cache = {}
    for mname, mcfg in methods.items():
        if mcfg["kind"] in ("bartlett", "diffloc"):
            est_cache[mname] = get_pat(mcfg["pat"]) if mcfg["kind"] == "bartlett" else None
            continue
        sig_key = (
            mcfg["pat"], bool(mcfg["include_ref"]), bool(mcfg.get("use_reflectors", False)),
            bool(mcfg.get("second_order", True)), bool(mcfg.get("robust_huber", False)),
        )
        if sig_key not in map_est_pool:
            pat = get_pat(mcfg["pat"])
            node_ests = [
                RingMapEstimator(
                    grid=grid,
                    node_pos_est=node_pos_est[n],
                    dims_est=dims_est,
                    cal_pat=pat,
                    df_hz=df_hz,
                    heading_est=float(heading_est[n]),
                    include_ref=mcfg["include_ref"],
                    reflector_faces=refl_est if mcfg.get("use_reflectors", False) else (),
                    second_order=mcfg.get("second_order", True),
                    robust_huber=mcfg.get("robust_huber", False),
                    huber_delta=huber_delta,
                )
                for n in range(len(nodes_true))
            ]
            map_est_pool[sig_key] = (pat, node_ests)
        est_cache[mname] = map_est_pool[sig_key]

    out = {
        m: dict(
            one_node=[],
            by_node={n: [] for n in range(len(nodes_true))},
            fused_adj=[],
            fused_diag=[],
            sign_accs=[],
            spreads=[],
        )
        for m in methods
    }

    n_cams = len(cams_true)
    n_nodes = len(nodes_true)

    for r in range(n_cams):
        cam = cams_true[r]
        for mname, mcfg in methods.items():
            kind = mcfg["kind"]
            if kind == "diffloc":
                for n in range(n_nodes):
                    paths_rn = (paths["a"][r, n, 0], paths["a"][r, n, 1], paths["tau"][r, n], paths["th"][r, n], paths["ph"][r, n])
                    az_est = simulate_diffloc_proxy(
                        paths_rn, cam, nodes_true[n], df_hz, fc, rng_sig,
                        dyn_frac=dyn_frac, dyn_std_deg=dyn_std, snr_db=snr_db,
                    ) - float(heading_est[n])
                    vt = cam - nodes_true[n]
                    e = az_err_deg(az_est, float(np.arctan2(vt[1], vt[0])))
                    out[mname]["one_node"].append(e)
                    out[mname]["by_node"][n].append(e)
                continue

            if kind == "bartlett":
                pat = est_cache[mname]
                for n in range(n_nodes):
                    paths_rn = (paths["a"][r, n, 0], paths["a"][r, n, 1], paths["tau"][r, n], paths["th"][r, n], paths["ph"][r, n])
                    X, st = synthesize_ring_csi(
                        paths_rn, cam, nodes_true[n], pat, df_hz, rng_sig,
                        n_sweep=n_sweep, dyn_frac=dyn_frac, dyn_std_deg=dyn_std,
                        cal_phase_deg=cal_phase_deg, cal_amp_db=cal_amp_db, snr_db=snr_db,
                        flip_mode=mcfg.get("flip_mode", "sign_anchor"),
                        agg_mode=mcfg.get("agg_mode", "ratio"),
                        include_ref=mcfg.get("include_ref", False),
                        weight_by_ref_amp=True,
                    )
                    az_est = bartlett_azimuth(X, node_pos_est[n], dims_est, pat, heading_est=float(heading_est[n]), include_ref=mcfg.get("include_ref", False))
                    vt = cam - nodes_true[n]
                    e = az_err_deg(az_est, float(np.arctan2(vt[1], vt[0])))
                    out[mname]["one_node"].append(e)
                    out[mname]["by_node"][n].append(e)
                    out[mname]["sign_accs"].append(st["sign_acc"])
                continue

            pat, node_ests = est_cache[mname]
            lls, sps = [], []
            for n in range(n_nodes):
                paths_rn = (paths["a"][r, n, 0], paths["a"][r, n, 1], paths["tau"][r, n], paths["th"][r, n], paths["ph"][r, n])
                X, st = synthesize_ring_csi(
                    paths_rn, cam, nodes_true[n], pat, df_hz, rng_sig,
                    n_sweep=n_sweep, dyn_frac=dyn_frac, dyn_std_deg=dyn_std,
                    cal_phase_deg=cal_phase_deg, cal_amp_db=cal_amp_db, snr_db=snr_db,
                    flip_mode=mcfg["flip_mode"],
                    agg_mode=mcfg["agg_mode"],
                    include_ref=mcfg["include_ref"],
                    weight_by_ref_amp=mcfg.get("weight_by_ref_amp", True),
                )
                ll, sp = node_ests[n].log_likelihood(X)
                az_est, _ = node_ests[n].estimate_azimuth(ll)
                vt = cam - nodes_true[n]
                e = az_err_deg(az_est, float(np.arctan2(vt[1], vt[0])))
                out[mname]["one_node"].append(e)
                out[mname]["by_node"][n].append(e)
                out[mname]["sign_accs"].append(st["sign_acc"])
                out[mname]["spreads"].append(sp)
                lls.append(ll)
                sps.append(sp)

            conf_w = mcfg.get("conf_weight", True)
            for key_f, (i0, i1) in (("fused_adj", (0, 1)), ("fused_diag", (0, 3))):
                _, az_pair, _ = fuse_two_nodes(
                    [lls[i0], lls[i1]], [sps[i0], sps[i1]], grid,
                    [node_pos_est[i0], node_pos_est[i1]], confidence_weighted=conf_w,
                )
                for idx_local, ni in enumerate((i0, i1)):
                    vt = cam - nodes_true[ni]
                    out[mname][key_f].append(az_err_deg(az_pair[idx_local], float(np.arctan2(vt[1], vt[0]))))

    return out


def aggregate_runs(run_outputs, methods):
    agg = {}
    for m in methods:
        one = np.concatenate([o[m]["one_node"] for o in run_outputs])
        by_node_means = [float(np.mean(np.concatenate([o[m]["by_node"][n] for o in run_outputs]))) for n in range(4)]
        by_node_p95s = [float(np.percentile(np.concatenate([o[m]["by_node"][n] for o in run_outputs]), 95)) for n in range(4)]
        worst_node_idx = int(np.argmax(by_node_means))
        f_adj = np.concatenate([o[m]["fused_adj"] for o in run_outputs]) if run_outputs[0][m]["fused_adj"] else np.array([])
        f_diag = np.concatenate([o[m]["fused_diag"] for o in run_outputs]) if run_outputs[0][m]["fused_diag"] else np.array([])
        sa = np.concatenate([o[m]["sign_accs"] for o in run_outputs]) if run_outputs[0][m]["sign_accs"] else np.array([1.0])
        sp = np.concatenate([o[m]["spreads"] for o in run_outputs]) if run_outputs[0][m]["spreads"] else np.array([])
        unrel_rate = float(100.0 * np.mean(sp > 45.0)) if len(sp) > 0 else np.nan
        agg[m] = dict(
            one_node=stats_dict(one),
            worst_pos=dict(node=worst_node_idx, mean=round(by_node_means[worst_node_idx], 2), p95=round(by_node_p95s[worst_node_idx], 2)),
            fused_adj=stats_dict(f_adj),
            fused_diag=stats_dict(f_diag),
            sign_acc_pct=round(float(100.0 * np.mean(sa)), 2),
            unreliable_pct=round(unrel_rate, 1) if np.isfinite(unrel_rate) else None,
        )
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "dev", "eval_2g4", "eval_5g", "extra", "posthoc"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    data_dir = os.path.join(THIS_DIR, "data")
    res_dir = os.path.join(THIS_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    out_json = args.out if args.out else os.path.join(res_dir, "v1_summary.json")
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = dict(dev={}, eval={}, eval_overall={}, ladder={}, ablations={}, bw5g={})

    nec_path = os.path.join(os.path.dirname(THIS_DIR), "em", "data", "ring_nec2_patterns.npy")
    nec_db = np.load(nec_path, allow_pickle=True).item()

    PRIMARY_METHODS = {
        "Bartlett": dict(kind="bartlett", pat="spec002plus_8_dz0_abs", include_ref=False, flip_mode="sign_anchor", agg_mode="ratio"),
        "DiffLoc-Proxy": dict(kind="diffloc"),
        "SPEC-002-NaiveFlip": dict(
            kind="map", pat="spec002_dz65_abs", include_ref=False, flip_mode="naive_flip",
            agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False,
        ),
        "SPEC-002-NoFlip": dict(
            kind="map", pat="spec002_dz65_abs", include_ref=False, flip_mode="none",
            agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False,
        ),
        "SPEC-002+": dict(
            kind="map", pat="spec002plus_8_dz0_abs", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
        "SPEC-002++(12+1)": dict(
            kind="map", pat="spec002pp_12_dz0_abs", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
    }

    ABLATION_METHODS = {
        "SPEC-002+ (ref)": PRIMARY_METHODS["SPEC-002+"],
        "Abl: Conj-Avg (biased)": dict(
            kind="map", pat="spec002plus_8_dz0_abs", include_ref=True, flip_mode="sign_anchor",
            agg_mode="conj", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
        "Abl: Gated70-Ratio": dict(
            kind="map", pat="spec002plus_8_dz0_abs", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio_gated70", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
        "Abl: Reflect-OPEN-SW": dict(
            kind="map", pat="spec002plus_8_dz0_open", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
        "Abl: Ideal-Ant (upper)": dict(
            kind="map", pat="ideal_8_dz0", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
        "Abl: No-Reflector-Map": dict(
            kind="map", pat="spec002plus_8_dz0_abs", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=False, robust_huber=True, conf_weight=True,
        ),
        "Abl: D=85mm (8+1)": dict(
            kind="map", pat="spec002plus_d85_dz0", include_ref=True, flip_mode="sign_anchor",
            agg_mode="ratio", use_reflectors=True, robust_huber=True, conf_weight=True,
        ),
    }

    def save_now():
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def run_split(split_name, scene_list, bands):
        for band in bands:
            all_outs = []
            for sc in scene_list:
                sc_outs = []
                for s in (0, 1, 2):
                    fp = os.path.join(data_dir, f"{sc}_{band}_s{s}.npy")
                    if os.path.exists(fp):
                        sc_outs.append(evaluate_scene_file(fp, nec_db, PRIMARY_METHODS, bw_hz=20e6))
                if sc_outs:
                    agg = aggregate_runs(sc_outs, PRIMARY_METHODS)
                    results[split_name].setdefault(band, {})[sc] = agg
                    all_outs.extend(sc_outs)
                    sp = agg["SPEC-002+"]
                    dfp = agg["DiffLoc-Proxy"]
                    s2 = agg["SPEC-002-NoFlip"]
                    print(
                        f"[{split_name.upper():4s}] {sc:12s} {band:3s} | "
                        f"DiffLocP={dfp['one_node']['mean']:5.1f}° | "
                        f"S002={s2['one_node']['mean']:5.1f}°(2p {s2['fused_adj']['mean']:4.1f}°) | "
                        f"S002+={sp['one_node']['mean']:5.1f}°(adj {sp['fused_adj']['mean']:4.1f}° P95 {sp['fused_adj']['p95']:4.1f}°, "
                        f"diag {sp['fused_diag']['mean']:4.1f}°, sign {sp['sign_acc_pct']:.2f}%)",
                        flush=True,
                    )
                    save_now()
            if all_outs and split_name == "eval":
                results["eval_overall"][band] = aggregate_runs(all_outs, PRIMARY_METHODS)
                save_now()

    if args.part in ("all", "dev"):
        run_split("dev", DEV_SCENES, ("2g4", "5g"))
    if args.part in ("all", "eval_2g4"):
        run_split("eval", EVAL_SCENES, ("2g4",))
    if args.part in ("all", "eval_5g"):
        run_split("eval", EVAL_SCENES, ("5g",))

    if args.part in ("all", "extra"):
        # Ladder L0 -> L5 on classroom_a (dev) & classroom_b (eval)
        for sc in ("classroom_a", "classroom_b"):
            for band in ("2g4", "5g"):
                lad = {}
                l0, l1, l2, l3, l4, l5 = [], [], [], [], [], []
                for s in (0, 1, 2):
                    f_l0 = os.path.join(data_dir, f"{sc}_l0_{band}_s{s}.npy")
                    f_l1 = os.path.join(data_dir, f"{sc}_l1_{band}_s{s}.npy")
                    f_full = os.path.join(data_dir, f"{sc}_{band}_s{s}.npy")
                    if os.path.exists(f_l0):
                        m0 = {"L0": dict(kind="map", pat="ideal_8_dz0", include_ref=False, flip_mode="none", agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False)}
                        l0.append(evaluate_scene_file(f_l0, nec_db, m0, pose_err=False, snr_db=99.0, cal_phase_deg=0.0, cal_amp_db=0.0, dyn_frac=0.0, n_sweep=1))
                    if os.path.exists(f_l1):
                        m1 = {"L1": dict(kind="map", pat="ideal_8_dz0", include_ref=False, flip_mode="none", agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False)}
                        l1.append(evaluate_scene_file(f_l1, nec_db, m1, pose_err=False, snr_db=99.0, cal_phase_deg=0.0, cal_amp_db=0.0, dyn_frac=0.0, n_sweep=1))
                    if os.path.exists(f_full):
                        m2 = {"L2": dict(kind="map", pat="ideal_8_dz0", include_ref=False, flip_mode="none", agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False)}
                        l2.append(evaluate_scene_file(f_full, nec_db, m2, pose_err=False, snr_db=99.0, cal_phase_deg=0.0, cal_amp_db=0.0, dyn_frac=0.35, n_sweep=10))
                        m3 = {"L3": dict(kind="map", pat="spec002_dz65_abs", include_ref=False, flip_mode="none", agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False)}
                        l3.append(evaluate_scene_file(f_full, nec_db, m3, pose_err=False, snr_db=99.0, cal_phase_deg=0.0, cal_amp_db=0.0, dyn_frac=0.35, n_sweep=10))
                        m4 = {"L4": dict(kind="map", pat="spec002_dz65_abs", include_ref=False, flip_mode="sign_anchor", agg_mode="ratio", use_reflectors=False, robust_huber=False, conf_weight=False)}
                        l4.append(evaluate_scene_file(f_full, nec_db, m4, pose_err=True, snr_db=20.0, cal_phase_deg=4.5, cal_amp_db=0.4, dyn_frac=0.35, n_sweep=10))
                        m5 = {"L5": PRIMARY_METHODS["SPEC-002+"]}
                        l5.append(evaluate_scene_file(f_full, nec_db, m5, pose_err=True, snr_db=20.0, cal_phase_deg=4.5, cal_amp_db=0.4, dyn_frac=0.35, n_sweep=10))
                for k, outs in (("L0", l0), ("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4), ("L5", l5)):
                    if outs:
                        lad[k] = aggregate_runs(outs, {k: None})[k]
                results["ladder"][f"{sc}_{band}"] = lad
                print(f"[LADDER] {sc} {band} done", flush=True)
                save_now()

        # Failed / no-gain ablations on 4 eval scenes
        for band in ("2g4", "5g"):
            ab_outs = []
            for sc in ("classroom_b", "lecture_a", "bathhouse_a", "worst_metal"):
                for s in (0, 1, 2):
                    fp = os.path.join(data_dir, f"{sc}_{band}_s{s}.npy")
                    if os.path.exists(fp):
                        ab_outs.append(evaluate_scene_file(fp, nec_db, ABLATION_METHODS, bw_hz=20e6))
            if ab_outs:
                results["ablations"][band] = aggregate_runs(ab_outs, ABLATION_METHODS)
                print(f"[ABLATIONS] {band} done", flush=True)
                save_now()

        # 5 GHz BW comparison (20 / 40 / 80 MHz)
        for bw_mhz in (20, 40, 80):
            bw_outs = []
            for sc in ("classroom_b", "lecture_a", "bathhouse_a", "lodging_b"):
                for s in (0, 1, 2):
                    fp = os.path.join(data_dir, f"{sc}_5g_s{s}.npy")
                    if os.path.exists(fp):
                        bw_outs.append(evaluate_scene_file(fp, nec_db, {"SPEC-002+": PRIMARY_METHODS["SPEC-002+"]}, bw_hz=bw_mhz * 1e6))
            if bw_outs:
                results["bw5g"][f"{bw_mhz}MHz"] = aggregate_runs(bw_outs, {"SPEC-002+": None})["SPEC-002+"]
                save_now()
        print("[BW5G] done", flush=True)

    if args.part in ("all", "posthoc"):
        POSTHOC_METHODS = {
            "PostHoc: D85+Conj+6Wall": dict(
                kind="map", pat="spec002plus_d85_dz0", include_ref=True, flip_mode="sign_anchor",
                agg_mode="conj", use_reflectors=False, robust_huber=True, conf_weight=True,
            ),
            "PostHoc: D100+Conj+6Wall": dict(
                kind="map", pat="spec002plus_8_dz0_abs", include_ref=True, flip_mode="sign_anchor",
                agg_mode="conj", use_reflectors=False, robust_huber=True, conf_weight=True,
            ),
        }
        results.setdefault("posthoc", {})
        for split_name, sc_list in (("dev", DEV_SCENES), ("eval", EVAL_SCENES)):
            for band in ("2g4", "5g"):
                all_outs = []
                for sc in sc_list:
                    sc_outs = []
                    for s in (0, 1, 2):
                        fp = os.path.join(data_dir, f"{sc}_{band}_s{s}.npy")
                        if os.path.exists(fp):
                            sc_outs.append(evaluate_scene_file(fp, nec_db, POSTHOC_METHODS, bw_hz=20e6))
                    if sc_outs:
                        agg = aggregate_runs(sc_outs, POSTHOC_METHODS)
                        results["posthoc"].setdefault(f"{split_name}_scenes", {}).setdefault(band, {})[sc] = agg
                        all_outs.extend(sc_outs)
                if all_outs:
                    overall = aggregate_runs(all_outs, POSTHOC_METHODS)
                    results["posthoc"].setdefault(f"{split_name}_overall", {})[band] = overall
                    d85 = overall["PostHoc: D85+Conj+6Wall"]
                    d100 = overall["PostHoc: D100+Conj+6Wall"]
                    print(
                        f"[POSTHOC {split_name:4s} {band:3s}] "
                        f"D85+Conj: 1p={d85['one_node']['mean']:.1f}°(P95 {d85['one_node']['p95']:.1f}°) "
                        f"adj={d85['fused_adj']['mean']:.1f}°(P95 {d85['fused_adj']['p95']:.1f}°) "
                        f"diag={d85['fused_diag']['mean']:.1f}°(P95 {d85['fused_diag']['p95']:.1f}°) | "
                        f"D100+Conj: 1p={d100['one_node']['mean']:.1f}° adj={d100['fused_adj']['mean']:.1f}° diag={d100['fused_diag']['mean']:.1f}°",
                        flush=True,
                    )
                    save_now()

    print(f"\nPart '{args.part}' COMPLETE in {time.time() - t0:.1f}s -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
