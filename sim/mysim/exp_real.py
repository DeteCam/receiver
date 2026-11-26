"""Realistic rooms (plan sim/plan_20261126_real.md): the present design and the candidates.
Usage: python3 mysim/exp_real.py <set A|B|scene list> <cond design|real> <bands> <seeds> <cfg names> [tag]
  outside rooms: design -> data_real/nop, real -> data_real/op1 ; exam rooms -> data_real/base
Results: results/real_<tag>.pkl  {(scene, band, seed): (meta, {cfg: out})}"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E
import exp_final as F
import real_scenes as R
from risk_synth import synth_risk

ROOT = "/opt/detecam/sim"
NEC_REAL = os.path.join(ROOT, "mysim", "nec_real.npy")
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
# hardware / driver risks of the "real" condition, each with its software counter-measure switched on
REAL_KW = dict(p_swap=0.10, swap_fix="energy",
               agc=dict(mode="indep", step_db=2.0, ph_sig_deg=7.0, amp_sig_db=0.3, lvl_jit_db=2.0, raw=True, fix="table"),
               p_mis=0.05, mis_guard=True, agg="trim", leak_db=-30.0, alpha_err_deg=6.0, alpha_err_db=0.5)
# "real2" (POST-HOC, after the first results of "real"): the chain-order counter-measure corrected for the H4 anchor.
# The energy rule of "real" assumes an anchor stronger than the reference chain; with H4 it is 3.5 dB weaker.
# real2 = AX210 orders the CSI rows by RSS, a 6 dB attenuator at the chain-B input, rule "stronger row = reference".
REAL2_KW = dict(order_model="rss", order_proc="naive", sw_pad_db=6.0,
                agc=dict(mode="indep", step_db=2.0, ph_sig_deg=7.0, amp_sig_db=0.3, lvl_jit_db=2.0, raw=True, fix="table"),
                p_mis=0.05, mis_guard=True, agg="trim", leak_db=-30.0, alpha_err_deg=6.0, alpha_err_db=0.5)
E.EXAM_EXTRA = ("exam_s", "seminar_g")

# name -> (pattern base key, looks, extra cfg).  looks: "nod" (three looks, maps summed) | "one" | "poses"
DESIGNS = {
    "P0": ("d120_16", "nod", {}),
    "P0_one": ("d120_16", "one", {}),
    "P1": ("d190zz25", "poses", dict(poses=[(None, 7.5), (None, 15.0)])),
    "P2": ("d190zz40", "poses", dict(poses=[(None, 7.5), (None, 15.0)])),
    "P3": ("d170zz25", "poses", dict(poses=[(None, 7.5), (None, 15.0)])),
    "P4": ("d190zz25", "poses", dict(poses=[(None, 11.25)])),
    "P5": ("d190_16", "nod", {}),
    "P6": ("d190zz25", "one", {}),
    "P7": ("d120_16", "poses", dict(poses=[(None, 7.5), (None, 15.0)])),
    # software-only variants of P0
    "SW1": ("d120_16", "nod", dict(use_reflectors=True)),
    "SW3": ("d120_16", "poses", dict(poses=[("d120_16_H4_t+20", 0.0), ("d120_16_H4_t-20", 0.0)], pose_err_deg=0.0)),
    # estimator-model variants of P0 (POST-HOC additions after the first set-A results)
    "E_noimg": ("d120_16", "nod", dict(second_order=False, est_kwargs=dict(refl_wall=1e-3))),   # direct path only
    "E_1st": ("d120_16", "nod", dict(second_order=False)),                                        # first-order wall images only
    "E_rw03": ("d120_16", "nod", dict(est_kwargs=dict(refl_wall=0.3))),
    "E_rw07": ("d120_16", "nod", dict(est_kwargs=dict(refl_wall=0.7))),
    "E_wv5": ("d120_16", "nod", dict(est_cls="EstWV", est_kwargs=dict(w_vis=0.5))),
    "E_wv9": ("d120_16", "nod", dict(est_cls="EstWV", est_kwargs=dict(w_vis=0.9))),
    "G_z5": ("d120_16", "nod", dict(grid=("cart", 0.30, (0.08, 0.30, 0.55, 0.80, 0.97), 260))),   # floor / ceiling layers
    "G_polar": ("d120_16", "nod", dict(grid=("polar",))),
    "C_surf": ("d120_16", "nod", dict(cand="surface_furn", prior_logeps=-1.0, reg_sig=0.10)),     # cameras sit on surfaces
    "P0F": ("d120_16", "nod", dict(est_cls="EstFeat")),      # P0 with evidence values stored (same numbers as P0)
    "P5F": ("d190_16", "nod", dict(est_cls="EstFeat")),
    # structure + nod mixes (POST-HOC): bigger rings keep the nod instead of the turn
    "P8": ("d190zz25", "nod", {}),
    "P9": ("d160_16", "nod", {}),          # D160 flat ring + nod
    "P10": ("d150_16", "nod", {}),         # D150 flat ring + nod
}


def build_nec_real(extra_files=("nec_cand.npy",)):
    """H4 tables (ring path gain g applied to the switched elements) for every head that is available."""
    db = {}
    for f in ("nec_ring.npy", "nec_wall.npy") + tuple(extra_files):
        fp = os.path.join(ROOT, "mysim", f)
        if os.path.exists(fp):
            db.update(np.load(fp, allow_pickle=True).item())
    out = {}
    for (key, band), d in db.items():
        if not (key.startswith(("d120_16", "d190", "d170", "d160_16", "d150")) and "_r" not in key):
            continue
        for h in (("H4", "H1", "H5") if key.startswith("d120_16") else ("H4",)):
            alpha, g, snr = F.ANCHOR[h][band]
            d2 = dict(d)
            d2["eth"] = (d["eth"] * g).astype(np.complex64)
            d2["eph"] = (d["eph"] * g).astype(np.complex64)
            if "_ty" in key:
                base, t = key.split("_ty")
                name = f"{base}_{h}_t{t}"
            else:
                name = f"{key}_{h}"
            out[(name, band)] = d2
    tmp = NEC_REAL + ".tmp.npy"
    np.save(tmp, out, allow_pickle=True)
    os.replace(tmp, NEC_REAL)          # atomic: running evaluations never read a half-written table
    return sorted(out)


def cond_cfg(cond, band, exam, pat):
    alpha, g, snr = F.ANCHOR["H4"][band]
    c = dict(S5, pat=pat, anchor_alpha=alpha, snr_db=snr)
    if exam:
        c.update(cand="auto", prior_logeps=-1.0, reg_sig=0.10)
    else:
        c.update(dyn_frac=0.0)
    if cond in ("real", "real2"):
        c.update(snr_db=snr - 4.0, cal_phase_deg=8.0, cal_amp_db=0.6, sig_xy=0.08, sig_z=0.04, sig_head=3.0, n_sweep=6,
                 synth_fn=synth_risk, synth_kw=dict(REAL_KW if cond == "real" else REAL2_KW))
        if exam:
            c.update(reg_sig=0.15)
        else:
            c.update(dyn_frac=0.20)
    return c


def setup_scene(sc, looks, pat):
    spots = R.SPOTS[sc]
    n = len(spots)
    yaw = [float(np.radians(s[4])) for s in spots]
    E.NODE_YAW = {i: yaw[i] for i in range(n)}
    E.AUG_NODES, E.NODE_PAT, E.POSE_SHARE = [], {}, {}
    if looks == "nod":
        E.AUG_NODES = list(range(n)) * 2
        E.NODE_PAT = {**{n + i: f"{pat}_t+20" for i in range(n)}, **{2 * n + i: f"{pat}_t-20" for i in range(n)}}
        E.POSE_SHARE = {**{n + i: i for i in range(n)}, **{2 * n + i: i for i in range(n)}}
        for i in range(n):
            E.NODE_YAW[n + i] = yaw[i]
            E.NODE_YAW[2 * n + i] = yaw[i]
        L = lambda i: (i, n + i, 2 * n + i)  # noqa: E731
    else:
        L = lambda i: (i,)  # noqa: E731
    E.SINGLE_GROUPS = {spots[i][0]: list(L(i)) for i in range(n)}
    groups = {}
    for name, idx in R.COMBO[sc].items():
        if name == "single" or len(idx) < 2:
            continue
        groups[name] = tuple(j for i in idx for j in L(i))
    if os.environ.get("REAL_ONE") == "1":
        # one spot only (the first spot of the pair): what is left when the camera stops sending before the unit is moved
        groups["one"] = tuple(L(R.COMBO[sc]["single"]))
    E.EXTRA_GROUPS = groups
    E.ONLY_NODES = None
    E.GROUP_SPREAD = True


def data_dir(sc, cond):
    if sc in R.EXAM:
        return os.path.join(ROOT, "data_real", "base")
    return os.path.join(ROOT, "data_real", "nop" if cond == "design" else "op1")


def main():
    which, cond, bands, seeds, names = sys.argv[1:6]
    tag = sys.argv[6] if len(sys.argv) > 6 else f"{which}_{cond}"
    scenes = {"A": R.SET_A, "B": R.SET_B}.get(which) or which.split(",")
    bands = bands.split(",")
    seeds = [int(x) for x in seeds.split(",")]
    names = names.split(",")
    if os.environ.get("REAL_LL") == "1":
        E.STORE_LL_ALL = True
    out = os.path.join(ROOT, "results", f"real_{tag}.pkl")
    for band in bands:
        for sc in scenes:
            for nm in names:
                base, looks, extra = DESIGNS[nm]
                pat = f"{base}_H4"
                cfg = dict(cond_cfg(cond, band, sc in R.EXAM, pat), **extra)
                if isinstance(cfg.get("est_cls"), str):
                    import est_variants
                    cfg["est_cls"] = getattr(est_variants, cfg["est_cls"])
                if sc in R.EXAM and nm == "C_surf":
                    continue      # exam rooms keep their desk candidates
                setup_scene(sc, looks, pat)
                E.run({nm: cfg}, [sc], [band], seeds, data_dir(sc, cond), NEC_REAL, out, store_zone=False, verbose=True)
    print("REAL_DONE", which, cond, bands, seeds, names, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "nec":
        print(build_nec_real())
    else:
        main()
