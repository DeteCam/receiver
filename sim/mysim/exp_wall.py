"""Wall-mounted receivers (plan sim/plan_20261126_wall.md).
Data: data_wall/<variant>/ (16 nodes = 8 wall positions x heights A 2.0 m / B 1.5 m, facing into the room).
Usage: python3 mysim/exp_wall.py <variant> <seeds> <cfg1,cfg2,...> [part 0|1|all] [aug]
  aug = "nod": extra looks with the head nodded +20 / -20 deg (virtual copies of nodes 0-15)
        "pan": extra looks with the head turned +20 / -20 deg about the vertical axis"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E

LAB = ["c0", "c1", "c2", "c3", "m0", "m1", "m2", "m3"]
YAW8 = [np.pi / 2, np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi, -np.pi / 2, 0.0]
IDX = {f"{l}{h}": i + 8 * k for k, h in enumerate("AB") for i, l in enumerate(LAB)}
FRESH = ["classroom_c", "lodging_c", "office_a", "restroom_c"]
SCENES = E.DEV_SCENES + E.EVAL_SCENES + FRESH
# pre-registered position groups (names without height); the board pair / tri group is chosen per room at analysis
GROUPS = {
    "c01": ("c0", "c1"), "c03": ("c0", "c3"), "c23": ("c2", "c3"), "diag": ("c0", "c2"),
    "madj": ("m0", "m1"), "mopp": ("m0", "m2"),
    "c01m2": ("c0", "c1", "m2"), "c03m1": ("c0", "c3", "m1"), "c23m0": ("c2", "c3", "m0"),
    "quadc": ("c0", "c1", "c2", "c3"), "quadm": ("m0", "m1", "m2", "m3"),
    "all8": tuple(LAB),
}
# corner-mounted heads (data_wall/corner_<variant>): nodes 0-3 = corners k0..k3 at 2.0 m, 4-7 at 1.5 m
KYAW = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
KGROUPS = {"k01": (0, 1), "k03": (0, 3), "k23": (2, 3), "kdiag": (0, 2),
           "k012": (0, 1, 2), "k032": (0, 3, 2), "k230": (2, 3, 0), "kquad": (0, 1, 2, 3)}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {k: dict(S5, pat=k) for k in ("d120_16", "semi16", "semi16s", "plan2x8", "semi16L")}
CFG["d120_16_examz"] = dict(S5, pat="d120_16", grid=("examz",))
# calibration residual sensitivity: a ring 4-6 cm from a wall vs its free-space calibration (proxy: larger
# per-element phase / gain residuals); the plate-backed head keeps a smaller residual on non-metal walls
CFG["d120_16_cal10"] = dict(S5, pat="d120_16", cal_phase_deg=10.0, cal_amp_db=0.8)
CFG["d120_16_cal20"] = dict(S5, pat="d120_16", cal_phase_deg=20.0, cal_amp_db=1.5)
CFG["semi16_cal8"] = dict(S5, pat="semi16", cal_phase_deg=8.0, cal_amp_db=0.6)
# exam rooms only: desks / seated people found AUTOMATICALLY by the phone (AR plane + person detection), not entered
# by hand -> candidate points over each desk with a common placement error (10 cm / 25 cm), soft prior
CFG["d120_16_seat10"] = dict(S5, pat="d120_16", cand="auto", prior_logeps=-1.0, reg_sig=0.10)
CFG["d120_16_seat25"] = dict(S5, pat="d120_16", cand="auto", prior_logeps=-1.0, reg_sig=0.25)
CFG["d120_16_seat10m"] = dict(S5, pat="d120_16", cand="auto", prior_logeps=-1.0, reg_sig=0.10)  # + desk masses
for _k in range(3):
    CFG[f"d120_16_seat10m{_k}"] = CFG["d120_16_seat10m"]   # one output file per seed (parallel runs)
CFG["d120_16_ref"] = dict(S5, pat="d120_16", refine=(0.6, 0.1))
# near field of the wall behind the head (mysim/wall_nf.py, Thevenin 50 ohm patterns): truth = currents solved in
# front of a wall, estimator = free-space calibration ("f") or a calibration on a concrete reference board ("c")
CFG["d120_16_fv"] = dict(S5, pat="d120_16_fv")
for _w in ("wc", "wp", "wm"):
    CFG[f"d120_16_{_w}f"] = dict(S5, pat=f"d120_16_{_w}", est_pat="d120_16_fv")
    CFG[f"d120_16_{_w}c"] = dict(S5, pat=f"d120_16_{_w}", est_pat="d120_16_wc")
CFG["d120_16_wmm"] = dict(S5, pat="d120_16_wm", est_pat="d120_16_wm")   # a separate metal-surface table
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all4.npy")


def setup_corner(aug=None, pat=None):
    E.NODE_YAW = {i + 4 * k: KYAW[i] for k in range(2) for i in range(4)}
    E.GROUP_SPREAD = True
    groups = {}
    for k, h in enumerate("AB"):
        for g, idx in KGROUPS.items():
            groups[f"{g}{h}"] = tuple(i + 4 * k for i in idx)
    for g, idx in KGROUPS.items():
        groups[f"{g}AB"] = tuple(i + 4 * k for k in range(2) for i in idx)
    if aug == "nod":
        E.AUG_NODES = list(range(8)) * 2
        tk = "_ty" if pat == "d120_16" else "_t"
        E.NODE_PAT = {**{8 + i: f"{pat}{tk}+20" for i in range(8)}, **{16 + i: f"{pat}{tk}-20" for i in range(8)}}
        E.POSE_SHARE = {**{8 + i: i for i in range(8)}, **{16 + i: i for i in range(8)}}
        for i in range(8):
            E.NODE_YAW[8 + i] = E.NODE_YAW[i]
            E.NODE_YAW[16 + i] = E.NODE_YAW[i]
        groups.update({f"nod_{g}": tuple(j for i in idx for j in (i, 8 + i, 16 + i)) for g, idx in list(groups.items())})
    E.EXTRA_GROUPS = groups


def setup(aug=None, pat=None):
    E.NODE_YAW = {i + 8 * k: YAW8[i] for k in range(2) for i in range(8)}
    E.GROUP_SPREAD = True
    groups = {}
    for h in "AB":
        for g, labs in GROUPS.items():
            groups[f"{g}{h}"] = tuple(IDX[f"{l}{h}"] for l in labs)
    for g, labs in GROUPS.items():   # the same positions measured at both heights (unit moved up/down once)
        groups[f"{g}AB"] = tuple(IDX[f"{l}{h}"] for h in "AB" for l in labs)
    if aug == "pan":
        # nodes 16-31 / 32-47: the same heads turned +20 / -20 deg about the vertical axis (servo pan)
        E.AUG_NODES = list(range(16)) * 2
        E.NODE_PAT = {}
        E.POSE_SHARE = {**{16 + i: i for i in range(16)}, **{32 + i: i for i in range(16)}}
        for i in range(16):
            E.NODE_YAW[16 + i] = E.NODE_YAW[i] + np.radians(20.0)
            E.NODE_YAW[32 + i] = E.NODE_YAW[i] - np.radians(20.0)
        groups.update({f"pan_{g}": tuple(j for i in idx for j in (i, 16 + i, 32 + i)) for g, idx in list(groups.items())})
    if aug == "nod":
        # nodes 16-31: tilt +20 (looks down) copies of 0-15, nodes 32-47: tilt -20 copies
        E.AUG_NODES = list(range(16)) * 2
        tk = "_ty" if pat == "d120_16" else "_t"
        E.NODE_PAT = {**{16 + i: f"{pat}{tk}+20" for i in range(16)}, **{32 + i: f"{pat}{tk}-20" for i in range(16)}}
        E.POSE_SHARE = {**{16 + i: i for i in range(16)}, **{32 + i: i for i in range(16)}}
        for i in range(16):
            E.NODE_YAW[16 + i] = E.NODE_YAW[i]
            E.NODE_YAW[32 + i] = E.NODE_YAW[i]
        nod = {}
        for g, idx in groups.items():
            nod[f"nod_{g}"] = tuple(j for i in idx for j in (i, 16 + i, 32 + i))
        groups.update(nod)
    E.EXTRA_GROUPS = groups


if __name__ == "__main__":
    variant = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(",")]
    cfgs = sys.argv[3].split(",")
    part = sys.argv[4] if len(sys.argv) > 4 else "all"
    aug = sys.argv[5] if len(sys.argv) > 5 else None
    if aug == "ll":          # store every node's map for offline fusion studies (no extra looks)
        E.STORE_LL_ALL = True
    scenes = {"all": SCENES, "0": SCENES[:9], "1": SCENES[9:],
              "exam": [s for s in SCENES if s.startswith(("classroom", "lecture"))]}[part]
    for c in cfgs:
        (setup_corner if variant.startswith("corner") else setup)(aug if aug in ("nod", "pan") else None, CFG[c]["pat"])
        if CFG[c].get("refine"):
            E.REFINE_ONLY = {g for g in E.EXTRA_GROUPS if g.endswith("A") and g[:-1] in
                             ("c01", "c03", "c23", "diag", "c01m2", "c03m1", "c23m0", "quadc", "quadm", "all8")}
        tag = f"{variant}_{c}" + (f"_{aug}" if aug else "")
        E.run({c: CFG[c]}, scenes, ["2g4", "5g"], seeds, f"/opt/detecam/sim/data_wall/{variant}", NEC,
              f"/opt/detecam/sim/results/exp_wall_{tag}_p{part}.pkl", store_zone=False, verbose=False)
        print("done", tag, part, flush=True)
    print("WALL_DONE", flush=True)
