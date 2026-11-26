"""Which head motion is enough?  Every receiver is also measured after small known moves (data_ext/head:
HEAD_OFFSETS_CM in gen_ext.py, 28 offsets per receiver, 2.4 GHz, seed 0) and the looks are fused by summing the
log-likelihoods.  Compares one-axis moves (along the wall x / along the other wall y / vertical z) with two-axis
patterns, at equal or different numbers of looks.  Rev.2 + S5; every look of a receiver shares its pose error."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E

NOFF = 28
E.POSE_SHARE = {4 + b * NOFF + k: b for b in range(4) for k in range(NOFF)}
MOTIONS = {
    "none": (),
    "x5": (0, 1), "x10": (2, 3), "x20": (4, 5), "x10_20": (2, 3, 4, 5), "xline": (0, 1, 2, 3, 4, 5),
    "z5": (6, 7), "z10": (8, 9), "z20": (10, 11), "z10_20": (8, 9, 10, 11), "zline": (6, 7, 8, 9, 10, 11),
    "y5": (12, 13), "y10": (14, 15),
    "cross10": (2, 3, 8, 9), "cross20": (4, 5, 10, 11), "sq10": (16, 17, 18, 19), "grid9": (2, 3, 8, 9, 16, 17, 18, 19),
    "orb5": (0, 1, 12, 13, 20, 21, 22, 23), "orb10": (2, 3, 14, 15, 24, 25, 26, 27), "orb10_4": (24, 25, 26, 27),
    "x10z10_2": (2, 8),   # only two extra looks, one per axis
}
PAIRS = {"adj": (0, 1), "p02": (0, 2), "diag": (0, 3), "all4": (0, 1, 2, 3)}


def looks(b, ks):
    return (b,) + tuple(4 + b * NOFF + k for k in ks)


E.EXTRA_GROUPS = {f"{mn}|{pn}": tuple(x for b in bs for x in looks(b, ks))
                  for mn, ks in MOTIONS.items() for pn, bs in PAIRS.items()}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {"d85": dict(S5, pat="spec002plus_d85_dz0")}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    names = (sys.argv[1] if len(sys.argv) > 1 else "d85").split(",")
    cfg = {k: v for k, v in CFG.items() if k in names}
    E.run(cfg, E.DEV_SCENES + E.EVAL_SCENES, ["2g4"], [0], "/opt/detecam/sim/data_ext/head", NEC,
          "/opt/detecam/sim/results/exp_head.pkl", store_zone=False, verbose=False)
    print("HEAD_DONE", flush=True)
