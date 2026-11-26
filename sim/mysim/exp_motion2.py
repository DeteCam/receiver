"""More head motions for the recommended D120 16-element head (base corner data, 13 rooms, 2.4 / 5 GHz, seed 0):
  nod   : head tilted +-20 deg about the horizontal axis through the reference (extra looks, likelihoods summed)
  fold  : a folding head measured folded (D120) and opened (D240 ring), likelihoods summed; and opened alone
  rep   : control - the same number of extra looks without any change (pass 2)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.AUG_NODES = [0, 1, 2, 3] * 3                    # nodes 4-7, 8-11, 12-15
E.POSE_SHARE = {4 + 4 * j + b: b for j in range(3) for b in range(4)}
PAIRS = {"adj": (0, 1), "p02": (0, 2), "diag": (0, 3), "all4": (0, 1, 2, 3)}
LOOKS = {"one": (0,), "two": (0, 1), "three": (0, 1, 2), "unf": (3,), "fold2": (0, 3)}
E.EXTRA_GROUPS = {f"{ln}|{pn}": tuple((b if j == 0 else 4 + 4 * (j - 1) + b) for b in bs for j in js)
                  for ln, js in LOOKS.items() for pn, bs in PAIRS.items()}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5, pat="d120_16")
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all2.npy")
if __name__ == "__main__":
    # pass 1: nodes 4-7 nod +20, 8-11 nod -20, 12-15 opened D240
    E.NODE_PAT = {**{4 + b: "c120_16_t+20" for b in range(4)}, **{8 + b: "c120_16_t-20" for b in range(4)},
                  **{12 + b: "c240_16" for b in range(4)}}
    E.run({"nodfold": S5}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_motion2.pkl", store_zone=False, verbose=False)
    print("done nodfold", flush=True)
    # pass 2: the same looks repeated without a change (control)
    E.NODE_PAT = {}
    E.run({"repeat": S5}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_motion2.pkl", store_zone=False, verbose=False)
    print("MOTION2_DONE", flush=True)
