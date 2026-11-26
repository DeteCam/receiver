"""How much do 3 or 4 measurement positions help compared with 2? (Rev.2 default, equal-weight log-likelihood sum)
Groups: all four corners, and every 3-corner subset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from fresh_scenes import FRESH_SCENES
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3), "t012": (0, 1, 2), "t013": (0, 1, 3), "t023": (0, 2, 3), "t123": (1, 2, 3)}
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy")
if __name__ == "__main__":
    E.run({"R2": R2}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_multi.pkl", store_zone=False, verbose=False)
    E.run({"R2": R2}, FRESH_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_fresh/base", NEC,
          "/opt/detecam/sim/results/exp_multi_fresh.pkl", store_zone=False, verbose=False)
    print("MULTI_DONE", flush=True)
