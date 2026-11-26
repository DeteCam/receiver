import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from fresh_scenes import FRESH_SCENES
R2 = dict(exact_manifold=True, tc_step_ns="auto")
CFG = {"gap80": dict(R2, pat="d85_gap80"), "gap100": dict(R2, pat="d85_gap100"), "EXACT_D85_MAP": dict(R2, pat="spec002plus_d85_dz0")}
E.run(CFG, FRESH_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_fresh/base",
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
      "/opt/detecam/sim/results/exp_gap_fresh.pkl", store_zone=False, verbose=False)
print("GAPF_DONE", flush=True)
