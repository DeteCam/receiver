import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
R2 = dict(exact_manifold=True, tc_step_ns="auto")
CFG = {"gap80": dict(R2, pat="d85_gap80"), "gap100": dict(R2, pat="d85_gap100")}
E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data",
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
      "/opt/detecam/sim/results/exp_gap.pkl", store_zone=False, verbose=False)
print("GAP_DONE", flush=True)
