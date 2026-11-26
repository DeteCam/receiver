import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
X = dict(exact_manifold=True, tc_step_ns="auto")
CFG = {"D100_cbox": dict(X, pat="d100_cbox"), "D85_cbox": dict(X, pat="d85_cbox"),
       "D100_fine": dict(X, pat="spec002plus_8_dz0_abs"), "D85_fine": dict(X, pat="spec002plus_d85_dz0")}
E.run(CFG, E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
      "/opt/detecam/sim/results/exp_boxsens.pkl", store_zone=False, verbose=False)
print("BOXSENS_DONE", flush=True)
