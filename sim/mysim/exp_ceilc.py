"""Receiver 0 hung upside down from the middle of the ceiling (ring pattern rotated 180 deg about x),
receivers 1-3 stay in their corners. Single ceiling position, ceiling + one corner, and all four."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.NODE_PAT = {0: "spec002plus_d85_dz0_inv"}
E.EXTRA_GROUPS = {"c0": (0,), "all4": (0, 1, 2, 3), "c012": (0, 1, 2), "c013": (0, 1, 3), "c023": (0, 2, 3)}
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
if __name__ == "__main__":
    E.run({"R2": R2}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_ext/ceilc",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/exp_ceilc.pkl", store_zone=False, verbose=False)
    print("CEILC_DONE", flush=True)
