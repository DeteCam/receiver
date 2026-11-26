"""Ceiling-height mounting (2.8 m) in the four classroom-type rooms: does looking over the seated students help?
Rooms whose ceiling is below 3 m are left out. Same cameras / people as the base scenes (gen_ext.py --tags z280)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
ROOMS = ["classroom_a", "classroom_b", "lecture_a", "lecture_b"]
if __name__ == "__main__":
    E.run({"R2": R2}, ROOMS, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_ext/z280",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/exp_z280.pkl", store_zone=False, verbose=False)
    print("CEIL_DONE", flush=True)
