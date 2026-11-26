"""Zone statistics for all six receiver pairs in the two classroom-type eval rooms whose board-side pair is (0,2).
rev2.pkl stored zone data only for (0,1) and the diagonal, so the board-side check needs this rerun.
Same configs as the Rev.2 reproduction (EXACT_D85_MAP = Rev.2 default, S5CAL_D85_MAP = with S5)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
CFG = {k: E.REV2_CONFIGS[k] for k in ("EXACT_D85_MAP", "S5CAL_D85_MAP")}
if __name__ == "__main__":
    E.run(CFG, ["classroom_b", "lecture_b"], ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/zone_front.pkl", store_zone=True, verbose=False)
    print("ZONE_FRONT_DONE", flush=True)
