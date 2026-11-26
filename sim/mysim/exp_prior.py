"""AR-registered prior information (software only).
Exam rooms (classroom/lecture): candidate points on every registered desk (desk top, chest, glasses heights),
  registration error 10 cm; metrics also as seat top-1 / top-3.
Other rooms: candidate points on walls and ceiling (room box only), or also on top of large furniture.
prior_logeps = log weight of the ordinary grid points; 0 = candidates only add resolution (no prior).
Baseline = Rev.2 default procedure incl. S5 on-site calibration. Weights are chosen on the 4 development rooms."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from fresh_scenes import FRESH_SCENES
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
E.STORE_PTS = True
S5 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
EXAM = ["classroom_a", "classroom_b", "lecture_a", "lecture_b"]
OTHER = [s for s in E.DEV_SCENES + E.EVAL_SCENES if s not in EXAM]
CFG_EXAM = {"S5": dict(S5)}
CFG_OTHER = {"S5": dict(S5)}
for le in (0.0, -1.0, -3.0, -6.0):
    CFG_EXAM[f"seat{le:g}"] = dict(S5, cand="seat", prior_logeps=le)
    CFG_OTHER[f"surf{le:g}"] = dict(S5, cand="surface", prior_logeps=le)
    CFG_OTHER[f"furn{le:g}"] = dict(S5, cand="surface_furn", prior_logeps=le)
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy")
if __name__ == "__main__":
    out = "/opt/detecam/sim/results/exp_prior.pkl"
    E.run(CFG_EXAM, EXAM, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", NEC, out, store_zone=False, verbose=True)
    E.run(CFG_OTHER, OTHER, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", NEC, out, store_zone=False, verbose=True)
    outf = "/opt/detecam/sim/results/exp_prior_fresh.pkl"
    E.run(CFG_EXAM, ["classroom_c"], ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_fresh/base", NEC, outf, store_zone=False)
    E.run(CFG_OTHER, ["lodging_c", "office_a", "restroom_c"], ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data_fresh/base",
          NEC, outf, store_zone=False)
    print("PRIOR_DONE", flush=True)
