"""Stages of the review experiments (see plan_review_20261125.md). Usage: exp_queue.py <stage>"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import my_eval as E
from oracle import synth_oracle
from fresh_scenes import FRESH_SCENES

NEC = os.path.join(HERE, "nec_patterns.npy")
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
ALL = E.DEV_SCENES + E.EVAL_SCENES
B = ["2g4", "5g"]
stage = sys.argv[1]

if stage == "size":       # E4 + E6 upper bound, base rooms
    CFG = {
        "gap40": dict(R2, pat="d85_gap40"), "gap20": dict(R2, pat="d85_gap20"), "gap10": dict(R2, pat="d85_gap10"),
        "gp88_up": dict(R2, pat="gp88_up"), "gp88_inv": dict(R2, pat="gp88_inv"),
        "E6_oracle": dict(R2, synth_fn=synth_oracle),
    }
    E.run(CFG, ALL, B, [0, 1, 2], "/opt/detecam/sim/data", NEC, "/opt/detecam/sim/results/exp_size.pkl", store_zone=False, verbose=False)
elif stage.startswith("h_"):   # E2 mounting height
    tag = stage[2:]
    CFG = {"R2": dict(R2)}   # ground-plane variants dropped: already far worse at the standard height (size stage, fresh rooms)
    E.run(CFG, ALL, B, [0, 1, 2], f"/opt/detecam/sim/data_ext/{tag}", NEC, f"/opt/detecam/sim/results/exp_{tag}.pkl", store_zone=False, verbose=False)
elif stage.startswith("p_"):   # E3 placement
    tag = stage[2:]
    E.run({"R2": dict(R2)}, ALL, B, [0, 1], f"/opt/detecam/sim/data_ext/{tag}", NEC, f"/opt/detecam/sim/results/exp_{tag}.pkl", store_zone=False, verbose=False)
elif stage == "fresh":    # rooms nobody tuned on
    CFG = dict(E.REV2_CONFIGS)
    CFG.pop("D75_EXACT_MAP")
    CFG.update({"gap20": dict(R2, pat="d85_gap20"), "gp88_inv": dict(R2, pat="gp88_inv"), "gp88_up": dict(R2, pat="gp88_up")})
    E.run(CFG, FRESH_SCENES, B, [0, 1, 2], "/opt/detecam/sim/data_fresh/base", NEC, "/opt/detecam/sim/results/fresh.pkl", store_zone=True, verbose=False)
print("STAGE_DONE", stage, flush=True)
