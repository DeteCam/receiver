"""Error budget for the 2-position estimate: switch off one error source at a time (base corner data, 13 rooms).
B0  Rev.2 default without on-site calibration (pose 15 cm / 5 deg, room box 15 cm)
B1  + S5 on-site calibration (pose 5 cm / 1.5 deg)            <- Rev.2 default procedure
B2  B1 + user-registered big reflectors (blackboard / mirrors / locker faces) in the estimator
B3  B1 with perfect pose and perfect room box
B4  B3 + no per-port / anchor calibration error
B5  B4 + no noise and no moving people (static channel)  -> what is left is pure model mismatch (furniture, people)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
S5 = dict(R2, sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {
    "B0": dict(R2),
    "B1": dict(S5),
    "B2": dict(S5, use_reflectors=True),
    "B3": dict(S5, sig_xy=0.0, sig_z=0.0, sig_head=0.0, sig_dims=(0.0, 0.0, 0.0)),
    "B4": dict(S5, sig_xy=0.0, sig_z=0.0, sig_head=0.0, sig_dims=(0.0, 0.0, 0.0), cal_phase_deg=0.0, cal_amp_db=0.0),
    "B5": dict(S5, sig_xy=0.0, sig_z=0.0, sig_head=0.0, sig_dims=(0.0, 0.0, 0.0), cal_phase_deg=0.0, cal_amp_db=0.0,
               snr_db=99.0, dyn_frac=0.0),
}
if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], seeds, "/opt/detecam/sim/data",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/exp_budget.pkl", store_zone=False, verbose=True)
    print("BUDGET_DONE", flush=True)
