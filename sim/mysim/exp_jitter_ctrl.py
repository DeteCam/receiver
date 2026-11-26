"""Controls for the small-move result: (1) per-port calibration error switched off so that moving cannot average a
residual calibration error that a real device would keep, (2) same total packets without moving (50 sweeps at one spot)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
import exp_jitter as J  # sets POSE_SHARE / EXTRA_GROUPS
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto", cal_phase_deg=0.0, cal_amp_db=0.0)
CFG = {"nocal_n10": dict(R2), "nocal_n50": dict(R2, n_sweep=50)}
if __name__ == "__main__":
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1], "/opt/detecam/sim/data_ext/jit10",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/exp_jit10_ctrl.pkl", store_zone=False, verbose=False)
    print("JITC_DONE", flush=True)
