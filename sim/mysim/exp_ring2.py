"""Re-run of the pan-step (coherent virtual 16-element) heads with the two sweeps sharing the device's calibration
residuals (port gains, anchor, reference row) as a real head would; pan-step error 1 deg or 0.3 deg rms."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from exp_ring import S5, NEC
E.AUG_NODES = [0, 1, 2, 3]
E.POSE_SHARE = {4 + b: b for b in range(4)}
two = lambda *bs: tuple(x for b in bs for x in (b, 4 + b))
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3),
                  "x2_adj": two(0, 1), "x2_p02": two(0, 2), "x2_diag": two(0, 3), "x2_all4": two(0, 1, 2, 3)}
CFG = {
    "d85_pan_coh2": dict(S5, pat="spec002plus_d85_dz0", virt=("d85_r225", 1.0)),
    "d120_8_pan_coh2": dict(S5, pat="d120_8", virt=("d120_8_r225", 1.0)),
    "d160_8_pan_coh2": dict(S5, pat="d160_8", virt=("d160_8_r225", 1.0)),
    "d160_8_pan_coh2_r03": dict(S5, pat="d160_8", virt=("d160_8_r225", 0.3)),
}
if __name__ == "__main__":
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_ring.pkl", store_zone=False, verbose=False)
    print("RING2_DONE", flush=True)
