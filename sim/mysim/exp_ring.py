"""Receiver head structure within ~20 x 20 cm (base corner data, 13 rooms, 2.4 / 5 GHz).
Ring diameter 85 / 120 / 160 mm, 8 or 16 elements (SP8T vs SP16T), and a pan step of the whole head by 22.5 deg
(half the 8-element spacing) between two sweeps:
  *_pan_coh  coherent virtual 16-element ring (both orientations against the same reference position,
             pan-step error 1 deg rms unknown to the estimator)
  *_pan_inc  the two orientations as two separate looks, log-likelihoods summed (AUG_NODES copies)
Every receiver also gets an unturned repeat look (AUG copy with the same pattern) as a control for "measure twice".
All with Rev.2 + S5 on-site calibration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.AUG_NODES = [0, 1, 2, 3]                       # node 4 + b = second look of receiver b
E.POSE_SHARE = {4 + b: b for b in range(4)}
two = lambda *bs: tuple(x for b in bs for x in (b, 4 + b))
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3),
                  "x2_adj": two(0, 1), "x2_p02": two(0, 2), "x2_diag": two(0, 3), "x2_all4": two(0, 1, 2, 3)}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {
    "d85": dict(S5, pat="spec002plus_d85_dz0"),
    "d85_pan_inc": dict(S5, pat="spec002plus_d85_dz0", aug_pat="d85_r225"),
    "d85_pan_coh": dict(S5, pat="spec002plus_d85_dz0", virt=("d85_r225", 1.0)),
    "d120_8": dict(S5, pat="d120_8"),
    "d120_8_pan_coh": dict(S5, pat="d120_8", virt=("d120_8_r225", 1.0)),
    "d160_8": dict(S5, pat="d160_8"),
    "d160_8_pan_inc": dict(S5, pat="d160_8", aug_pat="d160_8_r225"),
    "d160_8_pan_coh": dict(S5, pat="d160_8", virt=("d160_8_r225", 1.0)),
    "d120_16": dict(S5, pat="d120_16"),
    "d160_16": dict(S5, pat="d160_16"),
}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], seeds, "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_ring.pkl", store_zone=False, verbose=False)
    print("RING_DONE", flush=True)
