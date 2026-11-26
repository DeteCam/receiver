"""E5 (sweeps), E7 (sign anchor cost / weak signal), box-model sensitivity, all six receiver pairs.
Everything else is the Rev.2 default (D=85 mm, exact manifold, conj average, 6-wall map)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
CFG = {
    "R2":          dict(R2),
    "R2_S5":       dict(R2, sig_xy=0.05, sig_z=0.025, sig_head=1.5),
    "E7_noflip":   dict(R2, flip_mode="none"),
    "E7_a2":       dict(R2, anchor_alpha=2.0),
    "E7_a5":       dict(R2, anchor_alpha=5.0),
    "E7_snr10":    dict(R2, snr_db=10.0),
    "E7_noflip_snr10": dict(R2, flip_mode="none", snr_db=10.0),
    "E5_n5":       dict(R2, n_sweep=5),
    "E5_n30":      dict(R2, n_sweep=30),
    "BOX_coarse":  dict(R2, pat="d85_cbox"),
}
if __name__ == "__main__":
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/exp_base.pkl", store_zone=False, verbose=False)
    print("EXP_BASE_DONE", flush=True)
