"""Store per-receiver log-likelihood maps (Rev.2 + S5, ring D85) for the software / AI post-processing study:
posterior-mean and local-centroid estimates, and a learned re-ranking of the top candidates
(leave-one-room-out on the 13 base rooms, plus the 4 fresh rooms as an untouched test)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
E.STORE_LL = True
E.STORE_PTS = True
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {"d85": dict(S5, pat="spec002plus_d85_dz0")}
EXTRA = {"d120_16": dict(S5, pat="d120_16"), "d160_16": dict(S5, pat="d160_16"), "d160_8_pan_coh": dict(S5, pat="d160_8", virt=("d160_8_r225", 1.0))}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    names = (sys.argv[1] if len(sys.argv) > 1 else "d85").split(",")
    allc = dict(CFG, **EXTRA)
    cfg = {k: allc[k] for k in names}
    tag = "_".join(names)
    E.run(cfg, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", NEC,
          f"/opt/detecam/sim/results/exp_ll_{tag}.pkl", store_zone=False, verbose=False)
    E.run(cfg, ["classroom_c", "lodging_c", "office_a", "restroom_c"], ["2g4", "5g"], [0, 1, 2],
          "/opt/detecam/sim/data_fresh/base", NEC, f"/opt/detecam/sim/results/exp_ll_{tag}_fresh.pkl", store_zone=False,
          verbose=False)
    print("LL_DONE", flush=True)
