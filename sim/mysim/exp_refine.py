"""Second pass on a fine local grid (0.1 m, radius 0.6 m) round the fused MAP point, base corner data
(13 rooms, 2.4 / 5 GHz, seed 0): current head and the 16-element D120 head, with and without the exam-room seat prior."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
E.REFINE_ONLY = {"adj", "p02", "diag", "all4"}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
SEAT = dict(cand="auto", prior_logeps=-1.0)
CFG = {
    "d85_ref": dict(S5, pat="spec002plus_d85_dz0", refine=(0.6, 0.1)),
    "d120_16_ref": dict(S5, pat="d120_16", refine=(0.6, 0.1)),
    "d120_16_seat_ref": dict(S5, pat="d120_16", refine=(0.6, 0.1), **SEAT),
}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    for k in CFG:
        E.run({k: CFG[k]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
              "/opt/detecam/sim/results/exp_refine.pkl", store_zone=False, verbose=False)
        print("done", k, flush=True)
    print("REFINE_DONE", flush=True)
