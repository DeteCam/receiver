"""Putting the levers together (data_ext/jit10: every receiver also measured at +-10 cm along the wall and +-10 cm
vertically; 13 rooms, 2.4 / 5 GHz, seeds 0 and 1).
  head: D85 ring (Rev.2) | D160 8-element ring + 22.5 deg pan step (coherent virtual 16) | D160 16-element ring
  moves: none / along the wall only / vertical only / both
  prior: none | seat layout in exam rooms (AR-registered desks, 10 cm registration error)
All with Rev.2 + S5 on-site calibration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E

E.POSE_SHARE = {4 + 4 * b + k: b for b in range(4) for k in range(4)}
MOVES = {"s": (), "x": (0, 1), "z": (2, 3), "xz": (0, 1, 2, 3)}
PAIRS = {"adj": (0, 1), "p02": (0, 2), "diag": (0, 3), "all4": (0, 1, 2, 3)}
E.EXTRA_GROUPS = {f"{mv}|{pn}": tuple(x for b in bs for x in ((b,) + tuple(4 + 4 * b + k for k in ks)))
                  for mv, ks in MOVES.items() for pn, bs in PAIRS.items()}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
SEAT = dict(cand="auto", prior_logeps=-1.0)
CFG = {
    "d85": dict(S5, pat="spec002plus_d85_dz0"),
    "d160_16": dict(S5, pat="d160_16"),
    "d160p": dict(S5, pat="d160_8", virt=("d160_8_r225", 1.0)),
    "d85_seat": dict(S5, pat="spec002plus_d85_dz0", **SEAT),
    "d160_16_seat": dict(S5, pat="d160_16", **SEAT),
    "d160p_seat": dict(S5, pat="d160_8", virt=("d160_8_r225", 1.0), **SEAT),
    "d120_16": dict(S5, pat="d120_16"),
    "d120_16_seat": dict(S5, pat="d120_16", **SEAT),
}
REF = {"d120_16_seat_ref": dict(CFG["d120_16_seat"], refine=(0.6, 0.1))}
E.REFINE_ONLY = {f"{mv}|{pn}" for mv in ("s", "xz") for pn in PAIRS}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "d85,d120_16,d120_16_seat"
    if mode == "ref":      # refined estimates (costly): seed 0 only, separate file so it can run beside the main pass
        for k in REF:
            E.run({k: REF[k]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data_ext/jit10", NEC,
                  "/opt/detecam/sim/results/exp_combo_ref.pkl", store_zone=False, verbose=False)
            print("done", k, flush=True)
    else:
        seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1").split(",")]
        for k in mode.split(","):   # one config per pass: 20 looks x seat candidates per estimator set is memory-heavy
            E.run({k: CFG[k]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], seeds,
                  "/opt/detecam/sim/data_ext/jit10", NEC, "/opt/detecam/sim/results/exp_combo.pkl", store_zone=False,
                  verbose=False)
            print("done", k, flush=True)
    print("COMBO_DONE", flush=True)
