"""Per-receiver maps (three looks of the one-axis move summed) of the recommended set-up, for the learned re-ranking:
D120 16-element head + +-10 cm along the wall + seat layout in exam rooms (data_ext/jit10, 13 rooms, 2 bands, seeds 0-1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from exp_combo import CFG, NEC
E.POSE_SHARE = {4 + 4 * b + k: b for b in range(4) for k in range(4)}
E.STORE_LL = True
E.STORE_LL_GROUPS = {b: (b, 4 + 4 * b + 0, 4 + 4 * b + 1) for b in range(4)}
E.EXTRA_GROUPS = {f"x|{pn}": tuple(x for b in bs for x in E.STORE_LL_GROUPS[b])
                  for pn, bs in {"adj": (0, 1), "p02": (0, 2), "diag": (0, 3), "all4": (0, 1, 2, 3)}.items()}
if __name__ == "__main__":
    E.run({"d120_16_seat": CFG["d120_16_seat"]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1],
          "/opt/detecam/sim/data_ext/jit10", NEC, "/opt/detecam/sim/results/exp_ll_combo.pkl", store_zone=False,
          verbose=False)
    print("COMBO_LL_DONE", flush=True)
