"""Small motorised moves of a wall/corner receiver: each of the 4 receivers is also measured at +-10 cm along the wall
and +-10 cm vertically (known offsets, same pose error), and the five likelihoods are summed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
tag = sys.argv[1] if len(sys.argv) > 1 else "jit10"
E.POSE_SHARE = {4 + 4 * b + k: b for b in range(4) for k in range(4)}
mv = lambda b: (b,) + tuple(4 + 4 * b + k for k in range(4))
E.EXTRA_GROUPS = {"m0": mv(0), "s0": (0,), "m_adj": mv(0) + mv(1), "m_p02": mv(0) + mv(2), "m_diag": mv(0) + mv(3),
                  "m_all4": mv(0) + mv(1) + mv(2) + mv(3), "all4": (0, 1, 2, 3)}
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
if __name__ == "__main__":
    E.run({"R2": R2}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0, 1], f"/opt/detecam/sim/data_ext/{tag}",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          f"/opt/detecam/sim/results/exp_{tag}.pkl", store_zone=False, verbose=False)
    print("JIT_DONE", flush=True)
