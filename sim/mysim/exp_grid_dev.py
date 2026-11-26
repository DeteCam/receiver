import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
R2 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto")
CFG = {
    "G_base":   dict(R2),
    "G_cart15": dict(R2, grid=("cart", 0.15, (0.10, 0.30, 0.50, 0.70, 0.90), 4000)),
    "G_polar":  dict(R2, grid=("polar",)),
    "G_base5z": dict(R2, grid=("cart", 0.30, (0.10, 0.30, 0.50, 0.70, 0.90), 260)),
}
if __name__ == "__main__":
    E.run(CFG, E.DEV_SCENES, ["2g4", "5g"], [0, 1, 2], "/opt/detecam/sim/data", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_patterns.npy"),
          "/opt/detecam/sim/results/grid_dev.pkl", store_zone=False)
