"""Head shapes (16 elements each unless noted) against the recommended D120 ring: polygons, cross, grid, concentric
rings, stacked rings (3-D), dual-slant elements, and folding heads measured unfolded (D240 ring, 32 cm cross,
28 cm triangle).  Rev.2 + S5, 13 rooms, 2.4 / 5 GHz, seed 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
import shapes as SH
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
CFG = {"d120_16": dict(S5, pat="d120_16"), "d160_16": dict(S5, pat="d160_16")}
CFG.update({k: dict(S5, pat=k) for k in SH.GEOM})
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all2.npy")
if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CFG)
    for k in names:
        E.run({k: CFG[k]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
              "/opt/detecam/sim/results/exp_shapes.pkl", store_zone=False, verbose=False)
        print("done", k, flush=True)
    print("SHAPES_DONE", flush=True)
