"""Do the nod and the slide add up?  D120 16-element head on data_ext/jit10 (every receiver also measured at +-10 cm
along the wall and +-10 cm vertically; 13 rooms, 2.4 / 5 GHz, seed 0), plus two virtual looks at the base position
with the head tilted +20 / -20 deg about the horizontal axis through the reference.
Looks per receiver position (likelihoods summed):
  s     : base                                  (1 look)
  x     : base + slide -10 / +10 cm along wall   (3 looks)
  nod   : base + tilt +20 / -20 deg               (3 looks)
  xz    : base + slides along the wall and up/down (5 looks)  <- same number of looks as xnod
  xnod  : base + slides along the wall + tilts     (5 looks)
  xznod : all of the above                         (7 looks)
Run in two halves (argument 0 / 1) so both CPU cores work; results/exp_motion3_p{0,1}.pkl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E

E.AUG_NODES = [0, 1, 2, 3] * 2                                   # nodes 20-23 tilt +20, 24-27 tilt -20
E.NODE_PAT = {**{20 + b: "c120_16_t+20" for b in range(4)}, **{24 + b: "c120_16_t-20" for b in range(4)}}
E.POSE_SHARE = {**{4 + 4 * b + k: b for b in range(4) for k in range(4)},
                **{20 + b: b for b in range(4)}, **{24 + b: b for b in range(4)}}
PAIRS = {"adj": (0, 1), "p02": (0, 2), "diag": (0, 3), "all4": (0, 1, 2, 3)}


def looks(b, name):
    sl = {"x": (4 + 4 * b + 0, 4 + 4 * b + 1), "z": (4 + 4 * b + 2, 4 + 4 * b + 3), "nod": (20 + b, 24 + b)}
    parts = {"s": (), "x": ("x",), "nod": ("nod",), "xz": ("x", "z"), "xnod": ("x", "nod"), "xznod": ("x", "z", "nod")}
    return (b,) + tuple(n for p in parts[name] for n in sl[p])


E.EXTRA_GROUPS = {f"{ln}|{pn}": tuple(n for b in bs for n in looks(b, ln))
                  for ln in ("s", "x", "nod", "xz", "xnod", "xznod") for pn, bs in PAIRS.items()}
S5 = dict(exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5, pat="d120_16")
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all2.npy")
if __name__ == "__main__":
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    scenes = E.DEV_SCENES + E.EVAL_SCENES
    scenes = scenes[:7] if part == 0 else scenes[7:]
    E.run({"d120_16": S5}, scenes, ["2g4", "5g"], [0], "/opt/detecam/sim/data_ext/jit10", NEC,
          f"/opt/detecam/sim/results/exp_motion3_p{part}.pkl", store_zone=False, verbose=False)
    print(f"MOTION3_DONE part {part}", flush=True)
