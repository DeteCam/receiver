"""Final-configuration check (plan sim/plan_20261126_final.md).
Two wall positions at 2.0 m, the head nodded -20 / 0 / +20 deg, final parts stress (SNR 14 dB, calibration 10 deg).
Only the nodes the pre-registered pairs need are evaluated (my_eval.ONLY_NODES); RNG streams are per node, so the
kept nodes give the same numbers as a full run.
Usage: python3 mysim/exp_final.py <outside|exam> <seeds> [cfg1,cfg2,...]
  outside: data_wall/nop (12 rooms outside exam rooms, nobody in the room)
  exam:    data_wall/base (5 exam rooms, seated students)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E
import exp_wall as X
import analyze_wall as W

S5 = X.S5
CFG = {
    # outside exam rooms, nobody in the room: no moving paths (main), the old 35 % moving paths (comparison)
    "d0": dict(S5, pat="d120_16", dyn_frac=0.0),
    "d35": dict(S5, pat="d120_16"),
    "d0_snr14": dict(S5, pat="d120_16", dyn_frac=0.0, snr_db=14.0),
    "d0_cal10": dict(S5, pat="d120_16", dyn_frac=0.0, cal_phase_deg=10.0, cal_amp_db=0.8),
    # exam rooms: seated students, automatic desk candidates (same as d120_16_seat10m)
    "seat": dict(X.CFG["d120_16_seat10m"]),
    "seat_snr14": dict(X.CFG["d120_16_seat10m"], snr_db=14.0),
    "seat_cal10": dict(X.CFG["d120_16_seat10m"], cal_phase_deg=10.0, cal_amp_db=0.8),
}
OUTSIDE = ["lodging_a", "restroom_a", "locker_a", "bathhouse_a", "bathhouse_b", "lodging_b", "restroom_b",
           "locker_b", "worst_metal", "lodging_c", "office_a", "restroom_c"]
EXAM = ["classroom_a", "lecture_a", "lecture_b", "classroom_b", "classroom_c"]
# base nodes (height A): c0=0 c1=1 c2=2 c3=3 m0=4 m1=5 m2=6 ; nod copies +16 (looks down 20) / +32 (looks up 20)
NODES = {"outside": [0, 2, 4, 5, 6], "exam": [0, 1, 2, 3]}

def main():
    which = sys.argv[1]
    if which.startswith("anchor-"):
        run_anchor(which.split("-", 1)[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3].split(","))
        return
    seeds = [int(x) for x in sys.argv[2].split(",")]
    default = ["d0", "d35", "d0_snr14", "d0_cal10"] if which == "outside" else ["seat", "seat_snr14", "seat_cal10"]
    cfgs = sys.argv[3].split(",") if len(sys.argv) > 3 else default
    X.setup("nod", "d120_16")
    base = NODES[which]
    E.ONLY_NODES = base + [16 + i for i in base] + [32 + i for i in base]
    keep = ("diagA", "madjA", "moppA") if which == "outside" else ("c01A", "c03A", "c23A")
    E.EXTRA_GROUPS = {g: v for g, v in E.EXTRA_GROUPS.items() if g in keep or g in tuple("nod_" + k for k in keep)}
    variant = "nop" if which == "outside" else "base"
    scenes = OUTSIDE if which == "outside" else EXAM
    tag = "_".join(cfgs)
    out = f"/opt/detecam/sim/results/exp_final_{which}_{tag}.pkl"
    E.run({c: CFG[c] for c in cfgs}, scenes, ["2g4", "5g"], seeds, f"/opt/detecam/sim/data_wall/{variant}", X.NEC,
          out, store_zone=False, verbose=True)
    print("FINAL_DONE", which, tag, seeds, flush=True)


# ---- section 10 of the plan: anchor coupler with real part values (added before the results) ----
ANCHOR = {   # alpha (anchor / chain-A reference), g (ring path / chain-A reference), chain-A SNR (dB); per band
    "H1": {"2g4": (0.67, 0.70, 16.5), "5g": (0.56, 0.57, 15.0)},   # two Wilkinson dividers, 0 dB attenuator
    "H2": {"2g4": (2.00, 1.48, 10.0), "5g": (1.68, 1.01, 10.0)},   # 10 dB directional coupler + Wilkinson combiner
    "H4": {"2g4": (0.67, 0.35, 16.5), "5g": (0.56, 0.285, 15.0)},  # H1 + 6 dB attenuator after the switch common
    "H5": {"2g4": (0.67, 0.22, 16.5), "5g": (0.56, 0.18, 15.0)},   # H1 + 10 dB attenuator after the switch common
}
NEC_FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_final.npy")


def build_nec_final():
    db = np.load(X.NEC, allow_pickle=True).item()
    for h, per in ANCHOR.items():
        for band, (alpha, g, snr) in per.items():
            for src, dst in (("d120_16", f"d120_16_{h}"), ("d120_16_ty+20", f"d120_16_{h}_t+20"),
                             ("d120_16_ty-20", f"d120_16_{h}_t-20")):
                d = dict(db[(src, band)])
                d["eth"] = (d["eth"] * g).astype(np.complex64)
                d["eph"] = (d["eph"] * g).astype(np.complex64)
                db[(dst, band)] = d
    np.save(NEC_FINAL, db, allow_pickle=True)


def run_anchor(which, seeds, hs):
    if not os.path.exists(NEC_FINAL):
        build_nec_final()
    base_cfg = CFG["d0"] if which == "outside" else CFG["seat"]
    variant = "nop" if which == "outside" else "base"
    scenes = OUTSIDE if which == "outside" else EXAM
    keep = ("diagA", "madjA", "moppA") if which == "outside" else ("c01A", "c03A", "c23A")
    out = (f"/opt/detecam/sim/results/exp_final_anchor_{which}.pkl" if which == "outside"
           else f"/opt/detecam/sim/results/exp_final_anchor_{which}_{'_'.join(hs)}.pkl")
    board_nodes = {"c01": [0, 1], "c03": [0, 3], "c23": [2, 3]}
    for h in hs:
        for band in ("2g4", "5g"):
            alpha, g, snr = ANCHOR[h][band]
            cfg = dict(base_cfg, pat=f"d120_16_{h}", anchor_alpha=alpha, snr_db=snr)
            for sc_list in ([scenes] if which == "outside" else [[s] for s in scenes]):
                X.setup("nod", f"d120_16_{h}")
                if which == "outside":
                    base, kp = NODES[which], keep
                else:
                    bname = W.BOARD[sc_list[0]]
                    base, kp = board_nodes[bname], (bname + "A",)
                E.ONLY_NODES = base + [16 + i for i in base] + [32 + i for i in base]
                E.EXTRA_GROUPS = {k: v for k, v in E.EXTRA_GROUPS.items() if k in kp or k in tuple("nod_" + q for q in kp)}
                E.run({h: cfg}, sc_list, [band], seeds, f"/opt/detecam/sim/data_wall/{variant}", NEC_FINAL, out,
                      store_zone=False, verbose=True)
    print("ANCHOR_DONE", which, hs, seeds, flush=True)


if __name__ == "__main__":
    main()
