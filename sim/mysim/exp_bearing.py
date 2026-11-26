"""Ring single-receiver bearing (plan sim/plan_20261126_plate.md).
Final ring (anchor H4, real part losses) at the outside-exam wall spots, nobody in the room; every receiver's
three nod looks (-20 / 0 / +20 deg) are summed and the azimuth is read from that receiver's own map.
Usage: python3 mysim/exp_bearing.py <seeds> [bands]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
import exp_wall as X
import exp_final as F

GROUPS = {"c0": [0, 16, 32], "c2": [2, 18, 34], "m0": [4, 20, 36], "m2": [6, 22, 38]}


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    bands = sys.argv[2].split(",") if len(sys.argv) > 2 else ["2g4", "5g"]
    if not os.path.exists(F.NEC_FINAL):
        F.build_nec_final()
    out = "/opt/detecam/sim/results/exp_bearing_ring.pkl"
    for band in bands:
        alpha, g, snr = F.ANCHOR["H4"][band]
        cfg = dict(F.CFG["d0"], pat="d120_16_H4", anchor_alpha=alpha, snr_db=snr)
        X.setup("nod", "d120_16_H4")
        E.ONLY_NODES = sorted(i for v in GROUPS.values() for i in v)
        E.EXTRA_GROUPS = {}
        E.SINGLE_GROUPS = dict(GROUPS)
        E.run({"H4": cfg}, F.OUTSIDE, [band], seeds, "/opt/detecam/sim/data_wall/nop", F.NEC_FINAL, out,
              store_zone=False, verbose=True)
    print("BEARING_DONE", seeds, bands, flush=True)


if __name__ == "__main__":
    main()
