"""Ring with more wall spots (plan_20261126_plate.md section 7). Usage: python3 mysim/exp_ringmulti.py <seeds>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
import exp_wall as X
import exp_final as F

def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    if not os.path.exists(F.NEC_FINAL):
        F.build_nec_final()
    alpha, g, snr = F.ANCHOR["H4"]["2g4"]
    cfg = dict(F.CFG["d0"], pat="d120_16_H4", anchor_alpha=alpha, snr_db=snr)
    X.setup("nod", "d120_16_H4")
    w4 = [4, 5, 6, 7]; w8 = list(range(8))
    E.ONLY_NODES = [i + o for i in w8 for o in (0, 16, 32)]
    E.EXTRA_GROUPS = {"nod_W4": [i + o for i in w4 for o in (0, 16, 32)], "nod_W8": [i + o for i in w8 for o in (0, 16, 32)]}
    E.SINGLE_GROUPS = {}
    E.run({"H4": cfg}, F.OUTSIDE, ["2g4"], seeds, "/opt/detecam/sim/data_wall/nop", F.NEC_FINAL,
          "/opt/detecam/sim/results/exp_ringmulti.pkl", store_zone=False, verbose=True)
    print("RINGMULTI_DONE", seeds, flush=True)

if __name__ == "__main__":
    main()
