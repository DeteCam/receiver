"""Rod / film-cable heads (plan sim/plan_20261126_rod.md) on the exp_real pipeline.  [simulation]
Usage: python3 mysim/exp_rod.py nec                 -> rebuild mysim/nec_real.npy with nec_rod.npy added
       python3 mysim/exp_rod.py scales [band]       -> reference horizon gain of each head vs d120_16
       python3 mysim/exp_rod.py <exp_real.py args>  -> exp_real.main() with the Q* designs added (one band per call)
The 's' copies keep the receiver noise fixed in absolute terms (reference SNR moved by the reference horizon gain of
the head, as in exp_molex.py); the plain copies keep the reference SNR fixed."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_real as X  # noqa: E402
import exp_final as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = {"QF1": "d120_16f1", "QF2": "d120_16f2", "QR1": "d120_16r1", "QR2": "d120_16r2", "QR3": "d120_16r3",
         "QR4": "d120_16r4", "QF3": "d120_16f3", "QF4": "d120_16f4", "QF5": "d120_16f5", "QF6": "d120_16f6",
         "QF7": "d120_16f7", "QF8": "d120_16f8",
         "QX1": "d120_16x1", "QX3": "d120_16x3", "QX4": "d120_16x4", "QX5": "d120_16x5", "QX6": "d120_16x6",
         "QK1": "d120_16k1", "QK2": "d120_16k2", "QK3": "d120_16k3", "QK5": "d120_16k5",
         "QK1m": "d120_16k1m", "QK1n": "d120_16k1n", "QK1w": "d120_16k1w", "QK1o": "d120_16k1o",
         "QB0": "d120_16b0", "QBK1": "d120_16bk1", "QBK1s": "d120_16bk1s",
         "QBK1w": "d120_16bk1w", "QBK1x": "d120_16bk1x", "QBKV": "d120_16bkv",
         "QT1": "d120_16t1", "QT2": "d120_16t2",
         "QK1a": "d120_16k1a", "QK1b": "d120_16k1b", "QK1L": "d120_16k1l", "QF1a": "d120_16f1a", "QK1oa": "d120_16k1oa",
         "QK2": "d120_16k2", "QK2n": "d120_16k2n", "QK2i": "d120_16k2i", "QBK2": "d120_16bk2",
         "QBK2n": "d120_16bk2n", "QK2iw": "d120_16k2iw", "QK2i3": "d120_16k2i3", "QK2nw": "d120_16k2nw",
         "QK2n3": "d120_16k2n3", "QBK2i": "d120_16bk2i", "QBK2iw": "d120_16bk2iw",
         # plan_20261127_exterior.md: base metal (mysim/ext_heads.py)
         "QBK2iE0": "d120_16bk2ie0", "QBK2iE1": "d120_16bk2ie1", "QBK2iE1c": "d120_16bk2ie1c",
         "QK2iE1c": "d120_16k2ie1c", "QK2iE1": "d120_16k2ie1", "QK2iE0": "d120_16k2ie0",
         "QK2iE1m": "d120_16k2ie1m", "QK2iE1b": "d120_16k2ie1b", "QK2iE1cw": "d120_16k2ie1cw", "QK2iE1cr": "d120_16k2ie1cr"}
EXTRA = ("nec_cand.npy", "nec_molex.npy", "nec_molex2.npy", "nec_rod.npy", "nec_ext_a.npy", "nec_ext_b.npy", "nec_ext_c.npy",
         "nec_ext_d.npy", "nec_ext_e.npy", "nec_ext_f.npy", "nec_ext_g.npy")


def scales(band="2g4"):
    ring = np.load(os.path.join(HERE, "nec_ring.npy"), allow_pickle=True).item()
    rod = {}
    for f in ("nec_rod.npy",) + tuple(f for f in EXTRA if f.startswith("nec_ext_")):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            rod.update(np.load(fp, allow_pickle=True).item())
    s0 = float(ring[("d120_16", band)]["scale"])
    return {q: 20.0 * np.log10(float(rod[(k, band)]["scale"]) / s0) for q, k in HEADS.items() if (k, band) in rod}


def add_designs(band="2g4"):
    shift = scales(band)
    snr0 = F.ANCHOR["H4"][band][2]
    for q, k in HEADS.items():
        X.DESIGNS[q] = (k, "nod", dict(est_cls="EstFeat"))
        if q in shift:
            X.DESIGNS[q + "s"] = (k, "nod", dict(est_cls="EstFeat", snr_db=snr0 + shift[q]))
    if "QT1" in shift:      # plan_20261126_risk3.md section 3: the estimator believes in another table
        X.DESIGNS["QTM1s"] = ("d120_16t1", "nod", dict(est_cls="EstFeat", snr_db=snr0 + shift["QT1"], est_pat="d120_16k1_H4"))
        X.DESIGNS["QTM2s"] = ("d120_16t1", "nod", dict(est_cls="EstFeat", snr_db=snr0 + shift["QT1"], est_pat="d120_16t2_H4"))
    return shift


if __name__ == "__main__":
    if sys.argv[1] == "nec":
        print(len(X.build_nec_real(extra_files=EXTRA)))
    elif sys.argv[1] == "scales":
        band = sys.argv[2] if len(sys.argv) > 2 else "2g4"
        for q, d in scales(band).items():
            print(f"{q:5s} {HEADS[q]:14s} {band} reference horizon gain {d:+6.2f} dB vs d120_16")
    else:
        add_designs(sys.argv[3].split(",")[0])
        X.main()
