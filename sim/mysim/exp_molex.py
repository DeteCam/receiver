"""Box-distance / antenna-cable heads (plan sim/plan_20261126_molex.md) on the exp_real pipeline.  [simulation]
Usage: python3 mysim/exp_molex.py nec                 -> rebuild mysim/nec_real.npy with nec_molex.npy added
       python3 mysim/exp_molex.py scales              -> reference horizon gain of each head vs d120_16
       python3 mysim/exp_molex.py <exp_real.py args>  -> exp_real.main() with the Q* designs added
Q* = d120_16 with the box top 58.9 / 43.9 mm below the element centre (G30 / G15; d120_16 itself is 88.9 mm) and/or
the 16 antenna cables (C).  The 's' copies keep the receiver noise fixed in absolute terms: the reference SNR moves
with the reference horizon gain of the head (plan section 3.1); the plain copies keep the reference SNR fixed."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_real as X  # noqa: E402
import exp_final as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = {"Q30": "d120_16g30", "Q15": "d120_16g15", "Q30C": "d120_16g30c", "Q15C": "d120_16g15c", "Q60C": "d120_16g60c",
         "QL1": "d120_16l1", "QL2": "d120_16l2",       # QL1 / QL2: POST-HOC cable layouts (molex_layouts.py, plan 3.2)
         "QR45": "d120_16r45",                          # POST-HOC diagnostic: reference raised 45 mm, no cables
         "QL3": "d120_16l3", "QI30": "d120_16i30",      # POST-HOC (plan 3.3): upside-down heads
         "QL5": "d120_16l5", "QL6": "d120_16l6"}        # POST-HOC (plan 3.4): reference as the 17th place on the ring (L6 upside down)


def scales(band="2g4"):
    ring = np.load(os.path.join(HERE, "nec_ring.npy"), allow_pickle=True).item()
    mol = {}
    for f in ("nec_molex.npy", "nec_molex2.npy"):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            mol.update(np.load(fp, allow_pickle=True).item())
    s0 = float(ring[("d120_16", band)]["scale"])
    return {q: 20.0 * np.log10(float(mol[(k, band)]["scale"]) / s0) for q, k in HEADS.items() if (k, band) in mol}


def add_designs():
    shift = scales()
    snr0 = F.ANCHOR["H4"]["2g4"][2]
    for q, k in HEADS.items():
        X.DESIGNS[q] = (k, "nod", dict(est_cls="EstFeat"))
        if q in shift:
            X.DESIGNS[q + "s"] = (k, "nod", dict(est_cls="EstFeat", snr_db=snr0 + shift[q]))
    return shift


if __name__ == "__main__":
    if sys.argv[1] == "nec":
        print(X.build_nec_real(extra_files=("nec_cand.npy", "nec_molex.npy", "nec_molex2.npy")))
    elif sys.argv[1] == "scales":
        for q, d in scales().items():
            print(f"{q:5s} {HEADS[q]:14s} reference horizon gain {d:+6.2f} dB vs d120_16")
    else:
        add_designs()
        X.main()
