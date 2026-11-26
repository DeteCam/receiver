"""Chain swap = RSS-ordered antenna selection of the AX210 (Liang et al., Ucom/GLOBECOM 2024; Zhang et al., IEEE
Sensors J. 2025): the first RF chain takes the antenna port with the larger RSS (integer dB), so the CSI rows come out
in RSS order.  The anchor makes the switch port the stronger one by design; this run checks how often the order still
flips, what the design rule + drop rule leave, and what a fixed attenuator on the reference port costs.
Recommended head (D120, 16 elements), Rev.2 + S5, 13 rooms, 2.4 / 5 GHz, seed 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from risk_synth import synth_risk
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
H = dict(pat="d120_16", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
R = dict(H, synth_fn=synth_risk)
RSS = dict(order_model="rss")
AGCRAW = dict(mode="indep", step_db=2.0, ph_sig_deg=7.0, amp_sig_db=0.3, lvl_jit_db=2.0, raw=True, fix="table",
              state_err_db=0.6)
CFG = {
    "base": dict(H),
    "naive": dict(R, synth_kw=dict(RSS, order_proc="naive")),
    "stronger": dict(R, synth_kw=dict(RSS, order_proc="stronger")),
    "drop3": dict(R, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=3.0)),
    "drop6": dict(R, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=6.0)),
    "pad6": dict(R, synth_kw=dict(ref_pad_db=6.0)),
    "pad6_drop3": dict(R, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=3.0, ref_pad_db=6.0)),
    "pad10": dict(R, synth_kw=dict(ref_pad_db=10.0)),
    "a2_drop3": dict(R, anchor_alpha=2.0, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=3.0)),
    "a5": dict(R, anchor_alpha=5.0),
    "a5_drop3": dict(R, anchor_alpha=5.0, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=3.0)),
    "drop3_agcraw_n5": dict(R, n_sweep=5, synth_kw=dict(RSS, order_proc="drop", drop_margin_db=3.0, agc=AGCRAW)),
    "agcraw_n5": dict(R, n_sweep=5, synth_kw=dict(agc=AGCRAW)),
}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    for k in CFG:
        E.run({k: CFG[k]}, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
              "/opt/detecam/sim/results/exp_chain.pkl", store_zone=False, verbose=False)
        print("done", k, flush=True)
    print("CHAIN_DONE", flush=True)
