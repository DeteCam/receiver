"""Hardware / driver risks without hardware: each risk switched on alone on top of the Rev.2 + S5 procedure,
with and without its software counter-measure (paired: same rooms, cameras, noise and people draws).
Base corner data, 13 rooms (4 development + 9 evaluation), 2.4 / 5 GHz, seeds 0 and 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from risk_synth import synth_risk
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
S5 = dict(pat="spec002plus_d85_dz0", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5)
R = dict(S5, synth_fn=synth_risk)
AGC7 = dict(mode="indep", step_db=2.0, ph_sig_deg=7.0, amp_sig_db=0.3, lvl_jit_db=2.0)
CFG = {
    "S5": dict(S5),
    "S5_ratio": dict(S5, agg_mode="ratio"),
    # chain swap
    "sw10": dict(R, synth_kw=dict(p_swap=0.10)),
    "sw50_E": dict(R, synth_kw=dict(p_swap=0.50, swap_fix="energy")),
    "sw50_mag_raw": dict(R, synth_kw=dict(p_swap=0.50, swap_fix="mag", agc=dict(AGC7, raw=True, fix="table"))),
    "sw50_E_raw": dict(R, synth_kw=dict(p_swap=0.50, swap_fix="energy", agc=dict(AGC7, raw=True, fix="table"))),
    # AGC gain-state phase / gain errors
    "agc7": dict(R, synth_kw=dict(agc=AGC7)),
    "agc15": dict(R, synth_kw=dict(agc=dict(AGC7, ph_sig_deg=15.0))),
    "agc7_joint": dict(R, synth_kw=dict(agc=dict(AGC7, mode="joint"))),
    "a2": dict(R, anchor_alpha=2.0),
    "agc7_a2": dict(R, anchor_alpha=2.0, synth_kw=dict(agc=AGC7)),
    "agc15_a2": dict(R, anchor_alpha=2.0, synth_kw=dict(agc=dict(AGC7, ph_sig_deg=15.0))),
    "agc7_a15": dict(R, anchor_alpha=1.5, synth_kw=dict(agc=AGC7)),
    "agc7_tab": dict(R, synth_kw=dict(agc=dict(AGC7, fix="table"))),
    "agc15_tab": dict(R, synth_kw=dict(agc=dict(AGC7, ph_sig_deg=15.0, fix="table"))),
    "agcraw": dict(R, synth_kw=dict(agc=dict(AGC7, raw=True))),
    "agcraw_tab": dict(R, synth_kw=dict(agc=dict(AGC7, raw=True, fix="table"))),
    "agcraw_joint": dict(R, synth_kw=dict(agc=dict(AGC7, raw=True, mode="joint"))),
    # packet -> port label errors (switch timing)
    "trim": dict(R, synth_kw=dict(agg="trim")),
    "median": dict(R, synth_kw=dict(agg="median")),
    "mis5": dict(R, synth_kw=dict(p_mis=0.05)),
    "mis5_trim": dict(R, synth_kw=dict(p_mis=0.05, agg="trim")),
    "mis5_guard": dict(R, synth_kw=dict(p_mis=0.05, mis_guard=True)),
    "mis20": dict(R, synth_kw=dict(p_mis=0.20)),
    "mis20_trim": dict(R, synth_kw=dict(p_mis=0.20, agg="trim")),
    "mis20_guard_trim": dict(R, synth_kw=dict(p_mis=0.20, mis_guard=True, agg="trim")),
    # quiet cameras (few packets per port)
    "n1": dict(S5, n_sweep=1), "n2": dict(S5, n_sweep=2), "n3": dict(S5, n_sweep=3), "n5": dict(S5, n_sweep=5),
    # switch isolation, anchor calibration drift
    "leak30": dict(R, synth_kw=dict(leak_db=-30.0)),
    "leak20": dict(R, synth_kw=dict(leak_db=-20.0)),
    "acal10": dict(R, synth_kw=dict(alpha_err_deg=10.0, alpha_err_db=1.0)),
    # capture tool reads I/Q in the wrong order
    "iqconj": dict(R, synth_kw=dict(iq_conj=True)),
    "iqconj_st": dict(R, synth_kw=dict(iq_conj=True, iq_selftest=True)),
    "iqok_st": dict(R, synth_kw=dict(iq_selftest=True)),
    "iqconj_st5": dict(R, synth_kw=dict(iq_conj=True, iq_selftest=True, anchor_delay_ns=5.0)),
}
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], seeds, "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_risk.pkl", store_zone=False, verbose=False)
    print("RISK_DONE", flush=True)
