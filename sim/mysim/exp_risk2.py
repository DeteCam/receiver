"""Follow-up of exp_risk.py after moving the AGC table correction behind the chain-swap check (it needs the chain
identity) and adding misread gain states (state_err_db): raw per-chain AGC (CSI levels equalised by the chip)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
from risk_synth import synth_risk
from exp_risk import S5, AGC7, NEC
E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
R = dict(S5, synth_fn=synth_risk)
RAWT = dict(AGC7, raw=True, fix="table")
CFG = {
    "S5": dict(S5),
    "sw50_mag_raw": dict(R, synth_kw=dict(p_swap=0.50, swap_fix="mag", agc=RAWT)),
    "sw50_E_raw": dict(R, synth_kw=dict(p_swap=0.50, swap_fix="energy", agc=RAWT)),
    "agcraw_tabE": dict(R, synth_kw=dict(agc=dict(RAWT, state_err_db=0.6))),
    "agcraw_tabE_trim": dict(R, synth_kw=dict(agc=dict(RAWT, state_err_db=0.6), agg="trim")),
    "agc15_tabE": dict(R, synth_kw=dict(agc=dict(AGC7, ph_sig_deg=15.0, fix="table", state_err_db=0.6))),
}
if __name__ == "__main__":
    E.run(CFG, E.DEV_SCENES + E.EVAL_SCENES, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC,
          "/opt/detecam/sim/results/exp_risk2.pkl", store_zone=False, verbose=False)
    print("RISK2_DONE", flush=True)
