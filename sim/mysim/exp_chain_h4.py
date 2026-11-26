"""Receive-chain order and chain swap with the final anchor values (H4: anchor 3.5 dB BELOW the reference chain).
Earlier chain studies (exp_chain.py, exp_risk.py) assumed an anchor 11 dB ABOVE the reference chain.  Design
condition of plan sim/plan_20261126_real.md, P0, one risk at a time.  POST-HOC addition (after the first results of
the realistic condition).
Usage: python3 mysim/exp_chain_h4.py <scenes> <bands> <seeds> [names]"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_eval as E
import exp_final as F
import exp_real as X
import real_scenes as R
from risk_synth import synth_risk

RSS = dict(order_model="rss")


def variants(band, exam):
    out = {}
    def base(h="H4"):
        alpha, g, snr = F.ANCHOR[h][band]
        c = X.cond_cfg("design", band, exam, f"d120_16_{h}")
        c.update(anchor_alpha=alpha, snr_db=snr, synth_fn=synth_risk)
        return c
    b = base()
    out["base"] = dict(b, synth_kw={})
    # AX210 orders the CSI rows by RSS (stronger first)
    out["rss_ref"] = dict(b, synth_kw=dict(RSS, order_proc="naive"))                 # rule: stronger row = reference
    out["rss_ref_trim"] = dict(b, synth_kw=dict(RSS, order_proc="naive", agg="trim"))
    out["rss_sw"] = dict(b, synth_kw=dict(RSS, order_proc="stronger"))               # old rule: stronger row = switch chain
    out["rss_refdrop3"] = dict(b, synth_kw=dict(RSS, order_proc="refdrop", drop_margin_db=3.0))
    for pd in (6.0, 10.0):
        out[f"b{pd:.0f}"] = dict(b, synth_kw=dict(sw_pad_db=pd))                     # attenuator at the chain-B input alone
        out[f"rss_ref_b{pd:.0f}"] = dict(b, synth_kw=dict(RSS, order_proc="naive", sw_pad_db=pd))
    out["rss_ref_b10_trim"] = dict(b, synth_kw=dict(RSS, order_proc="naive", sw_pad_db=10.0, agg="trim"))
    for h in ("H5", "H1"):
        out[f"{h}"] = dict(base(h), synth_kw={})
        out[f"{h}_rss_ref"] = dict(base(h), synth_kw=dict(RSS, order_proc="naive"))
    # random chain swap of single packets
    out["sw10_mean"] = dict(b, synth_kw=dict(p_swap=0.10))
    out["sw10_trim"] = dict(b, synth_kw=dict(p_swap=0.10, agg="trim"))
    out["sw10_E"] = dict(b, synth_kw=dict(p_swap=0.10, swap_fix="energy"))
    out["sw30_trim"] = dict(b, synth_kw=dict(p_swap=0.30, agg="trim"))
    return out


def main():
    scenes, bands, seeds = sys.argv[1].split(","), sys.argv[2].split(","), [int(x) for x in sys.argv[3].split(",")]
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    tag = sys.argv[5] if len(sys.argv) > 5 else "chain"
    out = os.path.join(X.ROOT, "results", f"real_{tag}.pkl")
    for band in bands:
        for sc in scenes:
            V = variants(band, sc in R.EXAM)
            for nm, cfg in V.items():
                if names and nm not in names:
                    continue
                X.setup_scene(sc, "nod", cfg["pat"])
                E.run({nm: cfg}, [sc], [band], seeds, X.data_dir(sc, "design"), X.NEC_REAL, out, store_zone=False, verbose=True)
    print("CHAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
