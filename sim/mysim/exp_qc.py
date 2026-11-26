"""AI packet-quality filter: a classifier flags corrupted packets (overlapping transmitter, wrong switch-port label,
misread AGC state, chain order flip) from features a real receiver has (energy fields, ratio statistics, consistency
across sweeps), and flagged packets are left out of the per-port average.
1) per-packet data from synth_risk on 13 rooms x 2 bands (seed 0), D120 16-element head, corruption mix MIX
2) two classifiers, each trained on one half of the rooms; detection scored on the other half
3) localisation: no filter / rule filter / AI filter, every room scored with the classifier that never saw it"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import my_eval as E
from risk_synth import synth_risk
from pattern3d import make_pattern
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

AGCT = dict(mode="indep", step_db=2.0, ph_sig_deg=15.0, amp_sig_db=0.5, lvl_jit_db=2.0, raw=True, fix="table",
            state_err_db=0.6)
MIX = dict(order_model="rss", order_proc="stronger", collide=dict(p=0.10, sir_lo=0.0, sir_hi=15.0), p_mis=0.10, agc=AGCT)
ROOMS = E.DEV_SCENES + E.EVAL_SCENES
FOLD_A = set(E.DEV_SCENES[:2] + E.EVAL_SCENES[::2])
NEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nec_all.npy")
H = dict(pat="d120_16", exact_manifold=True, tc_step_ns="auto", sig_xy=0.05, sig_z=0.025, sig_head=1.5,
         synth_fn=synth_risk)
OUT = "/opt/detecam/sim/results/"


def gather():
    nec = np.load(NEC, allow_pickle=True).item()
    df = E.make_df(20e6)
    Xs, ys, rs = [], [], []
    for sc in ROOMS:
        for band in ("2g4", "5g"):
            D = np.load(f"/opt/detecam/sim/data/{sc}_{band}_s0.npy", allow_pickle=True).item()
            pat = make_pattern(nec[("d120_16", band)], float(D["fc"]))
            P = D["paths"]
            dyn = 45.0 if band == "2g4" else 90.0
            for r in range(len(D["cams"])):
                for n in range(4):
                    prn = (P["a"][r, n, 0], P["a"][r, n, 1], P["tau"][r, n], P["th"][r, n], P["ph"][r, n])
                    rng = np.random.default_rng(E.crc("qc", sc, band, r, n))
                    _, st = synth_risk(prn, D["cams"][r], D["nodes"][n], pat, df, rng, agg_mode="conj", dyn_std_deg=dyn,
                                       return_pkt=True, **MIX)
                    if "pkt" in st:
                        Xs.append(st["pkt"]["F"]); ys.append(st["pkt"]["y"]); rs += [sc] * len(st["pkt"]["y"])
    return np.vstack(Xs), np.concatenate(ys), np.array(rs)


if __name__ == "__main__":
    t0 = time.time()
    X, y, room = gather()
    print(f"packets {len(y)}, corrupted {y.mean():.3f}, {time.time() - t0:.0f}s", flush=True)
    inA = np.array([r in FOLD_A for r in room])
    models = {}
    for name, tr in (("A", inA), ("B", ~inA)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
        m.fit(X[tr], y[tr])
        te = ~tr
        p = m.predict_proba(X[te])[:, 1]
        fl = p >= 0.5
        rule = (X[te][:, 3] > 3.0) | (X[te][:, 0] < 3.0) | (X[te][:, 2] < 0.0)
        print(f"model trained on fold {name}: AUC {roc_auc_score(y[te], p):.3f}  AI recall {fl[y[te]].mean():.3f} "
              f"false-flag {fl[~y[te]].mean():.3f} | rule recall {rule[y[te]].mean():.3f} false-flag {rule[~y[te]].mean():.3f}",
              flush=True)
        models[name] = m
    with open(OUT + "qc_models.pkl", "wb") as f:
        pickle.dump(models, f)
    E.EXTRA_GROUPS = {"all4": (0, 1, 2, 3)}
    for fold_rooms, mname in ((sorted(FOLD_A), "B"), (sorted(set(ROOMS) - FOLD_A), "A")):
        cfg = {"clean": dict(H),
               "mix": dict(H, synth_kw=dict(MIX)),
               "mix_rule": dict(H, synth_kw=dict(MIX, qc_rule=True)),
               "mix_ai": dict(H, synth_kw=dict(MIX, qc_model=models[mname], qc_thr=0.5))}
        for k, v in cfg.items():
            E.run({k: v}, fold_rooms, ["2g4", "5g"], [0], "/opt/detecam/sim/data", NEC, OUT + "exp_qc.pkl",
                  store_zone=False, verbose=False)
            print("done", k, mname, flush=True)
    print("QC_DONE", flush=True)
