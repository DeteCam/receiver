"""Shared summary helpers for the 0.5 m design study.
Metrics (2-D camera position error, m):
  exam rooms  (classroom / lecture): the board-side receiver pair (the two receivers a proctor can mount at the front)
  other rooms (lodging / restroom / locker / bathhouse / metal room): diagonal pair, and all four corner positions
Reported as mean, median and the share of cameras within 0.5 m."""
import pickle
import numpy as np

BOARD = {"classroom_a": "adj", "lecture_a": "adj", "classroom_b": "p02", "lecture_b": "p02", "classroom_c": "p23"}
EXAM = ["classroom_a", "classroom_b", "lecture_a", "lecture_b"]
OTHER = ["lodging_a", "restroom_a", "locker_a", "bathhouse_a", "bathhouse_b", "lodging_b", "restroom_b", "locker_b",
         "worst_metal"]
DEV = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
FRESH_EXAM = ["classroom_c"]
FRESH_OTHER = ["lodging_c", "office_a", "restroom_c"]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def errs(res, cfg, scenes, key_fn, bands=("2g4", "5g"), seeds=None):
    """Concatenate position errors. key_fn(scene) -> pos_err key (e.g. 'adj_equal')."""
    out = []
    for k, (meta, o) in res.items():
        sc, band, seed = k
        if sc not in scenes or band not in bands or (seeds is not None and seed not in seeds) or cfg not in o:
            continue
        kk = key_fn(sc)
        if kk in o[cfg]["pos_err"]:
            out.append(o[cfg]["pos_err"][kk])
    return np.concatenate(out) if out else np.zeros(0)


def stat(e):
    if len(e) == 0:
        return dict(mean=np.nan, med=np.nan, w50=np.nan, n=0)
    return dict(mean=float(np.mean(e)), med=float(np.median(e)), w50=float(np.mean(e <= 0.5)), n=len(e))


def board_key(sc, suffix="_equal"):
    return BOARD[sc] + suffix


def triple(res, cfg, bands=("2g4", "5g"), seeds=None, exam=EXAM, other=OTHER, grp=None):
    """(exam board pair, other diag, other all4) stats.  grp: optional EXTRA_GROUPS name builder
    grp(pair_name) -> group key without '_equal' (e.g. lambda p: f'x10|{p}')."""
    if grp is None:
        kb, kd, ka = (lambda s: board_key(s)), (lambda s: "diag_equal"), (lambda s: "all4_equal")
    else:
        kb = lambda s: grp(BOARD[s]) + "_equal"
        kd = lambda s: grp("diag") + "_equal"
        ka = lambda s: grp("all4") + "_equal"
    return (stat(errs(res, cfg, exam, kb, bands, seeds)), stat(errs(res, cfg, other, kd, bands, seeds)),
            stat(errs(res, cfg, other, ka, bands, seeds)))


def fmt(t):
    return " | ".join(f"{d['mean']:.2f} ({100 * d['w50']:.0f}%)" for d in t)


def paired_delta(res, cfg_a, cfg_b, scenes, key_fn, bands=("2g4", "5g"), seeds=None, n_boot=2000, seed=0):
    """Mean difference b - a over the same cameras with a room-cluster bootstrap 95% interval."""
    da, db, room = [], [], []
    for k, (meta, o) in res.items():
        sc, band, sd = k
        if sc not in scenes or band not in bands or (seeds is not None and sd not in seeds):
            continue
        if cfg_a not in o or cfg_b not in o:
            continue
        kk = key_fn(sc)
        a, b = o[cfg_a]["pos_err"].get(kk), o[cfg_b]["pos_err"].get(kk)
        if a is None or b is None:
            continue
        da.append(a); db.append(b); room += [sc] * len(a)
    if not da:
        return np.nan, (np.nan, np.nan)
    a, b, room = np.concatenate(da), np.concatenate(db), np.array(room)
    d = b - a
    rooms = sorted(set(room))
    rng = np.random.default_rng(seed)
    idx = {r: np.where(room == r)[0] for r in rooms}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(rooms, len(rooms), replace=True)
        ii = np.concatenate([idx[r] for r in pick])
        bs.append(d[ii].mean())
    return float(d.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
