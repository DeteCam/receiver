"""Summaries for the wall-mounted receiver study (plan sim/plan_20261126_wall.md).
python3 mysim/analyze_wall.py heads [seeds]      head designs, 2/3/4 positions, heights A/B
python3 mysim/analyze_wall.py variants <cfg>      content variants base / nop / crowd / bare (seed 0)
python3 mysim/analyze_wall.py nod <cfg>           nodded looks vs single look"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

R = "/opt/detecam/sim/results/"
FRESH = ["classroom_c", "lodging_c", "office_a", "restroom_c"]
TYPE = {"classroom_a": "시험실", "classroom_b": "시험실", "lecture_a": "시험실", "lecture_b": "시험실", "classroom_c": "시험실",
        "lodging_a": "숙소", "lodging_b": "숙소", "lodging_c": "숙소", "restroom_a": "화장실", "restroom_b": "화장실",
        "restroom_c": "화장실", "bathhouse_a": "목욕탕", "bathhouse_b": "목욕탕", "locker_a": "탈의실", "locker_b": "탈의실",
        "office_a": "사무실", "worst_metal": "금속 방"}
TYPES = ["시험실", "숙소", "화장실", "목욕탕", "탈의실", "사무실", "금속 방"]
BOARD = {"classroom_a": "c01", "lecture_a": "c01", "classroom_b": "c03", "lecture_b": "c03", "classroom_c": "c23"}
TRI = {"classroom_a": "c01m2", "lecture_a": "c01m2", "classroom_b": "c03m1", "lecture_b": "c03m1", "classroom_c": "c23m0"}


KBOARD = {"classroom_a": "k01", "lecture_a": "k01", "classroom_b": "k03", "lecture_b": "k03", "classroom_c": "k23"}
KTRI = {"classroom_a": "k012", "lecture_a": "k012", "classroom_b": "k032", "lecture_b": "k032", "classroom_c": "k230"}


def gname(kind, sc, corner=False):
    """pre-registered group for a room: two = board pair (exam) / diagonal corners (others), three, four."""
    if corner:
        if kind == "two":
            return KBOARD.get(sc, "kdiag")
        if kind == "three":
            return KTRI.get(sc, "k012")
        return {"quadc": "kquad", "four": "kquad"}.get(kind, kind)
    if kind == "two":
        return BOARD.get(sc, "diag")
    if kind == "three":
        return TRI.get(sc, "c01m2")
    return kind


def load(variant, cfg, aug=None):
    tag = f"{variant}_{cfg}" + (f"_{aug}" if aug else "")
    res = {}
    for f in sorted(glob.glob(R + f"exp_wall_{tag}_p*.pkl")):
        with open(f, "rb") as fh:
            res.update(pickle.load(fh))
    return res


def collect(res, cfg, kind, h, bands=("2g4", "5g"), seeds=None, scenes=None, prefix="", corner=False):
    e, room, cams = [], [], []
    for (sc, band, sd), (meta, o) in sorted(res.items()):
        if band not in bands or (seeds is not None and sd not in seeds) or (scenes is not None and sc not in scenes):
            continue
        if cfg not in o:
            continue
        k = f"{prefix}{gname(kind, sc, corner)}{h}_equal"
        if k not in o[cfg]["pos_err"]:
            continue
        v = o[cfg]["pos_err"][k]
        e.append(v)
        room += [sc] * len(v)
        cams += [(sc, band, sd, i) for i in range(len(v))]
    if not e:
        return np.zeros(0), np.zeros(0, dtype=object), []
    return np.concatenate(e), np.array(room), cams


def st(e):
    if len(e) == 0:
        return "—"
    return (f"평균 {e.mean():.2f} · 중앙 {np.median(e):.2f} · 90% {np.percentile(e, 90):.2f} · 95% {np.percentile(e, 95):.2f}"
            f" · 0.5 m 안 {100 * np.mean(e <= 0.5):.0f}%")


def short(e):
    if len(e) == 0:
        return "—"
    return f"{e.mean():.2f}/{np.percentile(e, 90):.2f} ({100 * np.mean(e <= 0.5):.0f}%)"


def by_type(e, room):
    out = {}
    for t in TYPES:
        m = np.array([TYPE[r] == t for r in room])
        if m.any():
            out[t] = e[m]
    return out


def paired(ea, eb, room, n_boot=1000, seed=0):
    d = eb - ea
    rooms = sorted(set(room))
    idx = {r: np.where(room == r)[0] for r in rooms}
    rng = np.random.default_rng(seed)
    bs = [d[np.concatenate([idx[p] for p in rng.choice(rooms, len(rooms))])].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def heads(seeds=None):
    cfgs = ["d120_16", "semi16", "semi16s", "plan2x8"]
    R0 = {c: load("base", c) for c in cfgs}
    for h in "AB":
        print(f"\n===== height {h} ({'2.0' if h == 'A' else '1.5'} m)  mean/p90 (share <= 0.5 m), both bands")
        for kind in ("two", "three", "quadc", "quadm", "madj", "mopp", "all8"):
            print(f"-- {kind}")
            for c in cfgs:
                e, room, _ = collect(R0[c], c, kind, h, seeds=seeds)
                if len(e) == 0:
                    continue
                bt = by_type(e, room)
                print(f"  {c:9s} all {short(e):20s} | " + " | ".join(f"{t} {short(v)}" for t, v in bt.items()))
    # paired deltas vs d120_16 (two positions, height A)
    print("\n== paired delta vs d120_16 (two positions / four corner positions, height A, room bootstrap 95%)")
    for kind in ("two", "quadc"):
        ea, room, ca = collect(R0["d120_16"], "d120_16", kind, "A", seeds=seeds)
        for c in cfgs[1:]:
            eb, _, cb = collect(R0[c], c, kind, "A", seeds=seeds)
            if len(eb) == len(ea) and len(ea):
                d, lo, hi = paired(ea, eb, room)
                print(f"  {kind:6s} {c:9s} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")


def full_table(variant, cfg, kinds=("two", "three", "quadc", "quadm"), hs="AB", seeds=None, aug=None, prefix=""):
    res = load(variant, cfg, aug)
    for h in hs:
        for kind in kinds:
            e, room, _ = collect(res, cfg, kind, h, seeds=seeds, prefix=prefix)
            if len(e) == 0:
                continue
            print(f"\n[{variant} {cfg} {prefix}{kind} {h}] {st(e)}")
            worst = sorted(((np.mean(e[room == r]), r) for r in set(room)), reverse=True)[:2]
            for t, v in by_type(e, room).items():
                print(f"   {t:5s} {st(v)}")
            print("   가장 나쁜 방: " + ", ".join(f"{r} {m:.2f}" for m, r in worst))
            for b in ("2g4", "5g"):
                eb, _, _ = collect(res, cfg, kind, h, bands=(b,), seeds=seeds, prefix=prefix)
                print(f"   {b}: {st(eb)}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "heads":
        seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
        heads(seeds)
    elif what == "full":
        full_table(sys.argv[2], sys.argv[3], seeds=[int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None)
