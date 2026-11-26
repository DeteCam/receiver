"""Exam rooms: true desk inside the top-1/top-3 suspicious desks, all desks vs alternate-seat occupancy.
Alternate seating = every other desk along a row is empty (rows run along the axis with more distinct desk
positions).  The ray-traced rooms keep every desk occupied, so bodies block more than in a real alternate-seat
exam (conservative); only the candidate set shrinks to the occupied desks the phone sees."""
import sys, os, pickle, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as W


def col_parity(ctr):
    nx = len(np.unique(np.round(ctr[:, 0], 1))); ny = len(np.unique(np.round(ctr[:, 1], 1)))
    a = 0 if nx >= ny else 1                       # row direction
    v = np.round(ctr[:, a], 1)
    u = np.unique(v)
    # merge values closer than 0.3 m (tiered rooms jitter)
    cols, last = [], None
    for x in u:
        if last is None or x - last > 0.3:
            cols.append(x)
        last = x
    idx = np.array([int(np.argmin(np.abs(np.array(cols) - x))) for x in v])
    return idx % 2


def run(seeds):
    acc = {}
    for s in seeds:
        f = f"/opt/detecam/sim/results/exp_wall_base_d120_16_seat10m{s}_pexam.pkl"
        if not os.path.exists(f):
            continue
        res = pickle.load(open(f, "rb"))
        for (sc, band, seed), (meta, o) in res.items():
            oc = o[f"d120_16_seat10m{s}"]
            par = col_parity(oc["desk_ctr"])
            td = oc["true_desk"]
            groups = {"2곳": W.BOARD.get(sc, "diag") + "A", "4곳": "quadmA", "8곳": "all8A"}
            for g, key in groups.items():
                M = oc["desk_mass"][key]
                for r in range(len(td)):
                    m = M[r].copy()
                    o_all = np.argsort(-m)
                    alt = np.where(par == par[td[r]], m, -1.0)
                    o_alt = np.argsort(-alt)
                    a = acc.setdefault(g, {"all1": [], "all3": [], "alt1": [], "alt3": [], "room": []})
                    a["all1"].append(o_all[0] == td[r]); a["all3"].append(td[r] in o_all[:3])
                    a["alt1"].append(o_alt[0] == td[r]); a["alt3"].append(td[r] in o_alt[:3])
                    a["room"].append(sc)
    for g, a in acc.items():
        room = np.array(a["room"])
        line = f"{g}: n={len(room)}  모든 책상 1위 {100*np.mean(a['all1']):.0f}% 3곳 {100*np.mean(a['all3']):.0f}%  |  한 칸씩 띄움 1위 {100*np.mean(a['alt1']):.0f}% 3곳 {100*np.mean(a['alt3']):.0f}%"
        per = " ".join(f"{r} {100*np.mean(np.array(a['alt3'])[room == r]):.0f}%" for r in sorted(set(room)))
        print(line, "\n    방별(띄움 3곳):", per)


if __name__ == "__main__":
    run([int(x) for x in sys.argv[1].split(",")])
