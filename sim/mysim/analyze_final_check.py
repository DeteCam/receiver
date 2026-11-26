"""Summaries for the final-configuration check (plan sim/plan_20261126_final.md).
python3 mysim/analyze_final_check.py outside      rooms outside exam rooms, nobody inside (nop), 2 positions
python3 mysim/analyze_final_check.py exam         exam rooms, suspicious-desk top-3 (all desks / alternate seats)
python3 mysim/analyze_final_check.py sanity       the same configs against the earlier full-node result files
python3 mysim/analyze_final_check.py json         compact numbers for the documents"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as W
from analyze_alt_seat import col_parity

R = "/opt/detecam/sim/results/"
OUT_F = R + "exp_final_outside_d0_d35_d0_snr14_d0_cal10.pkl"
EX_F = R + "exp_final_exam_seat_seat_snr14_seat_cal10.pkl"
OTYPES = ["숙소", "화장실", "목욕탕", "탈의실", "사무실", "금속 방"]


def load(f):
    with open(f, "rb") as fh:
        return pickle.load(fh)


def collect(res, cfg, group, seeds=None, bands=("2g4", "5g")):
    e, room, band, seed = [], [], [], []
    for (sc, b, s), (meta, o) in sorted(res.items()):
        if b not in bands or (seeds is not None and s not in seeds) or cfg not in o:
            continue
        k = f"{group}_equal"
        if k not in o[cfg]["pos_err"]:
            continue
        v = o[cfg]["pos_err"][k]
        e.append(v)
        room += [sc] * len(v)
        band += [b] * len(v)
        seed += [s] * len(v)
    if not e:
        return np.zeros(0), np.array([]), np.array([]), np.array([])
    return np.concatenate(e), np.array(room), np.array(band), np.array(seed)


def pct(x):
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def boot_pp(a, b, room, n=1000, seed=0):
    """paired difference of share <= 0.5 m (b - a, percentage points) with a room bootstrap 95 % interval"""
    d = (b <= 0.5).astype(float) - (a <= 0.5).astype(float)
    rooms = sorted(set(room))
    idx = {r: np.where(room == r)[0] for r in rooms}
    rng = np.random.default_rng(seed)
    bs = [100 * d[np.concatenate([idx[p] for p in rng.choice(rooms, len(rooms))])].mean() for _ in range(n)]
    return 100 * d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def stats(e):
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                p95=float(np.percentile(e, 95)), in05=pct(e <= 0.5), in10=pct(e <= 1.0))


def outside(show=True):
    res = load(OUT_F)
    seeds = sorted({k[2] for k in res})
    out = {"seeds": seeds}
    for cfg in ("d0", "d35", "d0_snr14", "d0_cal10"):
        for g in ("nod_diagA", "diagA", "nod_madjA", "madjA", "nod_moppA", "moppA"):
            e, room, band, seed = collect(res, cfg, g)
            if not len(e):
                continue
            typ = np.array([W.TYPE[r] for r in room])
            d = dict(all=stats(e), types={}, seeds={}, bands={}, rooms={})
            for t in OTYPES:
                m = typ == t
                if m.any():
                    d["types"][t] = stats(e[m])
            for s in seeds:
                m = seed == s
                if m.any():
                    d["seeds"][int(s)] = stats(e[m])
            for b in ("2g4", "5g"):
                d["bands"][b] = stats(e[band == b])
            for r in sorted(set(room)):
                d["rooms"][r] = stats(e[room == r])
            d["pass"] = all(v["in05"] >= 80.0 for v in d["types"].values())
            out[f"{cfg}|{g}"] = d
            if show:
                a = d["all"]
                print(f"\n## {cfg} {g} seeds {seeds} n={a['n']}: 평균 {a['mean']:.2f} 중앙 {a['median']:.2f} 90% {a['p90']:.2f} "
                      f"95% {a['p95']:.2f} | 0.5 m 안 {a['in05']:.0f}% 1 m 안 {a['in10']:.0f}% | 합격(종류마다 80%) {d['pass']}")
                print("   종류: " + " · ".join(f"{t} {v['in05']:.0f}%/{v['in10']:.0f}% (중앙 {v['median']:.2f})" for t, v in d["types"].items()))
                print("   시드: " + " · ".join(f"s{s} {v['in05']:.0f}%" for s, v in d["seeds"].items())
                      + " | 대역: " + " · ".join(f"{b} {v['in05']:.0f}%" for b, v in d["bands"].items()))
                worst = sorted(d["rooms"].items(), key=lambda kv: kv[1]["in05"])[:3]
                print("   가장 나쁜 방(0.5 m 안): " + ", ".join(f"{r} {v['in05']:.0f}% (평균 {v['mean']:.2f})" for r, v in worst))
    # paired differences against the main condition
    print("\n## 짝지은 차이 (같은 카메라, 0.5 m 안 비율 %p, 방 부트스트랩 95%)")
    for g in ("nod_diagA", "diagA", "nod_moppA", "nod_madjA"):
        ea, room, _, _ = collect(res, "d0", g)
        for cfg in ("d35", "d0_snr14", "d0_cal10"):
            eb, rb, _, _ = collect(res, cfg, g)
            if len(eb) == len(ea) and len(ea):
                dd, lo, hi = boot_pp(ea, eb, room)
                dm = W.paired(ea, eb, room)
                print(f"   {g:10s} {cfg:9s} {dd:+.1f}%p [{lo:+.1f}, {hi:+.1f}] · 평균 오차 {dm[0]:+.3f} m [{dm[1]:+.3f}, {dm[2]:+.3f}]")
                out[f"diff|{g}|{cfg}"] = dict(pp=dd, lo=lo, hi=hi, dmean=dm[0], dlo=dm[1], dhi=dm[2])
    # nod gain
    for g0 in ("diagA", "madjA", "moppA"):
        ea, room, _, _ = collect(res, "d0", g0)
        eb, _, _, _ = collect(res, "d0", "nod_" + g0)
        if len(ea) and len(eb) == len(ea):
            dd, lo, hi = boot_pp(ea, eb, room)
            print(f"   끄덕이기 이득 {g0}: {dd:+.1f}%p [{lo:+.1f}, {hi:+.1f}]")
            out[f"nodgain|{g0}"] = dict(pp=dd, lo=lo, hi=hi)
    return out


def exam(show=True):
    res = load(EX_F)
    seeds = sorted({k[2] for k in res})
    out = {"seeds": seeds}
    for cfg in ("seat", "seat_snr14", "seat_cal10"):
        for pre in ("nod_", ""):
            acc = {"all1": [], "all3": [], "alt1": [], "alt3": [], "room": [], "seed": []}
            for (sc, band, s), (meta, o) in sorted(res.items()):
                if cfg not in o:
                    continue
                oc = o[cfg]
                key = pre + W.BOARD[sc] + "A"
                if key not in oc["desk_mass"]:
                    continue
                par = col_parity(oc["desk_ctr"])
                td = oc["true_desk"]
                M = oc["desk_mass"][key]
                for r in range(len(td)):
                    m = M[r]
                    o_all = np.argsort(-m)
                    alt = np.where(par == par[td[r]], m, -1.0)
                    o_alt = np.argsort(-alt)
                    acc["all1"].append(o_all[0] == td[r]); acc["all3"].append(td[r] in o_all[:3])
                    acc["alt1"].append(o_alt[0] == td[r]); acc["alt3"].append(td[r] in o_alt[:3])
                    acc["room"].append(sc); acc["seed"].append(s)
            if not acc["room"]:
                continue
            room = np.array(acc["room"]); sd = np.array(acc["seed"])
            d = {k: pct(np.array(acc[k])) for k in ("all1", "all3", "alt1", "alt3")}
            d["n"] = len(room)
            d["rooms"] = {r: pct(np.array(acc["alt3"])[room == r]) for r in sorted(set(room))}
            d["seeds"] = {int(s): pct(np.array(acc["alt3"])[sd == s]) for s in seeds}
            d["raw_alt3"] = np.array(acc["alt3"], dtype=float)
            d["room_arr"] = room
            out[f"{cfg}|{pre}board"] = d
            if show:
                print(f"\n## {cfg} {pre}board seeds {seeds} n={d['n']}: 모든 책상 1위 {d['all1']:.0f}% 3곳 {d['all3']:.0f}% | "
                      f"한 칸씩 띄움 1위 {d['alt1']:.0f}% 3곳 {d['alt3']:.0f}%")
                print("   방별(띄움 3곳): " + " · ".join(f"{r} {v:.0f}%" for r, v in d["rooms"].items()))
                print("   시드별(띄움 3곳): " + " · ".join(f"s{s} {v:.0f}%" for s, v in d["seeds"].items()))
    print("\n## 짝지은 차이 (띄움 3곳 비율 %p, 방 부트스트랩 95%)")
    for pre in ("nod_", ""):
        a = out.get(f"seat|{pre}board")
        for cfg in ("seat_snr14", "seat_cal10"):
            b = out.get(f"{cfg}|{pre}board")
            if a is None or b is None or len(a["raw_alt3"]) != len(b["raw_alt3"]):
                continue
            dvec = b["raw_alt3"] - a["raw_alt3"]
            room = a["room_arr"]
            rooms = sorted(set(room)); idx = {r: np.where(room == r)[0] for r in rooms}
            rng = np.random.default_rng(0)
            bs = [100 * dvec[np.concatenate([idx[p] for p in rng.choice(rooms, len(rooms))])].mean() for _ in range(1000)]
            print(f"   {pre}board {cfg}: {100*dvec.mean():+.1f}%p [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")
            out[f"diff|{pre}board|{cfg}"] = dict(pp=100 * dvec.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
    a, b = out.get("seat|board"), out.get("seat|nod_board")
    if a is not None and b is not None and len(a["raw_alt3"]) == len(b["raw_alt3"]):
        dvec = b["raw_alt3"] - a["raw_alt3"]
        print(f"   끄덕이기 이득(띄움 3곳): {100*dvec.mean():+.1f}%p")
        out["nodgain|board"] = dict(pp=100 * dvec.mean())
    return out


def sanity():
    """d35 on nop / seat on base must equal the earlier full-node runs for the same cameras."""
    res = load(OUT_F)
    old = W.load("nop", "d120_16")
    for (sc, b, s), (meta, o) in sorted(res.items()):
        if s != 0 or (sc, b, s) not in old:
            continue
        a = o["d35"]["pos_err"]["diagA_equal"]
        c = old[(sc, b, s)][1]["d120_16"]["pos_err"]["diagA_equal"]
        print(f"nop {sc} {b}: max|diff| {np.max(np.abs(a - c)):.2e}")
    ex = load(EX_F)
    for s in (0, 1, 2):
        f = R + f"exp_wall_base_d120_16_seat10m{s}_pexam.pkl"
        if not os.path.exists(f):
            continue
        old = load(f)
        for (sc, b, ss), (meta, o) in sorted(ex.items()):
            if ss != s or (sc, b, ss) not in old:
                continue
            k = W.BOARD[sc] + "A"
            a = o["seat"]["desk_mass"][k]
            c = old[(sc, b, ss)][1][f"d120_16_seat10m{s}"]["desk_mass"][k]
            print(f"exam {sc} {b} s{s}: max|diff| desk mass {np.max(np.abs(a - c)):.2e}")


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "outside":
        outside()
    elif w == "exam":
        exam()
    elif w == "sanity":
        sanity()
    elif w == "json":
        o = outside(False)
        x = exam(False)
        for d in x.values():
            if isinstance(d, dict):
                d.pop("raw_alt3", None); d.pop("room_arr", None)
        print(json.dumps({"outside": o, "exam": x}, ensure_ascii=False, indent=1, default=float))


def anchor(show=True):
    """section 10: anchor coupler with real part values; compared with the idealised main run (alpha 3.5, g 1)"""
    import glob
    main_out = load(OUT_F)
    res_o = load(os.environ.get("ANCHOR_OUT", R + "exp_final_anchor_outside.pkl"))
    res_e = {}
    for f in sorted(glob.glob(R + "exp_final_anchor_exam_*.pkl")):
        for k, v in load(f).items():
            if k in res_e:
                res_e[k][1].update(v[1])
            else:
                res_e[k] = v
    main_ex = load(EX_F) if os.path.exists(EX_F) else {}
    out = {}
    for h in ("H1", "H2", "H4", "H5"):
        # sign recovery over the evaluated nodes (outside), and position error of the pre-registered pairs
        sa = []
        for (sc, b, s), (meta, o) in sorted(res_o.items()):
            if h in o:
                m = o[h]["sign_acc"]
                sa.append(m[:, m.sum(axis=0) > 0].ravel())
        d = {}
        if sa:
            sa = np.concatenate(sa)
            d["sign_mean"] = float(sa.mean()); d["sign_p10"] = float(np.percentile(sa, 10))
        for g in ("nod_diagA", "nod_moppA"):
            ea, room, band, seed = collect(main_out, "d0", g)
            eb, rb, bb, sb = collect(res_o, h, g)
            if len(eb) and len(eb) == len(ea):
                typ = np.array([W.TYPE[r] for r in rb])
                dd, lo, hi = boot_pp(ea, eb, room)
                d[g] = dict(stats(eb), types={t: pct(eb[typ == t] <= 0.5) for t in OTYPES if (typ == t).any()},
                            pp=dd, lo=lo, hi=hi, seeds=sorted(set(int(x) for x in sb)))
        # exam: alternate-seat top-3 against the idealised main run
        if main_ex:
            for pre in ("nod_", ""):
                a3, b3, rooms = [], [], []
                for (sc, band, s), (meta, o) in sorted(res_e.items()):
                    if h not in o or (sc, band, s) not in main_ex:
                        continue
                    key = pre + W.BOARD[sc] + "A"
                    ob, oa = o[h], main_ex[(sc, band, s)][1]["seat"]
                    if key not in ob["desk_mass"] or key not in oa["desk_mass"]:
                        continue
                    par = col_parity(ob["desk_ctr"]); td = ob["true_desk"]
                    for r in range(len(td)):
                        for M, acc in ((oa["desk_mass"][key], a3), (ob["desk_mass"][key], b3)):
                            alt = np.where(par == par[td[r]], M[r], -1.0)
                            acc.append(td[r] in np.argsort(-alt)[:3])
                        rooms.append(sc)
                if b3:
                    a3 = np.array(a3, float); b3 = np.array(b3, float); rooms = np.array(rooms)
                    dvec = b3 - a3
                    rr = sorted(set(rooms)); idx = {r: np.where(rooms == r)[0] for r in rr}
                    rng = np.random.default_rng(0)
                    bs = [100 * dvec[np.concatenate([idx[p] for p in rng.choice(rr, len(rr))])].mean() for _ in range(1000)]
                    d[f"exam_{pre}board"] = dict(alt3=pct(b3), base=pct(a3), pp=100 * dvec.mean(), lo=np.percentile(bs, 2.5),
                                                 hi=np.percentile(bs, 97.5), n=int(len(b3)),
                                                 rooms={r: pct(b3[rooms == r]) for r in rr})
        out[h] = d
        if show:
            print(f"\n## {h}: 부호 복원 평균 {d.get('sign_mean', float('nan')):.3f} (하위 10% {d.get('sign_p10', float('nan')):.2f})")
            for g in ("nod_diagA", "nod_moppA"):
                if g in d:
                    x = d[g]
                    print(f"   {g}: 0.5 m 안 {x['in05']:.0f}% 1 m 안 {x['in10']:.0f}% 중앙 {x['median']:.2f} (이상값 대비 {x['pp']:+.1f}%p [{x['lo']:+.1f}, {x['hi']:+.1f}]) seeds {x['seeds']}"
                          + " | " + " ".join(f"{t}{v:.0f}%" for t, v in x["types"].items()))
            for pre in ("nod_", ""):
                k = f"exam_{pre}board"
                if k in d:
                    x = d[k]
                    print(f"   시험실 {pre}board 띄움 3곳 {x['alt3']:.0f}% (이상값 {x['base']:.0f}%, {x['pp']:+.1f}%p [{x['lo']:+.1f}, {x['hi']:+.1f}]) n={x['n']} | "
                          + " ".join(f"{r} {v:.0f}%" for r, v in x["rooms"].items()))
    # sign recovery of the idealised main run for reference
    sa = []
    for (sc, b, s), (meta, o) in sorted(main_out.items()):
        m = o["d0"]["sign_acc"]
        sa.append(m[:, m.sum(axis=0) > 0].ravel())
    sa = np.concatenate(sa)
    out["ideal_sign_mean"] = float(sa.mean())
    if show:
        print(f"\n이상값(α 3.5, 손실 0) 부호 복원 평균 {sa.mean():.3f}")
    return out


if __name__ == "__main__" and sys.argv[1] == "anchor":
    anchor()


def pair_compare():
    """pre-registered primary (diag) vs secondary pairs (mopp, madj), same cameras, d0 with nod"""
    res = load(OUT_F)
    ea, room, band, seed = collect(res, "d0", "nod_diagA")
    out = {}
    for g in ("nod_moppA", "nod_madjA"):
        eb, _, _, _ = collect(res, "d0", g)
        dd, lo, hi = boot_pp(ea, eb, room)
        typ = np.array([W.TYPE[r] for r in room])
        per_t = {t: 100 * (np.mean(eb[typ == t] <= 0.5) - np.mean(ea[typ == t] <= 0.5)) for t in OTYPES}
        per_s = {int(s): 100 * (np.mean(eb[seed == s] <= 0.5) - np.mean(ea[seed == s] <= 0.5)) for s in sorted(set(seed))}
        print(f"{g} − nod_diagA: {dd:+.1f}%p [{lo:+.1f}, {hi:+.1f}] | 종류 " + " ".join(f"{t}{v:+.0f}" for t, v in per_t.items())
              + " | 시드 " + " ".join(f"s{s}{v:+.0f}" for s, v in per_s.items()))
        out[g] = dict(pp=dd, lo=lo, hi=hi, types=per_t, seeds=per_s)
    return out


if __name__ == "__main__" and sys.argv[1] == "pairs":
    pair_compare()
