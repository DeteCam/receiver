"""Final summaries for the wall-mounted receiver study (seeds 0-2, near-field wall coupling, content variants).
python3 mysim/analyze_final.py seeds [cfg]     pre-registered groups with all available seeds, per space type
python3 mysim/analyze_final.py nf              wall near-field mismatch configs vs the matched Thevenin baseline
python3 mysim/analyze_final.py variants        base / nop / crowd / bare (seed 0), paired against base
python3 mysim/analyze_final.py chart           per space type share <= 0.5 m for the s1 chart (json)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as W

KINDS = [("two", "A", "", "2곳"), ("three", "A", "", "3곳"), ("quadm", "A", "", "4곳(벽 가운데)"),
         ("quadc", "A", "", "4곳(모서리 옆)"), ("all8", "A", "", "8곳"), ("quadm", "AB", "", "4곳 × 두 높이"),
         ("two", "A", "nod_", "2곳 + 끄덕"), ("quadm", "A", "nod_", "4곳 + 끄덕"), ("all8", "A", "nod_", "8곳 + 끄덕")]


def res_for(variant, cfg, prefix):
    return W.load(variant, cfg, "nod" if prefix else None)


def seeds_in(res):
    return sorted({k[2] for k in res})


def line(e, room):
    worst = max(((np.mean(e[room == r]), r) for r in set(room)))
    return W.st(e) + f" · 최악 {worst[1]} {worst[0]:.2f}"


def seeds_table(cfg="d120_16", kinds=KINDS):
    for kind, h, pre, lab in kinds:
        res = res_for("base", cfg, pre)
        sd = seeds_in(res)
        e, room, _ = W.collect(res, cfg, kind, h, seeds=sd, prefix=pre)
        if not len(e):
            continue
        print(f"\n## {lab} [{cfg}, seeds {sd}, n={len(e)}]\n  전체 {line(e, room)}")
        for t, v in W.by_type(e, room).items():
            m = np.array([W.TYPE[r] == t for r in room])
            print(f"  {t:5s} {line(v, room[m])}")
        for b in ("2g4", "5g"):
            eb, _, _ = W.collect(res, cfg, kind, h, bands=(b,), seeds=sd, prefix=pre)
            print(f"  {b}: {W.st(eb)}")
        # seed spread of the overall mean
        ms = []
        for s in sd:
            es, _, _ = W.collect(res, cfg, kind, h, seeds=[s], prefix=pre)
            ms.append((s, es.mean(), 100 * np.mean(es <= 0.5)))
        print("  시드별: " + ", ".join(f"s{s} {m:.2f} m ({p:.0f}%)" for s, m, p in ms))


def nf():
    base = W.load("base", "d120_16_fv")
    for kind, h, lab in (("two", "A", "2곳"), ("quadm", "A", "4곳"), ("all8", "A", "8곳")):
        ea, room, ca = W.collect(base, "d120_16_fv", kind, h, seeds=[0])
        if not len(ea):
            continue
        print(f"\n## {lab}: matched Thevenin baseline d120_16_fv {W.short(ea)}")
        for c, txt in (("d120_16_wpf", "석고보드 앞, 자유공간 보정"), ("d120_16_wcf", "콘크리트 앞, 자유공간 보정"),
                       ("d120_16_wmf", "금속 앞, 자유공간 보정"), ("d120_16_wcc", "콘크리트 앞, 콘크리트 판 보정"),
                       ("d120_16_wmc", "금속 앞, 콘크리트 판 보정")):
            r = W.load("base", c)
            eb, _, cb = W.collect(r, c, kind, h, seeds=[0])
            if len(eb) != len(ea):
                continue
            d, lo, hi = W.paired(ea, eb, room)
            bt = W.by_type(eb, room)
            print(f"  {txt:22s} {W.short(eb):20s} 차이 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  " +
                  " ".join(f"{t}{100*np.mean(v<=0.5):.0f}%" for t, v in bt.items()))
            for b in ("2g4", "5g"):
                ea_b, room_b, _ = W.collect(base, "d120_16_fv", kind, h, bands=(b,), seeds=[0])
                eb_b, _, _ = W.collect(r, c, kind, h, bands=(b,), seeds=[0])
                d2, lo2, hi2 = W.paired(ea_b, eb_b, room_b)
                print(f"      {b}: {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")


def variants(cfg="d120_16", pre=""):
    base = res_for("base", cfg, pre)
    for kind, h, lab in (("two", "A", "2곳"), ("quadm", "A", "4곳"), ("all8", "A", "8곳"), ("quadm", "AB", "4곳 × 두 높이")):
        ea, room, ca = W.collect(base, cfg, kind, h, seeds=[0], prefix=pre)
        if not len(ea):
            continue
        print(f"\n## {pre}{lab}: base {W.short(ea)}  " + " ".join(f"{t}{100*np.mean(v<=0.5):.0f}%" for t, v in W.by_type(ea, room).items()))
        for v, txt in (("nop", "사람 없음"), ("crowd", "사람 많음"), ("bare", "사람·물건 없음")):
            r = res_for(v, cfg, pre)
            eb, rb, cb = W.collect(r, cfg, kind, h, seeds=[0], prefix=pre)
            if len(eb) != len(ea):
                print(f"  {txt}: (아직 없음 {len(eb)}/{len(ea)})")
                continue
            d, lo, hi = W.paired(ea, eb, room)
            bt = W.by_type(eb, room)
            print(f"  {txt:10s} {W.short(eb):20s} 차이 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  " +
                  " ".join(f"{t}{100*np.mean(x<=0.5):.0f}%" for t, x in bt.items()) +
                  f" · 최악 {max(((np.mean(eb[rb == q]), q) for q in set(rb)))[1]}")


def chart(cfg="d120_16"):
    out = {}
    for kind, h, pre, lab in [k for k in KINDS if k[3] in ("2곳", "4곳(벽 가운데)", "4곳 + 끄덕", "4곳 × 두 높이", "8곳", "8곳 + 끄덕")]:
        res = res_for("base", cfg, pre)
        sd = seeds_in(res)
        e, room, _ = W.collect(res, cfg, kind, h, seeds=sd, prefix=pre)
        out[lab] = {"seeds": sd, **{t: round(100 * float(np.mean(v <= 0.5)), 1) for t, v in W.by_type(e, room).items()},
                    "전체": round(100 * float(np.mean(e <= 0.5)), 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "seeds":
        seeds_table(sys.argv[2] if len(sys.argv) > 2 else "d120_16")
    elif w == "nf":
        nf()
    elif w == "variants":
        variants(pre=(sys.argv[2] if len(sys.argv) > 2 else ""))
    elif w == "chart":
        chart()


def variant_split(cfg="d120_16"):
    """share <= 0.5 m and mean error, outside exam rooms vs exam rooms, per content variant (seed 0)."""
    for kind, h, lab in (("two", "A", "2곳"), ("quadm", "A", "4곳"), ("quadm", "AB", "4곳 × 두 높이"), ("all8", "A", "8곳")):
        base = W.load("base", cfg)
        ea, room, _ = W.collect(base, cfg, kind, h, seeds=[0])
        ex = np.array([W.TYPE[r] == "시험실" for r in room])
        print(f"\n## {lab}")
        for v, txt in (("base", "기본"), ("nop", "사람 없음"), ("crowd", "사람 많음"), ("bare", "사람·물건 없음")):
            r = base if v == "base" else W.load(v, cfg)
            eb, rb, _ = W.collect(r, cfg, kind, h, seeds=[0])
            if len(eb) != len(ea):
                print(f"  {txt}: 없음")
                continue
            d_o = W.paired(ea[~ex], eb[~ex], room[~ex]) if v != "base" else (0, 0, 0)
            d_e = W.paired(ea[ex], eb[ex], room[ex]) if v != "base" else (0, 0, 0)
            bt = W.by_type(eb, rb)
            print(f"  {txt:8s} 밖 {eb[~ex].mean():.2f} m {100*np.mean(eb[~ex]<=0.5):.0f}% ({d_o[0]:+.2f} [{d_o[1]:+.2f}, {d_o[2]:+.2f}])"
                  f" | 시험실 {eb[ex].mean():.2f} m {100*np.mean(eb[ex]<=0.5):.0f}% ({d_e[0]:+.2f} [{d_e[1]:+.2f}, {d_e[2]:+.2f}])"
                  f" | 최저 공간 " + min(((100*np.mean(x<=0.5), t) for t, x in bt.items() if t != "시험실"))[1] +
                  f" {min(100*np.mean(x<=0.5) for t, x in bt.items() if t != '시험실'):.0f}%")


if __name__ == "__main__" and sys.argv[1] == "split":
    variant_split()
