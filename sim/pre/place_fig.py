"""패널 자리 찾기(3차원)와 사진 16장의 그림.

    python3 place_fig.py collect   ../rt/study/place_*.json -> place.json (숫자 한 파일)
    python3 place_fig.py figs      place.json -> docs/images/배치_*.png
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "images")
ST = os.path.join(ROOT, "sim", "rt", "study"); sys.path.insert(0, os.path.join(ROOT, "sim", "rt"))
KEYS = ["now", "best2_reach", "best2_noceil", "best2_any", "best3_reach", "best3_noceil", "best3_any"]
LABEL = {"now": "지금 (창 유리 가운데 + 복도 창턱 앞쪽)", "best2_reach": "두 곳, 손이 닿는 높이 (창 쪽을 앞으로)", "best2_noceil": "두 곳, 벽 2.35 m (발판)", "best2_any": "두 곳, 천장까지",
         "best3_reach": "세 곳, 손이 닿는 높이", "best3_noceil": "세 곳, 천장 빼고", "best3_any": "세 곳, 천장까지"}


def _j(name):
    p = os.path.join(ST, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def slim(m):
    keep = ("exact", "near", "top3", "min_seat", "worst5", "below40", "below60", "below80", "min_near", "per_seat", "by_kind")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in keep}


def collect():
    import place_spots as PS
    G = _j("place_g16.json"); V = _j("place_v16_eval.json"); F = _j("place_final.json")
    sp = [tuple(p) for p in PS.all_spots()]
    out = dict(made="2026-11-18", spots=[list(p) for p in sp], names=[PS.name(p, i) for i, p in enumerate(sp)], wall=[PS.where(p) for p in sp], reach=[bool(PS.reach(p)) for p in sp])
    out["seats_xy"] = np.load(os.path.join(ROOT, "sim", "rt", "paths", "kr40_tg_s1.npz"))["seats"].round(4).tolist()      # 자리 번호 차례 (per_seat 와 같다)
    if G:
        sel = G["sel"]; g = lambda c: [sel[i] for i in c]
        out["search"] = dict(cameras=1200, states=16, pairs_n=G.get("pairs_n"), triples_n=G.get("triples_n"),
                             single={str(sel[r["combo"][0]]): slim(r) for r in G["single"]})
        for key in ("pairs", "pairs_reach", "pairs_noceil", "pairs_by_mean", "triples", "triples_reach", "triples_noceil"):
            out["search"][key] = [dict(combo=g(r["combo"]), **{k: v for k, v in slim(r).items() if k != "per_seat"}) for r in G.get(key, [])[:10]]
        out["search"]["detail"] = {",".join(str(sel[int(i)]) for i in k.split(",")): {kk: (slim(vv) if isinstance(vv, dict) and "exact" in vv else vv) for kk, vv in d.items()}
                                   for k, d in G["detail"].items()}
    if V:
        out["valid"] = {k: slim(m) for k, m in V.items()}
    if F:
        out["final"] = F
        out["curve"] = {}
        for tag in ("c32", "r32", "x16", "n16", "o0", "o30", "oa", "q16", "w16", "w16r"):
            C = _j("place_%s_curve.json" % tag)
            if C:                                                        # 배치 이름마다 (같은 조합이 두 이름에 들 수 있다)
                out["curve"][tag] = {key: {k: slim(m) for k, m in C["combos"][",".join(map(str, loc))].items()}
                                     for key, loc in F["local"].items() if ",".join(map(str, loc)) in C["combos"]}
    S = _j("place_single.json")
    if S:                                                                # 패널 하나만 걸 때 (다른 방법 검토 2절)
        out["single_panel"] = S
    with open(os.path.join(HERE, "place.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("ok place.json", list(out))


# ---------------------------------------------------------------- 그림
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(f):
            fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    NAVY, TEAL, AMBER, RED, GRAY, INK, LINE = "#10304F", "#0E8F82", "#B36B00", "#E0483A", "#6B7780", "#0E1B26", "#C9D3DC"
    CM = LinearSegmentedColormap.from_list("t", ["#F3F6F8", "#BFE3DE", "#5DB8AC", TEAL, "#0A5F57"])
    P = json.load(open(os.path.join(HERE, "place.json"), encoding="utf-8")); SP = np.array(P["spots"]); W, L, H = 7.5, 9.0, 2.7
    XY = np.array(P["seats_xy"])
    F = P["final"]; C = P["curve"]; os.makedirs(OUT, exist_ok=True)

    # 1) 자리 지도: 위 줄 사진 1장, 아래 줄 사진 16장
    show = [k for k in ("now", "best2_reach", "best2_noceil", "best3_reach") if k in F["local"]]
    seen = []
    for k in list(show):
        if F[k] in seen:
            show.remove(k)
        seen.append(F[k])
    for k in ("best2_any", "best3_noceil", "best3_any"):
        if len(show) < 4 and k in F["local"] and F[k] not in [F[s] for s in show]:
            show.append(k)
    fig, axs = plt.subplots(2, len(show), figsize=(3.9 * len(show), 10.4), dpi=150, squeeze=False)

    def room(ax, vals, combo, title, sub):
        ax.add_patch(Rectangle((0, 0), W, L, fill=False, lw=1.2, ec=GRAY))
        ax.add_patch(Rectangle((1.35, -0.06), 4.8, 0.12, color=NAVY)); ax.text(1.4, 0.38, "칠판", ha="left", va="center", fontsize=8, color=NAVY)
        ax.text(0, L + 0.15, "창 쪽", ha="left", va="top", fontsize=8.5, color=GRAY); ax.text(W, L + 0.15, "복도 쪽", ha="right", va="top", fontsize=8.5, color=GRAY)
        for (x, y), v in zip(XY, vals):
            v = 100 * v; y = y + 0.125
            ax.add_patch(Rectangle((x - 0.45, y - 0.37), 0.9, 0.74, color=CM(v / 100.0), ec=RED if v < 40 else "white", lw=2.0 if v < 40 else 0.6))
            ax.text(x, y, "%d" % round(v), ha="center", va="center", fontsize=8, color="white" if v >= 62 else INK, fontweight="bold" if v < 40 else None)
        for j, i in enumerate(combo):
            x, y, z = SP[i]; wall = P["wall"][i]
            mk = "s" if wall == "천장" else "o"
            ax.plot([x], [y], marker=mk, ms=11, color=AMBER, mec="white", mew=1.2, zorder=6)
            ax.text(x, y, "ABC"[j], ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=7)
            zt = ("%.2f" % z).rstrip("0").rstrip(".") + " m"                       # 높이는 벽 밖에 적는다 (자리 칸을 가리지 않게)
            if wall == "천장":
                continue
            if x < 1.0:
                ax.text(x - 0.22, y, zt, ha="right", va="center", fontsize=7.2, color=AMBER, zorder=7)
            elif x > W - 1.0:
                ax.text(x + 0.22, y, zt, ha="left", va="center", fontsize=7.2, color=AMBER, zorder=7)
            else:
                ax.text(x + 0.25, y - 0.22, zt, ha="left", va="bottom", fontsize=7.2, color=AMBER, zorder=7)
        ax.set_xlim(-1.15, W + 1.15); ax.set_ylim(L + 0.7, -0.85); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=14); ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")

    for col, k in enumerate(show):
        for row, kk in enumerate(("1", "16")):
            m = C["c32"][k][kk]
            sub = "평균 %d%%, 옆자리까지 %d%%, 40%% 아래 %d자리" % (round(100 * m["exact"]), round(100 * m["near"]), m["below40"])
            room(axs[row, col], m["per_seat"], F[k], ("%s\n사진 %s장" % (LABEL[k], kk)), sub)
    fig.suptitle("패널 자리마다: 자리를 정확히 맞힌 비율 (%), 위는 사진 1장, 아래는 사진 16장", fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.008, "책상 위·가슴·안경 카메라 1,200대를 합친 값 (고를 때 쓰지 않은 새 카메라). 빨간 테두리 = 40% 아래. 노란 점 = 벽 패널, 네모 = 천장, 숫자 = 높이. 전파 경로 계산 (Sionna RT)",
             fontsize=8.2, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.03, wspace=0.05, hspace=0.18)
    fig.savefig(os.path.join(OUT, "배치_자리지도.png"), facecolor="white"); plt.close(fig); print("ok 배치_자리지도.png")

    # 2) 사진 수에 따른 정확도
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    cols = {"now": GRAY, "best2_reach": TEAL, "best2_noceil": NAVY, "best3_reach": AMBER, "best2_any": "#5DB8AC", "best3_any": "#D9A441", "best3_noceil": "#8A5A00"}
    for ax, (idx, nm) in zip(axs, ((0, "정확히 맞힘"), (1, "옆자리까지"))):
        for k in show:
            ks = sorted(int(x) for x in C["c32"][k]); y = [100 * (C["c32"][k][str(x)]["exact"] if idx == 0 else C["c32"][k][str(x)]["near"]) for x in ks]
            ax.plot(ks, y, "-o", color=cols.get(k, INK), lw=2, ms=4, label=LABEL[k])
        b = "best2_reach" if "best2_reach" in show else show[1]
        if "r32" in C and b in C["r32"]:
            ks = sorted(int(x) for x in C["r32"][b]); y = [100 * (C["r32"][b][str(x)]["exact"] if idx == 0 else C["r32"][b][str(x)]["near"]) for x in ks]
            ax.plot(ks, y, "--", color=TEAL, lw=1.6, label="대안: 수신 체인 둘 (같은 두 곳)")
        if "x16" in C and b in C["x16"]:
            ks = sorted(int(x) for x in C["x16"][b]); y = [100 * (C["x16"][b][str(x)]["exact"] if idx == 0 else C["x16"][b][str(x)]["near"]) for x in ks]
            ax.plot(ks, y, ":", color=RED, lw=1.8, label="나쁜 것이 모두 겹친 패널 (같은 두 곳)")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_ylim(30 if idx == 0 else 70, 100.5); ax.grid(True, color=LINE, lw=0.6); ax.set_xlabel("모은 사진 수 (전송 횟수)"); ax.set_ylabel("%")
        ax.set_title(nm, loc="left", fontsize=11.5, fontweight="bold", color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7.8, frameon=False, loc="lower right")
    fig.suptitle("사진을 몇 장 모으면 자리를 맞히나 (교실 40석, 카메라 1,200대, 사람 자세가 다른 32순간)", fontsize=12.5, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.13, wspace=0.18)
    fig.savefig(os.path.join(OUT, "배치_곡선.png"), facecolor="white"); plt.close(fig); print("ok 배치_곡선.png")

    # 3) 후보 122곳: 네 벽을 펼친 그림 + 천장 (색 = 패널 하나로 사진 16장일 때 정확히)
    single = P["search"]["single"]
    fig = plt.figure(figsize=(14.5, 6.2), dpi=150)
    ax = fig.add_axes([0.04, 0.42, 0.89, 0.46]); ax2 = fig.add_axes([0.04, 0.03, 0.3, 0.33]); ax3 = fig.add_axes([0.40, 0.05, 0.56, 0.28])
    segs = [("창 쪽 벽", 0.0, L), ("뒷벽", L, L + W), ("복도 쪽 벽", L + W, 2 * L + W), ("앞벽 (칠판)", 2 * L + W, 2 * (L + W))]

    def unfold(i):
        x, y, z = SP[i]; w = P["wall"][i]
        if w == "안":                                                       # 방 안에 올려 둔 곳은 가장 가까운 벽에
            w = ["창", "복도", "앞", "뒤"][int(np.argmin([x, W - x, y, L - y]))]
        if w == "창":
            return y
        if w == "뒤":
            return L + x
        if w == "복도":
            return L + W + (L - y)
        return 2 * L + W + (W - x)
    vals = np.array([100 * single[str(i)]["exact"] if str(i) in single else np.nan for i in range(len(SP))])
    for name, a, b in segs:
        ax.add_patch(Rectangle((a, 0), b - a, H, fill=False, ec=GRAY, lw=1)); ax.text((a + b) / 2, H + 0.08, name, ha="center", va="bottom", fontsize=9, color=INK)
    ax.add_patch(Rectangle((0.5, 0.9), 8.0, 1.5, color="#DCEAF2", zorder=0)); ax.add_patch(Rectangle((L + W + (L - 7.4), 1.8), 5.8, 0.6, color="#DCEAF2", zorder=0))
    ax.add_patch(Rectangle((2 * L + W + (W - 6.15), 0.9), 4.8, 1.2, color="#C5CED6", zorder=0)); ax.text(2 * L + W + (W - 3.75), 1.5, "칠판", ha="center", va="center", fontsize=8, color=NAVY)
    ax.text(4.5, 1.65, "창 유리", ha="center", va="center", fontsize=8, color=GRAY)
    wall_pts = [i for i in range(len(SP)) if P["wall"][i] != "천장"]
    lo, hi = np.nanpercentile(vals, 5), np.nanpercentile(vals, 99)
    s = ax.scatter([unfold(i) for i in wall_pts], [SP[i][2] for i in wall_pts], c=vals[wall_pts], cmap=CM, vmin=lo, vmax=hi, s=46, ec=INK, lw=0.4, zorder=3)
    for key, mk in (("best2_reach", "D"), ("best2_noceil", "s"), ("best3_reach", "^")):
        if key in F:
            for i in F[key]:
                if P["wall"][i] != "천장":
                    ax.scatter([unfold(i)], [SP[i][2]], marker=mk, s=150, facecolors="none", edgecolors=RED, lw=1.6, zorder=4)
    ax.axhline(2.05, color=AMBER, lw=0.8, ls="--"); ax.text(0.1, 2.08, "손이 닿는 높이 2.05 m", fontsize=7.5, color=AMBER, va="bottom")
    ax.set_xlim(-0.2, 2 * (L + W) + 0.2); ax.set_ylim(0, H + 0.35); ax.set_yticks([0, 1, 2, 2.7]); ax.set_ylabel("높이 (m)"); ax.set_xticks([])
    for sname in ("top", "right", "bottom"):
        ax.spines[sname].set_visible(False)
    ax.set_title("후보 122곳 (네 벽을 펼친 그림): 패널 하나만 걸고 사진 16장일 때 정확히 맞힌 비율", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=18)
    cb = fig.colorbar(s, ax=ax, fraction=0.015, pad=0.01); cb.set_label("정확히 (%)", fontsize=8)
    ce = [i for i in range(len(SP)) if P["wall"][i] == "천장"]
    ax2.add_patch(Rectangle((0, 0), W, L, fill=False, ec=GRAY)); ax2.scatter([SP[i][0] for i in ce], [SP[i][1] for i in ce], c=vals[ce], cmap=CM, vmin=lo, vmax=hi, s=46, ec=INK, lw=0.4)
    ax2.set_xlim(-1.0, W + 1.0); ax2.set_ylim(L + 0.3, -0.6); ax2.set_aspect("equal"); ax2.axis("off"); ax2.set_title("천장 (2.62 m), 위에서 본 것", fontsize=9.5, loc="left", color=INK)
    ax2.text(3.75, -0.35, "칠판 쪽", ha="center", fontsize=7.5, color=GRAY); ax2.text(-0.25, L / 2, "창", ha="right", va="center", fontsize=7.5, color=GRAY)
    ax2.text(W + 0.25, L / 2, "복도", ha="left", va="center", fontsize=7.5, color=GRAY)
    ax3.axis("off")
    lines = ["빨간 테두리 = 고른 자리 (◇ 두 곳·손이 닿는 높이, □ 두 곳·2.35 m, △ 세 곳·손이 닿는 높이). 천장은 고른 배치에 들지 않았다."]
    for key in ("best2_reach", "best2_noceil", "best3_reach"):
        if key in F:
            lines.append("%s: %s" % (LABEL[key], " + ".join(P["names"][i] for i in F[key])))
    ax3.text(0, 1, "\n".join(lines), va="top", fontsize=8.6, color=INK, linespacing=1.6)
    fig.savefig(os.path.join(OUT, "배치_후보.png"), facecolor="white"); plt.close(fig); print("ok 배치_후보.png")


if __name__ == "__main__":
    {"collect": collect, "figs": figs}[sys.argv[1] if len(sys.argv) > 1 else "figs"]()
