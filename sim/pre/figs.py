"""구매 전 검증의 그림.  python3 figs.py   ->  docs/images/사전검증_*.png, 숙소_방_계산.png

숫자는 result.json (collect.py 가 모은 것)과 room_est.npz (room_study.py table 이 낸 추정 자리)에서 읽는다.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "images")
sys.path.insert(0, os.path.join(ROOT, "sim", "rt"))
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, INK, BG, LINE = "#10304F", "#0E8F82", "#B36B00", "#E0483A", "#6B7780", "#0E1B26", "#F5F7FA", "#C9D3DC"
KIND = [("책상 위", NAVY), ("가슴", AMBER), ("안경", TEAL)]
R = json.load(open(os.path.join(HERE, "result.json"), encoding="utf-8")); PRE = R["pre"]
TWO = "창 유리 가운데 2.0m + 복도 창턱 앞쪽 1.95m"
BASE = {1: [[70, 90, 85], [56, 81, 79], [70, 93, 88]], 4: [[80, 95, 95], [80, 94, 93], [88, 99, 98]]}       # 지금 설계 (패널 연구의 t1: 패널 둘, 전송 1번과 4번)


def two(tag, k=1):
    return PRE[tag]["%s | %d번" % (TWO, k)]


def style(ax):
    ax.set_facecolor("white"); ax.grid(axis="y", color="#E6EBF0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
    ax.tick_params(colors=GRAY, labelsize=9)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor=fig.get_facecolor(), dpi=150); plt.close(fig); print("ok", name)


def tolerance():
    """실물에서 생길 흔들림을 넣었을 때. 자리를 정확히 맞힌 비율 (전송 한 번, 패널 둘)"""
    rows = [("흔들림\n없음", BASE[1], BASE[4]), ("이득\n1 dB", two("i_g1 ura4x4p37.5h"), two("i_g1 ura4x4p37.5h", 4)), ("이득\n2 dB", two("i_g2 ura4x4p37.5h"), two("i_g2 ura4x4p37.5h", 4)), ("이득\n3 dB", two("i_g3 ura4x4p37.5h"), two("i_g3 ura4x4p37.5h", 4)),
            ("위상\n단계\n10도", two("i_p10 ura4x4p37.5h"), two("i_p10 ura4x4p37.5h", 4)), ("위상\n단계\n20도", two("i_p20 ura4x4p37.5h"), two("i_p20 ura4x4p37.5h", 4)), ("위상\n단계\n40도", two("i_p40 ura4x4p37.5h"), two("i_p40 ura4x4p37.5h", 4)),
            ("세기 다른\n패킷\n30%", two("i_tx3 ura4x4p37.5h"), two("i_tx3 ura4x4p37.5h", 4)), ("정수\n4비트", two("i_q4 ura4x4p37.5h"), two("i_q4 ura4x4p37.5h", 4)), ("잡음\n10 dB", two("i_n10 ura4x4p37.5h"), two("i_n10 ura4x4p37.5h", 4)),
            ("모두\n겹침", two("i_bad ura4x4p37.5h"), two("i_bad ura4x4p37.5h", 4)), ("겹침 +\n세 번\n재기", two("i_badr ura4x4p37.5h"), two("i_badr ura4x4p37.5h", 4))]
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=150, sharey=True); fig.patch.set_facecolor(BG)
    for ax, col, ttl in ((axs[0], 1, "전송 한 번"), (axs[1], 2, "전송 네 번을 쌓으면")):
        style(ax); x = np.arange(len(rows)); w = 0.26
        for j, (nm, c) in enumerate(KIND):
            v = [r[col][j][1] for r in rows]; e = [r[col][j][0] for r in rows]
            ax.bar(x + (j - 1) * w, v, w * 0.92, color=c, alpha=0.28, label="%s: 옆자리까지" % nm if col == 1 else None); ax.bar(x + (j - 1) * w, e, w * 0.92, color=c, label="%s: 정확히" % nm if col == 1 else None)
        ax.axvspan(-0.5, 0.5, color="#F0F4F8", zorder=0); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8.2, color=INK); ax.set_ylim(0, 105); ax.set_title(ttl, fontsize=12, fontweight="bold", color=INK, loc="left")
    axs[0].set_ylabel("자리를 맞힌 비율 (%)", color=GRAY, fontsize=10); axs[0].legend(ncol=3, fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.46))
    fig.suptitle("실물에서 흔들려도 버티나: 세기 재기에 흔들림을 넣은 계산", fontsize=15, fontweight="bold", color=INK, x=0.03, ha="left", y=0.98)
    fig.text(0.985, 0.03, "교실 40석, 패널 둘 (창 유리 가운데 + 복도 창턱 앞쪽). 진한 막대 = 자리를 정확히, 옅은 막대 = 옆자리까지. 나쁜 것을 모두 겹침 = 이득 2dB + 위상 단계 15도 + 세기가 다른 패킷 20% + 잡음 20dB",
             fontsize=8.5, color=GRAY, ha="right")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.3, wspace=0.05); save(fig, "사전검증_견딤.png")


def configs():
    """구성마다의 값과 정확도 (옆자리까지, 책상·가슴·안경의 평균)"""
    c2 = PRE["c1 ura2x2h"]; c3 = "칠판 윗변 가운데 + 창 유리 가운데 2.0m + 복도 창턱 앞쪽"
    avg = lambda v: sum(k[1] for k in v) / 3.0
    P16, P4 = 22.4, 13.5                                                                    # 패널 하나의 전자 부품 값 (만 원): 16안테나(보드 12 + ESP32 1.5 + 스위치 2개 5 + 합침기 1.5 + 선 2.4), 4안테나(안테나 4개 1.2 + 스위치 3개 7.5 + ...)
    pts = [("안테나 16개 패널 둘\n(지금 설계)", 2 * P16, avg(BASE[1]), avg(BASE[4]), NAVY, "r"), ("안테나 16개 패널 하나\n(먼저 만드는 것)", P16, avg(PRE["single"]["창 유리 가운데 2.0m | 1번"]), avg(PRE["single"]["창 유리 가운데 2.0m | 4번"]), TEAL, "b"),
           ("안테나 4개 패널 둘", 2 * P4, avg(two("c1 ura2x2h")), avg(two("c1 ura2x2h", 4)), AMBER, "r"), ("안테나 4개 패널 셋", 3 * P4, avg(c2[c3 + " | 1번"]), avg(c2[c3 + " | 4번"]), AMBER, "b"),
           ("세기만 재는 수신기 12개\n(거리만으로)", 7.0, avg(PRE["rss"]["(가) 거리만으로 셈한 세기, 전송 1번 평균 | 12개"]), avg(PRE["rss"]["(가) 거리만으로 셈한 세기, 전송 4번 평균 | 12개"]), GRAY, "b"),
           ("세기만 재는 수신기 20개\n(거리만으로)", 11.0, avg(PRE["rss"]["(가) 거리만으로 셈한 세기, 전송 1번 평균 | 20개 (둘레 전부)"]), avg(PRE["rss"]["(가) 거리만으로 셈한 세기, 전송 4번 평균 | 20개 (둘레 전부)"]), GRAY, "r")]
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=150); fig.patch.set_facecolor(BG); style(ax); ax.grid(axis="x", color="#E6EBF0", lw=0.8)
    for nm, cost, a1, a4, c, where in pts:
        ax.plot([cost, cost], [a1, a4], color=c, lw=2, alpha=0.5); ax.scatter([cost], [a1], s=90, color=c, zorder=3); ax.scatter([cost], [a4], s=90, facecolor="white", edgecolor=c, lw=2, zorder=3)
        if where == "r": ax.text(cost + 0.9, (a1 + a4) / 2, nm, fontsize=9.5, color=INK, va="center", linespacing=1.3)
        else: ax.text(cost, a1 - 3.2, nm, fontsize=9.5, color=INK, va="top", ha="center", linespacing=1.3)
        ax.text(cost - 0.7, a1, "%.0f" % a1, fontsize=9, color=c, ha="right", va="center", fontweight="bold"); ax.text(cost - 0.7, a4, "%.0f" % a4, fontsize=9, color=c, ha="right", va="center")
    ax.scatter([], [], s=90, color=INK, label="전송 한 번"); ax.scatter([], [], s=90, facecolor="white", edgecolor=INK, lw=2, label="전송 네 번을 쌓아"); ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.set_xlim(0, 62); ax.set_ylim(25, 102); ax.set_xlabel("방향을 재는 부품의 값 (만 원, 어림)", color=GRAY, fontsize=10); ax.set_ylabel("옆자리까지 맞힌 비율 (%), 책상 위·가슴·안경의 평균", color=GRAY, fontsize=10)
    fig.suptitle("더 싼 구성은 얼마나 맞히나", fontsize=15, fontweight="bold", color=INK, x=0.03, ha="left", y=0.97)
    fig.text(0.985, 0.015, "교실 40석 계산. 값은 패널 안의 전자 부품만 (허브, 출력물, 나사는 뺌). 세기만 재는 수신기는 ESP32-C3 급을 벽 둘레에 붙인 것", fontsize=8.5, color=GRAY, ha="right")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.14); save(fig, "사전검증_구성과_값.png")


_G = json.load(open(os.path.join(ROOT, "cad", "results", "gates.json"), encoding="utf-8"))["gates"]          # 관문마다의 값: cad/export/partlist.py 가 낸다
GATES = [("관문 0", "지금 (0원)", 0, ["공개된 시험 보고", "가진 기록의 세기 분석", "흔들림을 넣은 계산", "문헌"], "끝냄: 걸리는 것 없음"),
         ("관문 1", "ESP32-C5 2개", _G[0]["cost"] / 1e4, ["ACK 의 CSI 가 나오나", "세기가 고른가", "5GHz 도 엿듣나"], "안 되면: 카메라가 보낼 때만 잰다"),
         ("관문 2", "합침기 · 선 · 변환 단자", _G[1]["cost"] / 1e4, ["안테나 둘로 위상이", "되찾아지나 (선을 손으로 바꿔)"], "안 되면: 여기서 멈춘다 (돌판)"),
         ("관문 3", "RF 스위치 2개", _G[2]["cost"] / 1e4, ["자동으로 네 단계", "패킷 64개에 0.5초 안"], "안 되면: 느리게 잰다"),
         ("관문 4", "안테나 보드 1장 · 첫 패널 재료", _G[3]["cost"] / 1e4, ["패널 하나로 방향 (3~6도)", "교실에서 자리 맞히기"], "안 되면: 보정표를 만든다"),
         ("관문 5", "둘째 패널 · 동글 · 나머지", _G[4]["cost"] / 1e4, ["패널 둘로 자리 맞히기", "데모"], "")]


def gates():
    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=150); fig.patch.set_facecolor(BG); ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.4); ax.axis("off"); ax.set_position([0, 0, 1, 1])
    ax.text(0.35, 5.0, "돈을 나눠 쓰는 순서: 싼 것부터 사서 확인하고, 통과하면 다음 것을 산다", fontsize=15, fontweight="bold", color=INK, va="center")
    tot = sum(g[2] for g in GATES); cum = 0.0; w = 2.02
    for i, (nm, buy, cost, chk, fb) in enumerate(GATES):
        x = 0.35 + i * (w + 0.1); cum += cost; done = i == 0; col = TEAL if done else NAVY
        ax.add_patch(FancyBboxPatch((x, 1.25), w, 3.2, boxstyle="round,pad=0,rounding_size=0.1", fc="#E3F5F2" if done else "white", ec=col, lw=1.4))
        ax.text(x + 0.14, 4.18, nm, fontsize=12, fontweight="bold", color=col, va="center"); ax.text(x + w - 0.14, 4.18, "+%.1f만" % cost if cost else "0원", fontsize=11, fontweight="bold", color=RED if cost >= 10 else col, va="center", ha="right")
        ax.text(x + 0.14, 3.83, buy, fontsize=8.8, color=GRAY, va="center")
        ax.plot([x + 0.14, x + w - 0.14], [3.6, 3.6], color=LINE, lw=1)
        for j, c in enumerate(chk): ax.text(x + 0.14, 3.33 - 0.27 * j, ("· " if not c.startswith("되찾") else "  ") + c, fontsize=8.8, color=INK, va="center")
        if fb: ax.text(x + 0.14, 1.52, fb, fontsize=8.2, color=AMBER if not done else TEAL, va="center")
        ax.add_patch(Rectangle((x, 0.62), w, 0.3, fc="#E6EBF0", ec="none")); ax.add_patch(Rectangle((x, 0.62), w * cum / tot, 0.3, fc=col, ec="none", alpha=0.85))
        ax.text(x + w / 2, 0.36, "여기까지 %.1f만 원" % cum if cum else "여기까지 0원", fontsize=8.8, color=GRAY, ha="center", va="center")
    save(fig, "사전검증_관문.png")


def room():
    import scene_room as S
    Z = np.load(os.path.join(HERE, "room_est.npz")); cams, cat = Z["cams"], Z["cat"]; RM = R["room"]["기본"]; cats = list(RM["by_cat"])
    names = [k.split(" | ")[0] for k in RM["combos"]]; uniq = list(dict.fromkeys(names)); iA = uniq.index("가운데에 서서 다섯 쪽으로 돌림"); iB = uniq.index("네 자리 모두 (가운데·창턱·현관 통로·욕실 안)")
    fig = plt.figure(figsize=(13.4, 6.6), dpi=150); fig.patch.set_facecolor(BG)
    fig.suptitle("숙소 방에서 숨긴 카메라의 자리 찾기 (계산)", fontsize=15, fontweight="bold", color=INK, x=0.03, ha="left", y=0.975)
    for n, (i, ttl) in enumerate(((iA, "방 가운데에서 다섯 쪽으로 돌려 재면"), (iB, "창턱·현관 통로·욕실 안에서도 재면"))):
        ax = fig.add_axes([0.03 + 0.215 * n, 0.07, 0.2, 0.8]); e = Z["e%d" % i]; p = Z["p%d" % i]
        ax.add_patch(Rectangle((0, 0), S.W, S.D, fc="white", ec=NAVY, lw=1.6)); ax.add_patch(Rectangle((0, 0), S.BX, S.BY, fc="#EAF1F6", ec=NAVY, lw=1.2))
        for (x0, y0, x1, y1) in ((0.15, 3.0, 2.15, 4.6), (3.05, 2.6, 3.6, 4.2), (1.80, 0.05, 2.30, 0.60)):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#F0EBE1", ec="#B9AE98", lw=0.8))
        ok = e <= 0.5
        for k in np.where(~ok)[0]:
            ax.plot([cams[k, 0], p[k, 0]], [cams[k, 1], p[k, 1]], color=RED, lw=0.4, alpha=0.18)
        ax.scatter(cams[ok, 0], cams[ok, 1], s=9, color=TEAL, alpha=0.7, lw=0); ax.scatter(cams[~ok, 0], cams[~ok, 1], s=11, color=RED, alpha=0.9, lw=0)
        ax.set_xlim(-0.15, S.W + 0.15); ax.set_ylim(-0.15, S.D + 0.15); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("%s\n50cm 안 %.0f%%, 가운데값 %.2fm" % (ttl, 100 * ok.mean(), np.median(e)), fontsize=9.5, color=INK, loc="left", linespacing=1.4)
    fig.text(0.03, 0.03, "점 = 숨긴 카메라 864번 (위에서 본 그림). 청록: 50cm 안으로 맞힘. 빨강: 못 맞힘 (선은 잘못 짚은 곳까지)", fontsize=8.5, color=GRAY)
    ax = fig.add_axes([0.52, 0.56, 0.455, 0.3]); style(ax)
    show = [("한 쪽만\n(가운데에서 창 쪽)", "가운데에서 한 쪽만 (창 쪽) | 방"), ("가운데\n다섯 쪽", "가운데에 서서 다섯 쪽으로 돌림 | 방"), ("+ 창턱", "다섯 쪽 + 창턱 | 방"), ("+ 창턱\n+ 현관 통로", "다섯 쪽 + 창턱 + 현관 통로 | 방"), ("고정 둘\n(책상 위 + 창턱)", "책상 위 + 창턱 (고정 둘) | 방"),
            ("욕실 안\n한 번", "욕실 안에서 한 번 | 욕실"), ("욕실 문 앞\n+ 안", "욕실 문 앞 + 욕실 안 | 욕실")]
    x = np.arange(len(show)); v50 = [100 * RM["combos"][k]["in50"] for _, k in show]; v100 = [100 * RM["combos"][k]["in100"] for _, k in show]; col = [NAVY] * 5 + [TEAL] * 2
    ax.bar(x, v100, 0.62, color=col, alpha=0.28); ax.bar(x, v50, 0.62, color=col)
    for i, (a, b) in enumerate(zip(v50, v100)): ax.text(i, a - 6, "%.0f" % a, ha="center", fontsize=9, color="white", fontweight="bold"); ax.text(i, b + 2, "%.0f" % b, ha="center", fontsize=8.5, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in show], fontsize=8.2, color=INK); ax.set_ylim(0, 112); ax.set_title("어디서 재느냐에 따라: 50cm 안 (진한 막대) · 1m 안 (옅은 막대) %", fontsize=10.5, fontweight="bold", color=INK, loc="left")
    ax = fig.add_axes([0.52, 0.1, 0.455, 0.3]); style(ax); A, B = "가운데에 서서 다섯 쪽으로 돌림", "네 자리 모두 (가운데·창턱·현관 통로·욕실 안)"
    x = np.arange(len(cats)); a = [100 * RM["by_cat"][c][A]["in50"] for c in cats]; b = [100 * RM["by_cat"][c][B]["in50"] for c in cats]
    ax.bar(x - 0.2, a, 0.38, color=GRAY, label="가운데 다섯 쪽만"); ax.bar(x + 0.2, b, 0.38, color=NAVY, label="네 자리 모두")
    short = {"침대 머리맡 (콘센트·충전기·스탠드)": "침대\n머리맡", "현관 쪽 (옷장 위·문 위)": "현관 쪽", "천장 화재감지기": "천장\n감지기", "벽시계·액자": "벽시계\n·액자", "TV·셋톱박스 둘레": "TV\n둘레", "에어컨·커튼 상자": "에어컨\n·커튼", "창가 화분·휴지통": "창가 화분\n·휴지통", "욕실 환풍기": "욕실\n환풍기", "욕실 선반·거울 둘레": "욕실 선반\n·거울"}
    ax.set_xticks(x); ax.set_xticklabels([short.get(c, c) for c in cats], fontsize=8.2, color=INK); ax.set_ylim(0, 112); ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.2))
    ax.set_title("숨긴 곳마다: 50cm 안 %", fontsize=10.5, fontweight="bold", color=INK, loc="left")
    save(fig, "숙소_방_계산.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (tolerance, configs, gates, room):
        f()
