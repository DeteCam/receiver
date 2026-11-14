"""전파 계산(NEC2) 결과 그림.   python3 fig_em.py   ->  그림 폴더의 17_전파_검토.png
   가) 수신기 자기 부품을 '기준 곡선'에 얼마나 넣느냐에 따라 방향 오차가 어떻게 달라지나 (own_parts.json, mismatch.json)
   나) 판의 크기·팔 길이를 바꾸면 나아지나 (geo_eval.json: 안테나 모형 4가지의 평균)
   다) 한 자리에서 세 번 잴 때와 여섯 번 잴 때 (plan_eval.json)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def img_dir():
    """그림 폴더: 저장소에서는 docs/images, 작업용 배치에서는 out2/img (empaths 가 정한다)"""
    os.makedirs(empaths.IMG, exist_ok=True); return empaths.IMG + "/"


plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, SUB, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # 검증한 색 순서 (파랑, 주황, 청록)
J = lambda f: json.load(open(os.path.join(empaths.STUDIES, f), encoding="utf-8"))
OWN, MIS, GEO, PLAN = J("own_parts.json"), J("mismatch.json"), J("geo_eval.json"), J("plan_eval.json")
ANT = ("ez", "mz", "slot", "mono")


def style(ax, grid="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, labelsize=10.5, length=0); ax.grid(True, axis=grid, color=GRID, lw=1.0); ax.set_axisbelow(True)


def geo(name, env, key="med"):
    """판 모양 하나의 값: 안테나 모형 4가지의 평균"""
    return float(np.mean([GEO["geo_%s_%s|%s" % (a, name, env)][key] for a in ANT]))


fig = plt.figure(figsize=(15.0, 9.6), dpi=160, facecolor=SURF)
fig.text(0.03, 0.972, "전파 계산으로 본 것: 자기 부품의 영향, 판의 크기, 재는 횟수", fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.928, "안테나 프로그램 NEC2 로 수신기(기판, 단자, 모터, 알루미늄 판, 동전 균형추, 요크, 받침)를 금속 선 그물로 만들어 계산했다. 방향은 '곡선 전체를 기준 곡선과 견주는 방법'으로 찾았다.\n"
                     "숫자는 방향 오차의 가운데값(도)이다. 작을수록 좋다. 전부 계산이고, 실제 파이 안테나가 이 모형들과 얼마나 닮았는지는 아직 모른다.", fontsize=11.5, color=SUB, va="top", linespacing=1.5)

# ------------------------------------------------------------------ 가) 자기 부품과 기준 곡선
ax = fig.add_axes([0.285, 0.47, 0.30, 0.335]); style(ax)
rows = [("요크·받침까지 넣은 무늬 (받침 각마다)", "수신기 전체를 넣음\n(받침 각마다 따로)"), ("머리만 넣은 무늬 (요크·받침 없음)", "요크·받침을 뺌"), ("머리만, 균형추도 없음", "동전 균형추도 뺌"), ("기판과 단자만", "기판과 단자만 넣음")]
mis = [v["med"] for k, v in MIS.items() if k.split("|")[0] != k.split("|")[1]]
labs = [r[1] for r in rows] + ["안테나 모형을 다른 것으로\n(부품은 그대로)"]
quiet = [OWN[r[0] + "|반사 없음"]["med"] for r in rows] + [float(np.mean([v["med"] for k, v in MIS.items() if k.endswith("반사 없음") and k.split("|")[0] != k.split("|")[1]]))]
room = [OWN[r[0] + "|보통 방"]["med"] for r in rows] + [float(np.mean([v["med"] for k, v in MIS.items() if k.endswith("보통 방") and k.split("|")[0] != k.split("|")[1]]))]
y = np.arange(len(labs))[::-1]; h = 0.34
ax.barh(y + h / 2 + 0.02, quiet, height=h, color=C1, zorder=3); ax.barh(y - h / 2 - 0.02, room, height=h, color=C2, zorder=3)
for yy, a, b in zip(y, quiet, room):
    ax.text(a + 1.0, yy + h / 2 + 0.02, "%.1f°" % a, va="center", fontsize=10.5, color=INK); ax.text(b + 1.0, yy - h / 2 - 0.02, "%.1f°" % b, va="center", fontsize=10.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=10.5, color=INK, linespacing=1.25); ax.set_xlim(0, 72); ax.set_ylim(-0.7, len(labs) - 0.3)
ax.set_xticks([0, 15, 30, 45, 60]); ax.set_xticklabels(["0°", "15°", "30°", "45°", "60°"])
fig.text(0.03, 0.865, "가) 기준 곡선을 만들 때 무엇을 넣었나", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.835, "실제 장치는 전부 달려 있다. 기준 곡선에서 부품을 빼거나 안테나를 다르게 잡으면:", fontsize=10.8, color=SUB, va="top")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C1), plt.Rectangle((0, 0), 1, 1, color=C2)], labels=["반사 없는 곳", "보통 방"], loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=10.5, handlelength=1.0, borderaxespad=0.2)
fig.text(0.03, 0.435, "→ 자기 부품은 곡선을 크게 바꾸지만 늘 같은 모양으로 바꾼다. 그래서 '실제 장치로 잰 곡선'을\n    기준으로 쓰면 없앨 수 있다. 부품을 빼거나 안테나를 틀리게 잡은 계산 곡선은 쓸 수 없다.", fontsize=11, color=INK, va="top", linespacing=1.5)

# ------------------------------------------------------------------ 다) 세 번과 여섯 번
ax = fig.add_axes([0.70, 0.49, 0.27, 0.315]); style(ax, "y")
groups = ["보통 방", "반사가 심한 방"]; k3 = [PLAN["세 자세|보통"], PLAN["세 자세|나쁨"]]; k6 = [PLAN["여섯 자세|보통"], PLAN["여섯 자세|나쁨"]]
x = np.arange(2); w = 0.36
ax.bar(x - w / 2 - 0.02, [v["med"] for v in k3], width=w, color=C1, zorder=3); ax.bar(x + w / 2 + 0.02, [v["med"] for v in k6], width=w, color=C3, zorder=3)
for xx, a, b in zip(x, k3, k6):
    for dx, v in ((-w / 2 - 0.02, a), (w / 2 + 0.02, b)):
        ax.text(xx + dx, v["med"] + 0.25, "%.1f°" % v["med"], ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(["%s\n크게 틀림 %.1f%% → %.1f%%" % (g, a_["over30"], b_["over30"]) for g, a_, b_ in zip(groups, k3, k6)], fontsize=10.8, color=INK, linespacing=1.45); ax.set_ylim(0, 16.5); ax.set_yticks([0, 5, 10, 15]); ax.set_yticklabels(["0°", "5°", "10°", "15°"]); ax.set_xlim(-0.6, 1.6)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C1), plt.Rectangle((0, 0), 1, 1, color=C3)], labels=["세 번 (받침 120°씩)", "여섯 번 (60°씩)"], loc="upper left", frameon=False, fontsize=10.5, handlelength=1.0, borderaxespad=0.1)
fig.text(0.64, 0.865, "다) 한 자리에서 세 번과 여섯 번", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.64, 0.835, "수신기 전체 모형, 받침 각마다 맞는 기준 곡선을 쓸 때. 크게 틀림 = 30° 넘게", fontsize=10.8, color=SUB, va="top")
fig.text(0.64, 0.435, "→ 구조를 바꾸지 않고도, 반사가 심한 곳에서는\n    여섯 번 재면 크게 틀리는 일이 반으로 준다.", fontsize=11, color=INK, va="top", linespacing=1.5)

# ------------------------------------------------------------------ 나) 판의 크기와 팔 길이
ax = fig.add_axes([0.42, 0.055, 0.55, 0.255]); style(ax)
G = [("w100_R80", "지금: 폭 100 × 높이 150, 팔 8cm"), ("w60_R80", "폭 60"), ("w150_R80", "폭 150"), ("w200_R80", "폭 200"), ("w100h200_R80", "높이 200"), ("vee150_R80", "ㄱ자로 꺾은 판 (폭 150)"),
     ("tilt150_R80", "비스듬히 세운 판 (폭 150)"), ("w150_R66", "폭 150, 팔 6.6cm"), ("w100_R100", "폭 100, 팔 10cm"), ("w150_R100", "폭 150, 팔 10cm"), ("w200_R100", "폭 200, 팔 10cm"), ("w200h200_R100", "200 × 200, 팔 10cm"), ("w150_R120", "폭 150, 팔 12cm")]
y = np.arange(len(G))[::-1]
a = [geo(n, "보통") for n, _ in G]; b = [geo(n, "나쁨") for n, _ in G]
ax.hlines(y, a, b, color=GRID, lw=2.2, zorder=2)
ax.scatter(a, y, s=95, color=C1, zorder=4, edgecolors=SURF, linewidths=1.8); ax.scatter(b, y, s=95, color=C2, zorder=4, edgecolors=SURF, linewidths=1.8)
for yy, av, bv, (n, _) in zip(y, a, b, G):
    ax.text(av - 0.28, yy, "%.1f°" % av, va="center", ha="right", fontsize=10, color=INK)
    ax.text(bv + 0.28, yy, "%.1f°  (30° 넘게 틀림 %.0f%%)" % (bv, geo(n, "나쁨", "over30")), va="center", ha="left", fontsize=10, color=INK)
ax.axhspan(y[0] - 0.46, y[0] + 0.46, color="#eef3fb", zorder=1)
ax.set_yticks(y); ax.set_yticklabels([g[1] for g in G], fontsize=10.5, color=INK); ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(0, 19.5); ax.set_xticks([0, 5, 10, 15]); ax.set_xticklabels(["0°", "5°", "10°", "15°"]); ax.set_ylim(-0.7, len(G) - 0.3)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C1, ms=9), plt.Line2D([], [], marker="o", ls="", color=C2, ms=9)], labels=["보통 방", "반사가 심한 방"], loc="lower right", frameon=False, fontsize=10.5, borderaxespad=0.2)
fig.text(0.03, 0.368, "나) 판의 크기와 팔 길이를 바꾸면", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.338, "판 13가지 × 안테나 모형 4가지의 평균. 판만 있고 다른 부품은 없는 계산이다 (맞는 기준 곡선을 쓸 때).", fontsize=10.8, color=SUB, va="top")
fig.text(0.03, 0.300, "→ 판을 넓히거나 길게 해도 뚜렷이\n    나아지지 않는다.\n\n→ 팔을 10cm 로 늘린 것만 조금 낫다.\n    다만 머리를 숙일 수 있는 각이 70° 로\n    줄고, 논문이 확인한 치수(8cm)에서\n    벗어난다. 이번에는 바꾸지 않았다.", fontsize=11, color=INK, va="top", linespacing=1.5)
out = img_dir() + "17_전파_검토.png"; fig.savefig(out, facecolor=SURF); plt.close(fig)
print("saved", out)
print("가)", [(l.replace("\n", " "), round(q, 1), round(r, 1)) for l, q, r in zip(labs, quiet, room)])
print("나)", [(n, round(x1, 1), round(x2, 1), round(geo(n, "나쁨", "over30"), 1), round(geo(n, "보통", "over30"), 1)) for (n, _), x1, x2 in zip(G, a, b)])
