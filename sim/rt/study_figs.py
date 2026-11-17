"""패널 연구(rt_study)의 결과 그림 둘을 docs/images 에 그린다. 숫자는 rt_study 의 표에서 옮긴 것이다 (돌리는 법은 README 의 '패널 연구').

    python study_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, INK = "#10304F", "#19C2B0", "#FFB020", "#E0483A", "#6B7780", "#0E1B26"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "..", "docs", "images")
KINDS = ("책상 위", "가슴", "안경")

# 1) 돌판과 패널: 손이 닿는 높이 두 곳 (마주 보는 두 옆벽). 자리를 정확히 / 옆자리까지 (%)
#    돌판 = sc_plate_a 의 '옆벽 두 가운데 2.2m' (한 곳 110초). 패널 = sc_ura4x4p37.5h_t1 의 '창 유리 가운데 + 복도 창턱 앞쪽', 전송 1번과 8번
A = {"돌판 (한 곳 110초씩)": ((55, 80), (31, 66), (50, 90)), "패널, 전송 1번 (0.3초)": ((70, 90), (56, 81), (70, 93)), "패널, 전송 8번": ((84, 97), (87, 97), (90, 100))}
fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=150); w = 0.26; x = np.arange(3)
for i, (lab, rows) in enumerate(A.items()):
    col = (GRAY, TEAL, NAVY)[i]; e = [r[0] for r in rows]; n = [r[1] for r in rows]
    ax.bar(x + (i - 1) * w, n, w * 0.92, color=col, alpha=0.28); ax.bar(x + (i - 1) * w, e, w * 0.92, color=col, label=lab)
    for xx, ee, nn in zip(x + (i - 1) * w, e, n):
        ax.text(xx, ee - 6, "%d" % ee, ha="center", fontsize=9.5, color="white", fontweight="bold"); ax.text(xx, nn + 1.2, "%d" % nn, ha="center", fontsize=8.5, color=col)
ax.set_xticks(x); ax.set_xticklabels(KINDS, fontsize=11); ax.set_ylim(0, 112); ax.set_ylabel("맞힌 비율 (%)", color=GRAY); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
ax.text(1.0, -0.17, "진한 막대 = 자리를 정확히, 옅은 막대 = 옆자리까지. 교실 40석, 두 옆벽의 2 m 높이 두 곳", transform=ax.transAxes, ha="right", fontsize=8.5, color=GRAY)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "돌판과_패널.png")); plt.close(fig)

# 2) 전송을 모을수록: 교실 40석과 강의실 72석, 패널 둘과 셋. 자리를 정확히 (%)
K = (1, 2, 4, 8)
B = {"교실 40석 · 패널 둘": ((70, 78, 80, 84), (56, 68, 80, 87), (70, 81, 88, 90)),
     "교실 40석 · 패널 셋": ((74, 83, 87, 87), (63, 76, 86, 94), (80, 90, 95, 100)),
     "강의실 72석 · 패널 둘": ((56, None, 76, 81), (45, None, 71, 80), (63, None, 81, 87)),
     "강의실 72석 · 패널 셋": ((63, None, 82, 89), (57, None, 86, 94), (76, None, 93, 99))}
fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9), dpi=150, sharey=True)
for j, ax in enumerate(axs):
    for i, (lab, rows) in enumerate(B.items()):
        v = rows[j]; kk = [k for k, y in zip(K, v) if y is not None]; yy = [y for y in v if y is not None]
        ax.plot(range(len(K)) if len(kk) == 4 else [0, 2, 3], yy, marker="o", ms=5, lw=2.0, color=(TEAL, NAVY, AMBER, RED)[i], ls="-" if i < 2 else "--", label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(["%d번" % k for k in K]); ax.set_title(KINDS[j], fontsize=11, color=INK); ax.set_ylim(40, 102); ax.grid(axis="y", color="#E3E8ED", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("카메라가 보낸 횟수 (모아서 곱한 것)", fontsize=9, color=GRAY)
axs[0].set_ylabel("자리를 정확히 맞힌 비율 (%)", color=GRAY); axs[1].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "정확도_전송횟수.png")); plt.close(fig)
print("그림 2장:", OUT)
