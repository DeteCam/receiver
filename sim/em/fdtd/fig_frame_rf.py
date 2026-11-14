"""판 틀이 곡선을 바꾸나 (그림 폴더의 32_판_틀_전파.png).   python3 fig_frame_rf.py [저장할 곳]
frame_study.json (gprMax) 을 읽는다: 안테나 둘레 72곳에서 받은 세기를 판 없을 때와 견준 곡선 (0도 = 판 쪽)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
INK, SUB, BG, GRID = "#22262B", "#5B6470", "#F4F2EE", "#D9D5CC"
C_PL, C_FR = "#8E949C", "#22305E"
J = json.load(open(os.path.join(HERE, "frame_study.json"), encoding="utf-8"))
a = (np.arange(72) * 5.0 + 180) % 360 - 180; o = np.argsort(a); a = a[o]
dp, df = np.array(J["curve_plate"])[o], np.array(J["curve_frame"])[o]
fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150, facecolor=BG); ax.set_facecolor(BG)
ax.plot(a, dp, color=C_PL, lw=2.4, label="판만 (알루미늄)")
ax.plot(a, df, color=C_FR, lw=2.0, label="판 + 판 틀 (PLA)")
ax.axvline(0, color=GRID, lw=1.0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=SUB); ax.grid(True, color=GRID, lw=0.6)
ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 45))
ax.set_xlabel("받는 쪽의 방향 (도, 0 = 판이 가리는 쪽)", color=INK); ax.set_ylabel("판이 없을 때보다 (dB)", color=INK)
ax.legend(frameon=False, loc="lower right", fontsize=11)
P, F = J["plate"], J["frame"]
fig.text(0.07, 0.94, "판 틀이 곡선을 바꾸나 (gprMax 4.0.1, 시간 영역 전파 계산, 2.437GHz, 격자 %.1fmm)" % J["grid_mm"], fontsize=14, weight="bold", color=INK)
fig.text(0.07, 0.895, "골 두 개의 가운데 %.1f° → %.1f°, 깊이 %.1f → %.1f dB, -3dB 폭 %.0f° → %.0f°. 두 곡선의 차이 최대 %.1f dB, 평균 %.1f dB. 틀은 실제보다 조금 크게 넣었다." % (
    round(P["centre_deg"], 1) + 0.0, round(F["centre_deg"], 1) + 0.0, P["depth_db"], F["depth_db"], P["width_3db_deg"], F["width_3db_deg"], J["max_diff_db"], J["rms_diff_db"]), fontsize=10.5, color=SUB)
fig.subplots_adjust(top=0.85, bottom=0.13, left=0.08, right=0.97)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(empaths.IMG, "32_판_틀_전파.png")          # 저장소에서는 docs/images
fig.savefig(out, facecolor=BG); print("saved", out)
