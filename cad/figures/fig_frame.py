"""판 틀 그림 (11_판_틀.png).   xvfb-run -a python3 fig_frame.py
   가) 긴 변을 자른 면: 판 테두리를 앞뒤 틀이 감싸고, 바깥 둘레는 판 회전축에서 반지름 96 안
   나) 걸쇠를 옆에서 자른 면: 2mm 위에서 끼우고 → 아래로 밀면 갈고리가 뒤 틀 뒷면에 걸린다
   다) 뒤에서 본 걸쇠 (조립한 모습)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon as MPoly
import parts2 as P
import paths
from cadlib import *
from p2 import *

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
INK, SUB, BG = "#22262B", "#5B6470", "#F4F2EE"
C_BACK, C_FRONT, C_AL, C_BAR, HOT = "#22305E", "#4C6FD0", "#AEB4BC", "#7D8794", "#EB6834"


def polys(m, axis, at):
    """부품 m 을 자른 면의 다각형들. axis 'z': (x, y) 평면, 'y': (x, z) 평면"""
    if axis == "z":
        cs = m.slice(float(at))
    else:
        cs = m.rotate([-90, 0, 0]).slice(float(-at))
    return [np.asarray(p, float) for p in cs.to_polygons()]


def draw(ax, m, axis, at, color, alpha=1.0, z=1):
    # 구멍이 있는 단면: 바깥(반시계)과 구멍(시계)을 짝수-홀수 규칙으로 그린다
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    ps = polys(m, axis, at)
    if not ps:
        return
    verts = []; codes = []
    for p in ps:
        verts += list(p) + [p[0]]; codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor=INK, lw=0.6, alpha=alpha, zorder=z))


FB, FF, PL, BAR = P.frame_back(), P.frame_front_seated(), P.al_plate(), P.rotor_bar()
fig = plt.figure(figsize=(16.0, 9.6), dpi=150, facecolor=BG)
fig.text(0.03, 0.955, "판 틀: 알루미늄 판에 구멍을 뚫지 않고 테두리를 감싼다", fontsize=22, weight="bold", color=INK)
fig.text(0.03, 0.915, "앞뒤 두 틀이 판 둘레 3mm 를 물고, 앞 틀이 판의 옆·위·아래 변과 네 귀를 덮는다. 걸쇠 4개 + 위 나사 2개 (M3 × 8 접시머리)로 끝.", fontsize=13, color=SUB)

# ---- 가) 긴 변 단면 (판 가운데 높이)
ax = fig.add_axes([0.03, 0.08, 0.30, 0.78]); ax.set_facecolor(BG)
zc = BL_Z0 + 100.0
for m, c in ((FB, C_BACK), (PL, C_AL), (FF, C_FRONT)):
    draw(ax, m, "z", zc, c)
t = np.linspace(np.radians(28), np.radians(42), 200)
ax.plot(FR_ENV * np.cos(t), FR_ENV * np.sin(t), ls="--", color=HOT, lw=1.6)
ax.set_xlim(73.5, 86.5); ax.set_ylim(40.0, 57.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("가) 긴 변을 자른 면 (판 가운데 높이)", fontsize=15, color=INK, loc="left", weight="bold")
ax.annotate("판 회전축에서 반지름 %.0f\n(이 밖으로는 아무것도 없다)" % FR_ENV, xy=(FR_ENV * np.cos(np.radians(33.5)), FR_ENV * np.sin(np.radians(33.5))), xytext=(81.0, 55.6), fontsize=11, color=HOT,
            arrowprops=dict(arrowstyle="->", color=HOT, lw=1.2))
ax.annotate("알루미늄 판 0.5T\n(옆 변이 틀 속에 들어간다)", xy=(79.75, 46.0), xytext=(73.8, 42.0), fontsize=11, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax.annotate("앞 틀", xy=(81.2, 47.5), xytext=(84.0, 46.0), fontsize=12, color=C_FRONT, weight="bold", arrowprops=dict(arrowstyle="->", color=C_FRONT, lw=1.0))
ax.annotate("뒤 틀", xy=(78.7, 52.5), xytext=(74.0, 54.5), fontsize=12, color=C_BACK, weight="bold", arrowprops=dict(arrowstyle="->", color=C_BACK, lw=1.0))
ax.annotate("누름 턱 0.25\n(판을 뒤 틀에 누른다)", xy=(80.1, PLATE_W / 2 - 2.0), xytext=(82.6, 41.2), fontsize=10, color=SUB, arrowprops=dict(arrowstyle="->", color=SUB, lw=0.9))
ax.text(73.8, 40.4, "← 판 회전축 쪽 (안)            바깥 →", fontsize=10, color=SUB)

# ---- 나) 걸쇠 단면 (Y = 걸쇠 가운데), 끼울 때와 건 뒤
yl = (LUG_Y[0] + LUG_Y[1]) / 2; zl = P.frame_lugs()[0][1]
for k, (dz, ttl) in enumerate(((LUG_SLIDE, "나-1) 앞 틀을 %.0fmm 위에서 댄다" % LUG_SLIDE), (0.0, "나-2) 아래로 밀면 걸린다"))):
    ax = fig.add_axes([0.355 + k * 0.215, 0.27, 0.205, 0.60]); ax.set_facecolor(BG)
    draw(ax, FB, "y", yl, C_BACK)
    draw(ax, FF.translate([0, 0, dz]), "y", yl, C_FRONT, z=2)
    ax.set_xlim(72.5, 84.5); ax.set_ylim(zl - 7.0, zl + 12.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl, fontsize=12.5, color=INK, loc="left", weight="bold")
    ax.text(72.6, zl + 12.0, "왼쪽 = 판 회전축 쪽", fontsize=9.5, color=SUB)
    if k == 0:
        ax.annotate("뒤 틀 구멍", xy=(78.6, zl + 8.9), xytext=(72.6, zl + 10.2), fontsize=10.5, color=C_BACK, arrowprops=dict(arrowstyle="->", color=C_BACK, lw=1.0))
        ax.annotate("걸쇠\n(앞 틀)", xy=(77.3, zl + 4.0), xytext=(72.6, zl + 2.0), fontsize=10.5, color=C_FRONT, arrowprops=dict(arrowstyle="->", color=C_FRONT, lw=1.0))
        ax.annotate("", xy=(83.4, zl - 1.5), xytext=(83.4, zl + 3.5), arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.0))
        ax.text(81.6, zl - 3.4, "%.0fmm\n아래로" % LUG_SLIDE, fontsize=10, color=HOT)
    else:
        ax.annotate("갈고리가\n뒤 틀 뒷면에\n걸린다", xy=(77.3, zl - 0.8), xytext=(72.6, zl - 6.6), fontsize=10.5, color=HOT, arrowprops=dict(arrowstyle="->", color=HOT, lw=1.0))
fig.text(0.36, 0.205, "위 나사 2개 (M3 × 8 접시머리)가 앞 틀이 다시 올라가지 못하게 막는다.\n풀 때는 위 나사를 빼고 2mm 올려서 뗀다.", fontsize=11.5, color=INK)

# ---- 다) 뒤에서 본 걸쇠 (3D)
try:
    import pyvista as pv
    def pvm(m):
        me = m.to_mesh(); v = np.asarray(me.vert_properties)[:, :3]; f = np.asarray(me.tri_verts)
        return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    clip = P.box(60, 90, 25, 60, zl - 16, zl + 30)
    p = pv.Plotter(off_screen=True, window_size=(900, 1300)); p.set_background(BG)
    for m, c in ((FB, C_BACK), (PL, C_AL), (FF, C_FRONT)):
        p.add_mesh(pvm(m ^ clip), color=c, smooth_shading=True, specular=0.25)
    p.camera.position = (50, 82, zl - 14); p.camera.focal_point = (78.5, 51, zl + 1); p.camera.up = (0, 0, 1)
    p.screenshot(TMP + "frame_lug3d.png"); p.close()
    ax = fig.add_axes([0.79, 0.22, 0.20, 0.66]); ax.imshow(plt.imread(TMP + "frame_lug3d.png")); ax.axis("off")
    ax.set_title("다) 뒤에서 본 걸쇠", fontsize=12.5, color=INK, loc="left", weight="bold")
except Exception as e:
    print("3D 그림 건너뜀:", e)
import json
_J = json.load(open(os.path.join(paths.RES, "audit_sweep.json"), encoding="utf-8"))
SW = {k: min(_J[b][k] for b in ("받침 S · 판 150", "받침 L · 판 150") if b in _J) for k in ("gap_soft", "gap", "gap_wobble_stop")}
fig.text(0.36, 0.10, "왜 이렇게: 판을 깊이 숙여(85°) 돌리면 판 아래쪽이 요크·받침 덮개 바로 위를 지나간다. 긴 변에 나사 머리가 있으면 틈이 0.4mm 까지 줄어서\n"
         "나사를 위 띠로 옮기고 테두리를 반지름 %.0f 안으로 깎았다. 가장 좁은 틈: 85°까지 %.1f mm, 멈춤턱(86.9°)에서 %.1f mm (판 축이 1° 흔들려도 %.1f mm)." % (
             FR_ENV, SW["gap_soft"], SW["gap"], SW["gap_wobble_stop"]), fontsize=11, color=SUB)
fig.savefig(OUT + "11_판_틀.png", facecolor=BG); print("saved", OUT + "11_판_틀.png")
