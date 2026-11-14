"""판 틀 그림 (11_판_틀.png).   xvfb-run -a python3 fig_frame.py
   가) 긴 변을 자른 면: 판이 홈에 들어가 앞면이 테두리보다 0.15mm 낮다. 바깥 둘레는 판 회전축에서 반지름 96 안
   나) 아래 변을 자른 면: 판을 세워 아래 변을 45도 턱 밑으로 밀어 넣고 → 눕힌다
   다) 위 변을 자른 면 (누름쇠 나사 자리): 누름쇠가 판 위 변을 누르고, 나사는 틀 뒷면의 너트에 들어간다
   라) 앞에서 본 판 틀 (판을 넣고 누름쇠를 단 모습)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import parts2 as P
import paths
from cadlib import *
from p2 import *

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
INK, SUB, BG = "#22262B", "#5B6470", "#F4F2EE"
C_FR, C_CL, C_AL, C_BAR, C_HW, HOT = "#22305E", "#4C6FD0", "#AEB4BC", "#7D8794", "#5E646C", "#EB6834"


def polys(m, axis, at):
    """부품 m 을 자른 면의 다각형들. axis 'z': (x, y) 평면, 'y': (x, z) 평면"""
    if axis == "z":
        cs = m.slice(float(at))
    else:
        cs = m.rotate([-90, 0, 0]).slice(float(-at))
    return [np.asarray(p, float) for p in cs.to_polygons()]


def draw(ax, m, axis, at, color, alpha=1.0, z=1):
    """구멍이 있는 단면: 바깥(반시계)과 구멍(시계)을 짝수-홀수 규칙으로 그린다"""
    ps = polys(m, axis, at)
    if not ps:
        return
    verts = []; codes = []
    for p in ps:
        verts += list(p) + [p[0]]; codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor=INK, lw=0.6, alpha=alpha, zorder=z))


def plate_pose(pl, theta):
    """판을 theta 도 세운 자세 (audit_assembly.py 와 같은 모양): 판 아래 안쪽 모서리가 홈 바닥에 닿은 채 위끝이 바깥으로 들린다"""
    th = np.radians(theta); zp = max(BL_Z0, BL_Z0 - FR_GAP + np.sin(th) + 0.05)
    return pl.translate([-FL_X1, 0, -BL_Z0]).rotate([0, theta, 0]).translate([FL_X1, 0, zp])


FR, CL, PL, BAR = P.plate_frame(), P.plate_clamp(), P.al_plate(), P.rotor_bar()
HW = P.rotor_hw(150.0, P.COINS[(150.0, AL_T)])
fig = plt.figure(figsize=(16.0, 9.6), dpi=150, facecolor=BG)
fig.text(0.03, 0.955, "판 틀: 1T 알루미늄 판이 홈에 쏙 들어간다 (판에 구멍을 뚫지 않는다)", fontsize=22, weight="bold", color=INK)
fig.text(0.03, 0.915, "판 크기만 한 홈(깊이 %.2fmm, 둘레 틈 %.1fmm)에 판을 넣으면 판 앞면이 테두리보다 %.2fmm 낮다. 아래 변은 45° 턱이, 위 변은 누름쇠가 잡는다. "
         "나사는 누름쇠의 M3 × 8 접시머리 2개뿐 (너트는 틀 뒷면)." % (FR_DEPTH, FR_GAP, FR_DEPTH - AL_T), fontsize=12.5, color=SUB)

# ---- 가) 긴 변 단면 (판 가운데 높이)
ax = fig.add_axes([0.02, 0.27, 0.23, 0.59]); ax.set_facecolor(BG)
zc = BL_Z0 + 100.0
for m, c in ((FR, C_FR), (PL, C_AL)):
    draw(ax, m, "z", zc, c)
t = np.linspace(np.radians(28), np.radians(42), 200)
ax.plot(FR_ENV * np.cos(t), FR_ENV * np.sin(t), ls="--", color=HOT, lw=1.6)
ax.set_xlim(75.0, 85.5); ax.set_ylim(40.0, 57.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("가) 긴 변을 자른 면 (판 가운데 높이)", fontsize=14, color=INK, loc="left", weight="bold")
ax.annotate("판 회전축에서 반지름 %.0f\n(이 밖으로는 아무것도 없다)" % FR_ENV, xy=(FR_ENV * np.cos(np.radians(33.0)), FR_ENV * np.sin(np.radians(33.0))), xytext=(80.6, 55.2), fontsize=10.5, color=HOT,
            arrowprops=dict(arrowstyle="->", color=HOT, lw=1.2))
ax.annotate("알루미늄 판 %gT\n앞면이 테두리보다 %.2f 낮다" % (AL_T, FR_DEPTH - AL_T), xy=(80.0, 45.0), xytext=(81.6, 41.0), fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax.annotate("판 틀 (한 장)\n바닥 %.1f + 홈 벽 %.1f" % (PAD_T, FR_WALL), xy=(78.4, 48.5), xytext=(75.1, 52.6), fontsize=11, color=C_FR, weight="bold", arrowprops=dict(arrowstyle="->", color=C_FR, lw=1.0))
ax.annotate("틈 %.1f" % FR_GAP, xy=(80.0, PLATE_W / 2 + FR_GAP / 2), xytext=(82.6, 47.6), fontsize=10, color=SUB, arrowprops=dict(arrowstyle="->", color=SUB, lw=0.9))
ax.text(75.1, 40.3, "← 판 회전축 쪽 (안)        바깥 →", fontsize=9.5, color=SUB)

# ---- 나) 아래 변 단면 (Y = 0): 세워서 턱 밑으로 밀어 넣고 → 눕힌다
TH = 12.0
for k, (theta, ttl) in enumerate(((TH, "나-1) 판을 세워 아래 변을\n턱 밑으로 밀어 넣는다"), (0.0, "나-2) 판을 눕힌다\n(위 변은 누름쇠로)"))):
    ax = fig.add_axes([0.255 + k * 0.165, 0.27, 0.16, 0.59]); ax.set_facecolor(BG)
    draw(ax, FR, "y", 0.0, C_FR)
    draw(ax, plate_pose(PL, theta), "y", 0.0, C_AL, z=2)
    ax.set_xlim(76.5, 86.5); ax.set_ylim(BL_Z0 - 3.5, BL_Z0 + 9.0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl, fontsize=12, color=INK, loc="left", weight="bold")
    if k == 0:
        ax.annotate("45° 턱 (받침 없이 뽑힌다)", xy=(82.0, BL_Z0 + 1.0), xytext=(77.2, BL_Z0 - 3.3), fontsize=10, color=C_FR, arrowprops=dict(arrowstyle="->", color=C_FR, lw=1.0))
        ax.annotate("", xy=(80.9, BL_Z0 + 2.2), xytext=(82.5, BL_Z0 + 7.6), arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.0))
        ax.text(76.6, BL_Z0 + 8.2, "%.0f° 쯤 세운다" % TH, fontsize=10, color=HOT)
    else:
        ax.annotate("판 모서리와 턱 사이 %.2f" % FR_LIP_PLAY, xy=(80.6, BL_Z0 + 0.1), xytext=(76.6, BL_Z0 + 6.5), fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
        ax.annotate("홈 바닥", xy=(79.4, BL_Z0 + 3.0), xytext=(76.6, BL_Z0 + 4.4), fontsize=10, color=C_FR, arrowprops=dict(arrowstyle="->", color=C_FR, lw=0.9))
    if k == 1:
        ax.text(76.6, BL_Z0 - 3.3, "왼쪽 = 판 회전축 쪽", fontsize=9, color=SUB)

# ---- 다) 위 변 단면 (Y = 누름쇠 나사 자리)
ys_, zs_ = P.clamp_screws()[0]
ax = fig.add_axes([0.59, 0.27, 0.19, 0.59]); ax.set_facecolor(BG)
draw(ax, FR, "y", ys_, C_FR)
draw(ax, PL, "y", ys_, C_AL, z=2)
draw(ax, CL, "y", ys_, C_CL, z=2)
draw(ax, HW, "y", ys_, C_HW, z=3)
ax.set_xlim(74.0, 89.5); ax.set_ylim(BL_Z0 + 150.0 - 7.5, BL_Z0 + 150.0 + 11.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("다) 위 변 (누름쇠 나사 자리)", fontsize=12, color=INK, loc="left", weight="bold")
ax.annotate("누름쇠 %.1fmm\n판 위 변 %.0fmm 를 덮는다" % (CL_T, CL_LAP), xy=(81.9, BL_Z0 + 150.0 - 1.5), xytext=(82.4, BL_Z0 + 150.0 - 6.4), fontsize=10, color=C_CL, arrowprops=dict(arrowstyle="->", color=C_CL, lw=1.0))
ax.annotate("M3 × 8 접시머리\n(누름쇠 → 틀 위 띠)", xy=(82.6, zs_ + 1.2), xytext=(84.0, zs_ + 3.2), fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax.annotate("M3 너트\n(틀 뒷면 육각 자리,\n%.1f 나온다)" % (2.4 - NUT_POCKET), xy=(77.2, zs_ - 2.0), xytext=(74.1, zs_ + 4.2), fontsize=9.5, color=C_HW, arrowprops=dict(arrowstyle="->", color=C_HW, lw=1.0))
ax.annotate("누름쇠 자리가\n판 앞면보다\n%.1f 낮아\n판을 누른다" % CL_SEAT, xy=(80.4, zs_ + 3.0), xytext=(74.1, BL_Z0 + 150.0 - 7.3), fontsize=9.5, color=SUB, arrowprops=dict(arrowstyle="->", color=SUB, lw=0.9))

# ---- 라) 앞에서 본 판 틀 (3D)
try:
    import pyvista as pv
    def pvm(m):
        me = m.to_mesh(); v = np.asarray(me.vert_properties)[:, :3]; f = np.asarray(me.tri_verts)
        return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    p = pv.Plotter(off_screen=True, window_size=(900, 1560)); p.set_background(BG)
    for m, c in ((FR, C_FR), (PL, C_AL), (CL, C_CL), (HW, C_HW), (BAR, C_BAR)):
        p.add_mesh(pvm(m), color=c, smooth_shading=False, specular=0.25)
    foc = np.array([62.0, 0.0, BL_Z0 + 76.0]); az, el, dist = np.radians(-38.0), np.radians(16.0), 640.0
    p.camera.position = tuple(foc + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])); p.camera.focal_point = tuple(foc); p.camera.up = (0, 0, 1); p.camera.view_angle = 26.0
    p.screenshot(TMP + "frame_tray3d.png"); p.close()
    ax = fig.add_axes([0.79, 0.05, 0.20, 0.82]); ax.imshow(plt.imread(TMP + "frame_tray3d.png")); ax.axis("off")
    ax.set_title("라) 판을 넣고 누름쇠를 단 모습", fontsize=12, color=INK, loc="left", weight="bold")
except Exception as e:
    print("3D 그림 건너뜀:", e)
_J = json.load(open(os.path.join(paths.RES, "audit_sweep.json"), encoding="utf-8"))
_B = [b for b in ("받침 S · 판 150", "받침 L · 판 150") if b in _J]
SW = {k: min(_J[b][k] for b in _B) for k in ("gap", "gap_wobble_stop")}
SW["p85"] = min(_J[b]["by_pitch"]["85.0"] for b in _B); SW["lo"] = min(_J[b]["by_pitch"]["-5.3"] for b in _B)          # 85°에서, 뒤로 젖힌 끝(-5.3°)에서
def wrap_px(text, fs, width_frac):
    """글을 그림 폭(비율)에 맞게 줄로 나눈다: 실제로 그려 본 폭으로"""
    r = fig.canvas.get_renderer(); W_ = fig.get_figwidth() * fig.dpi * width_frac; out = []
    for para in text.split("\n"):
        cur = ""
        for w_ in para.split(" "):
            t_ = (cur + " " + w_).strip(); tt = fig.text(0, 0, t_, fontsize=fs); wpx = tt.get_window_extent(renderer=r).width; tt.remove()
            if wpx > W_ and cur:
                out.append(cur); cur = w_
            else:
                cur = t_
        out.append(cur)
    return "\n".join(out)


fig.text(0.03, 0.235, wrap_px("판을 뺄 때: 누름쇠 나사 2개를 풀고, 판 틀 위쪽 창으로 판 뒷면을 밀어 위 변을 들어 올린 뒤 아래로 빼낸다. "
         "너트는 판 받이(4개)와 틀 뒷면 위쪽(2개)의 육각 자리에 끼운다. 돌지 않으니 드라이버 하나로 조인다. 누름쇠가 판에 닿아 멈출 때까지만 조인다.", 11.0, 0.74), fontsize=11.0, color=INK, va="top", linespacing=1.5)
fig.text(0.03, 0.125, wrap_px("왜 이렇게: 판을 깊이 숙여(85°) 돌리면 판 쪽이 요크·받침 덮개 바로 위를 지나간다. 그래서 틀 바깥을 반지름 %.0f 안으로 깎고, 옆 변에는 턱을 두지 않았다 (1T 판은 휘지 않는다). "
         "판을 돌리며 훑은 가장 좁은 틈: 85°에서 %.1f mm, 뒤로 젖힌 끝(−5.3°)에서 %.1f mm, 멈춤턱(86.9°)에서 %.1f mm (판 축이 1° 흔들린다고 보면 멈춤턱에서 %.1f mm)." % (FR_ENV, SW["p85"], SW["lo"], SW["gap"], SW["gap_wobble_stop"]), 11.0, 0.74), fontsize=11, color=SUB, va="top", linespacing=1.5)
fig.savefig(OUT + "11_판_틀.png", facecolor=BG); print("saved", OUT + "11_판_틀.png")
