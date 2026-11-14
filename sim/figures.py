"""시뮬레이션 결과 그림 4장.  python3 figures.py  (먼저 validate.py, montecarlo.py 를 돌려 둔다)"""
import paths                                   # 계산 코드(core)를 찾는다
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from model import Plate
from fusion import trough, B_LO, B_HI
import montecarlo as mc
from estimator import azel, frame, to_scan, unit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out2", "img") + "/"      # 작업용 배치
if not os.path.isdir(OUT):
    OUT = os.path.join(os.path.dirname(HERE), "docs", "그림") + "/"                       # 저장소에서는 문서의 그림 폴더에
    os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, SUB, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # 검증한 색 순서 (파랑, 주황, 청록)


def style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SUB, labelsize=11, length=0)
    ax.grid(True, color=GRID, lw=1.0); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=SUB, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, color=SUB, fontsize=12)


def head(fig, title, sub, y=0.965):
    fig.text(0.045, y, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.045, y - 0.058, sub, fontsize=12.5, color=SUB, va="top")


def fig_depth():
    P = Plate()
    fig = plt.figure(figsize=(12.5, 6.4), dpi=160, facecolor=SURF)
    head(fig, "판이 카메라를 가릴 때 세기가 얼마나 꺼지나 (회절 모형)", "판 100×150, 팔 80mm, 카메라 3m. 스칼라 근사라서 모양은 실제와 다를 수 있다 (손 측정 비교 그림 참고).")
    ax = fig.add_axes([0.065, 0.12, 0.42, 0.66]); style(ax, "판이 카메라 방향에서 벗어난 각 (도)", "세기 (dB, 판이 없을 때 = 0)")
    x = np.arange(-120, 121, 1.0)
    for b, c, lab, dy in ((0, C1, "면 안 (0°)", -0.9), (30, C2, "면에서 +30°", 0.5), (-30, C3, "면에서 −30°", 0.9)):
        y = P.level(b, x); ax.plot(x, y, color=c, lw=2, label=lab, solid_capstyle="round")
        i = int(np.argmin(y)); ax.annotate("%.0f dB" % y[i], (x[i], y[i]), textcoords="offset points", xytext=(0, -14 if b >= 0 else 9), ha="center", fontsize=10.5, color=INK)
    ax.set_xlim(-120, 120); ax.set_ylim(-14, 2.5); ax.set_xticks(range(-120, 121, 30))
    ax.legend(frameon=False, fontsize=11, loc="lower right", labelcolor=SUB)
    ax.set_title("판을 돌릴 때의 곡선", loc="left", fontsize=13.5, color=INK, pad=10)
    ax2 = fig.add_axes([0.56, 0.12, 0.40, 0.66]); style(ax2, "카메라가 판이 도는 면에서 벗어난 각 (도)", "골 깊이 (dB)")
    depth = -np.min(P.level(P.betas[:, None], P.dphis[None, :]), axis=1)
    ax2.add_patch(Rectangle((B_LO, 0), B_HI - B_LO, 13, fc=C1, alpha=0.10, ec="none"))
    ax2.plot(P.betas, depth, color=C1, lw=2, solid_capstyle="round")
    ax2.set_xlim(-80, 80); ax2.set_ylim(0, 13); ax2.set_xticks(range(-80, 81, 20))
    ax2.text((B_LO + B_HI) / 2, 12.2, "판이 직선으로 가리는 범위 (%d° ~ +%d°)" % (B_LO, B_HI), ha="center", fontsize=10.5, color=SUB)
    ax2.annotate("판이 아래로 58mm뿐이라\n아래쪽이 먼저 얕아진다", (-30, np.interp(-30, P.betas, depth)), textcoords="offset points", xytext=(-8, 34), ha="center", fontsize=10.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=SUB, lw=1))
    ax2.set_title("면에서 벗어날수록 골이 얕아진다", loc="left", fontsize=13.5, color=INK, pad=10)
    fig.savefig(OUT + "21_모형_골깊이.png", facecolor=SURF); plt.close(fig)


def fig_validate():
    V = json.load(open(os.path.join(HERE, "validate.json"), encoding="utf-8"))
    from validate import model_curve
    fig = plt.figure(figsize=(12.5, 6.4), dpi=160, facecolor=SURF)
    head(fig, "모형과 손으로 잰 값 비교 (11월 11일 측정)", "판 200×200을 와이파이 모듈에서 10cm에 놓고 각도마다 3초씩. 깊이는 비슷하지만, 모형의 가운데 볼록한 모양(W)은 측정에 보이지 않는다.")
    names = {"data/scan_20261111_225427": ("측정 2 (실제 방향 100°)", 100.0), "data/scan_20261111_230202_blind": ("측정 3 (방향을 모르고 잼)", None)}
    for k, (key, (title, truth)) in enumerate(names.items()):
        v = V[key]; ax = fig.add_axes([0.065 + k * 0.49, 0.12, 0.42, 0.58]); style(ax, "판 각도 (도)", "세기 (dB, 기준 대비)" if k == 0 else None)
        ang = np.array(v["ang"]); meas = np.array(v["meas"])
        centre = truth if truth is not None else float(ang[np.argmin(meas)] + 7.5)
        grid = np.arange(0, 360, 1.0)
        m = np.interp(((grid - centre + 180) % 360) - 180, np.arange(-180, 180, 1.0), model_curve(np.arange(-180, 180, 1.0)))
        ax.plot(grid, m, color=C2, lw=2, label="모형 (가운데를 %s에 둠)" % ("실제 방향" if truth is not None else "가장 깊은 곳"), solid_capstyle="round")
        ax.plot(ang, meas, "o", color=C1, ms=9, mec=SURF, mew=2, label="손으로 잰 값", zorder=3)
        i = int(np.argmin(meas))                                               # 가장 깊은 점의 값은 곡선과 겹치지 않는 왼쪽 아래 빈 곳에 적고 가는 선으로 잇는다
        tx = max(8.0, ang[i] - 100.0)
        ax.annotate("잰 값 %.1f dB" % meas[i], (ang[i], meas[i]), xytext=(tx, -13.6), ha="left", va="center", fontsize=10.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=SUB, lw=0.9, shrinkA=2, shrinkB=6))
        ax.set_xlim(0, 360); ax.set_ylim(-16, 7); ax.set_xticks(range(0, 361, 60))
        ax.legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(-0.01, 1.0), labelcolor=SUB, ncol=2, columnspacing=1.6, handletextpad=0.5)
        ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=34)
    fig.savefig(OUT + "22_손측정과_비교.png", facecolor=SURF); plt.close(fig)


def bars(ax, cats, series, unit_txt, xmax):
    """가로 막대 묶음. series = [(이름, 값들, 색)]"""
    n = len(series); h = 0.30
    for j, (name, vals, col) in enumerate(series):
        for i, v in enumerate(vals):
            y = i + (j - (n - 1) / 2) * (h + 0.06)
            ax.add_patch(FancyBboxPatch((-2.0, y - h / 2), v + 2.0, h, boxstyle="round,pad=0,rounding_size=%.2f" % (h / 2 * 0.9), fc=col, ec="none", mutation_aspect=xmax / 100 * 0.022 * 60, clip_on=True))
            ax.text(v + xmax * 0.012, y, ("%.0f" if v >= 10 else "%.1f") % v + unit_txt, va="center", ha="left", fontsize=10.5, color=INK)
        ax.plot([], [], color=col, lw=7, solid_capstyle="round", label=name)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=11.5, color=INK)
    ax.set_xlim(0, xmax); ax.set_ylim(len(cats) - 0.45, -0.55); ax.grid(axis="y", visible=False)


def fig_error():
    R = json.load(open(os.path.join(HERE, "result.json"), encoding="utf-8")); S = R["summary"]
    bands = [b[2] for b in mc.BANDS]
    fig = plt.figure(figsize=(12.5, 7.4), dpi=160, facecolor=SURF)
    head(fig, "방향을 얼마나 잘 찾나 (시뮬레이션, 방향마다 %d번)" % R["n"], "반사가 보통인 방. 직접 재 본 값이 아니다. 편파, 기판·몸의 영향은 빠져 있어서 실제보다 좋게 나올 수 있다.", y=0.97)
    up = S["W·보통"]["세워서 1번"]; t3 = S["W·보통"]["기울여 3번 (+필요하면 1번 더)"]; t3v = S["V·보통"]["기울여 3번 (+필요하면 1번 더)"]
    ax = fig.add_axes([0.20, 0.10, 0.33, 0.60]); style(ax, "30° 넘게 틀린 비율")
    bars(ax, bands, [("세워서 1번 (논문 방식, 높이는 못 잼)", [up[b]["over30"] for b in bands], C1), ("기울여 3번 (+필요하면 1번 더)", [t3[b]["over30"] for b in bands], C2)], "%", 125)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.legend(frameon=False, fontsize=10.5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), labelcolor=SUB, ncol=1)
    ax.set_title("크게 틀리는 경우", loc="left", fontsize=13.5, color=INK, pad=56)
    ax2 = fig.add_axes([0.63, 0.10, 0.33, 0.60]); style(ax2, "방향 오차 가운데값")
    bars(ax2, bands, [("골 모양이 W (모형 그대로)", [t3[b]["med"] for b in bands], C2), ("골 모양이 V (손 측정처럼)", [t3v[b]["med"] for b in bands], C3)], "°", 12)
    ax2.set_xticks([0, 3, 6, 9, 12]); ax2.set_xticklabels(["0", "3", "6", "9", "12°"]); ax2.set_yticklabels([])
    ax2.legend(frameon=False, fontsize=10.5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), labelcolor=SUB, ncol=1)
    ax2.set_title("기울여 3번 잴 때의 오차", loc="left", fontsize=13.5, color=INK, pad=56)
    fig.savefig(OUT + "23_방향_오차.png", facecolor=SURF); plt.close(fig)


def fig_example():
    c = unit(35.0, -42.0)                                          # 책상 위 장치에서 본 바닥 쪽 카메라
    w = mc.World(np.random.default_rng(11), c, "V", "보통")
    frames = mc.STRATS["기울여 3번"]["frames"]
    from fusion import Fusion
    fus = Fusion(); curves = [w.scan(y, t) for y, t in frames]
    est, obs = fus.solve(curves, frames)
    eaz, eel = azel(est); err = float(np.degrees(np.arccos(np.clip(est @ c, -1, 1))))
    fig = plt.figure(figsize=(12.5, 5.6), dpi=160, facecolor=SURF)
    head(fig, "세 번 잰 곡선의 예: 바닥 쪽 카메라 (방위 35°, 높이 −42°)", "찾은 방향: 방위 %.0f°, 높이 %.0f° (오차 %.1f°). 골이 보인 스캔들의 가운데 각을 기하로 합쳐서 구한다." % (eaz, eel, err), y=0.96)
    for k, ((phi, lv), (yaw, tilt), o) in enumerate(zip(curves, frames, obs)):
        ax = fig.add_axes([0.065 + k * 0.315, 0.14, 0.275, 0.56]); style(ax, "판 각도 (도)", "세기 (dB)" if k == 0 else None)
        psi, beta = to_scan(c, frame(yaw, tilt))
        y = lv - np.nanmedian(lv)
        ax.plot(phi, y, color=C1, lw=2, solid_capstyle="round")
        if o["seen"]:
            xs = o["psi"] % 360
            ax.plot([xs], [np.nanmin(y) - 1.1], marker="^", ms=10, color=C2, mec=SURF, mew=2, zorder=3)
            right = xs > 200
            ax.annotate("찾은 가운데 %.0f°" % xs, (xs, np.nanmin(y) - 1.1), textcoords="offset points", xytext=(-10 if right else 10, -4), ha="right" if right else "left", fontsize=10.5, color=INK)
        ax.set_xlim(0, 360); ax.set_ylim(-12, 4); ax.set_xticks(range(0, 361, 90)); ax.set_yticks(range(-10, 5, 5))
        if k:
            ax.set_yticklabels([])
        ax.set_title("%d번째 (받침 %+d°)" % (k + 1, yaw), loc="left", fontsize=13, color=INK, pad=22)
        ax.text(0, 1.02, "카메라가 면에서 %+.0f° → %s" % (beta, "골이 보인다" if o["seen"] else "골이 안 보인다"), transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    fig.savefig(OUT + "24_스캔_예.png", facecolor=SURF); plt.close(fig)


def fig_coverage(thr=6.0):
    """방향마다 '골이 6dB 넘게 보이는 스캔이 몇 번인가' (모형 기준). 세워서 1번 / 기울여 3번."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    P = Plate()
    depth = -np.min(P.level(P.betas[:, None], P.dphis[None, :]), axis=1)
    az = np.arange(-180.0, 180.1, 2.0); el = np.arange(-88.0, 88.1, 2.0)
    A, E = np.meshgrid(az, el); C = unit(A.ravel(), E.ravel()); w = np.cos(np.radians(E))

    def count(frames):
        n = np.zeros(len(C), dtype=int)
        for y, t in frames:
            psi, beta = to_scan(C, frame(y, t))
            n += (np.interp(beta, P.betas, depth) >= thr)
        return n.reshape(A.shape)

    cols = ["#e6e5e0", "#bcd5f4", "#6ea5e7", "#1d5fb4"]
    cmap = ListedColormap(cols); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    fig = plt.figure(figsize=(12.5, 6.6), dpi=160, facecolor=SURF)
    head(fig, "어느 방향의 카메라가 몇 번의 스캔에서 보이나 (회절 모형)", "골이 %d dB 넘게 생기는 스캔의 수. 두 번 넘게 보여야 방향이 한 점으로 정해진다. 한 번만 보이면 한 번 더 잰다." % thr)
    panels = [("세워서 1번 (논문 방식)", mc.STRATS["세워서 1번"]["frames"]), ("54.7° 기울여 3번 (받침 −120°, 0°, +120°)", mc.STRATS["기울여 3번"]["frames"])]
    out = {}
    for k, (title, frames) in enumerate(panels):
        n = count(frames)
        ax = fig.add_axes([0.065 + k * 0.49, 0.13, 0.42, 0.56]); style(ax, "방위각 (도)", "높이각 (도, 아래가 바닥 쪽)" if k == 0 else None)
        ax.pcolormesh(az, el, n, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-88, 88); ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30)); ax.grid(False)
        fr = [float((w * (n == i)).sum() / w.sum() * 100) for i in range(4)]
        out[title] = fr
        txt = "안 보임 %.0f%% · 한 번 %.0f%%" % (fr[0], fr[1]) + ("" if k == 0 else " · 두 번 %.0f%% · 세 번 %.0f%%" % (fr[2], fr[3]))
        ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=28)
        ax.text(0, 1.025, "전체 방향 가운데: " + txt, transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    lx = 0.065
    for i, lab in enumerate(["안 보임", "한 번", "두 번", "세 번"]):
        fig.patches.append(Rectangle((lx, 0.028), 0.016, 0.026, transform=fig.transFigure, fc=cols[i], ec=GRID, lw=0.8))
        fig.text(lx + 0.022, 0.041, lab, fontsize=11, color=SUB, va="center"); lx += 0.085
    fig.savefig(OUT + "25_어느_방향이_보이나.png", facecolor=SURF); plt.close(fig)
    return out


if __name__ == "__main__":
    fig_depth(); fig_validate(); fig_error(); fig_example(); print(fig_coverage())
    print("그림 저장:", OUT)
