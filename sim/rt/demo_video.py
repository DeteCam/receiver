"""demo_run.json (교실에서 한 번 재는 과정)을 동작 시뮬레이션 영상으로 만든다.

    python demo_video.py [--out ../../docs/videos/교실_동작_시뮬레이션.mp4]

위에서 본 교실(치수 표시)에 수신기가 벽을 따라 옮겨 가고, 곳마다 판을 돌려 잰 신호 곡선과
자리 확률이 좁혀지는 모습을 보여 준다. 숫자는 모두 demo_run.json 의 계산값이다.
"""
import os, json, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, BG, INK = "#10304F", "#19C2B0", "#FFB020", "#E0483A", "#6B7780", "#F5F7FA", "#0E1B26"
HERE = os.path.dirname(os.path.abspath(__file__)); W, H, FPS = 1280, 720, 24
T_IN, T_MOVE, T_SCAN, T_UP, T_OUT = 3.0, 1.0, 3.4, 1.0, 5.0
ORD = "①②③④⑤⑥⑦⑧"


def heat(p):
    t = min(1.0, float(p) ** 0.5); a = np.array([1, 1, 1.0]); b = np.array([1.0, 0.69, 0.125]); c = np.array([0.878, 0.282, 0.227])
    return tuple(a + (b - a) * (t / 0.5)) if t < 0.5 else tuple(b + (c - b) * ((t - 0.5) / 0.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs", "videos", "교실_동작_시뮬레이션.mp4")); ap.add_argument("--still", type=float)
    a = ap.parse_args(); D = json.load(open(os.path.join(HERE, "demo_run.json"), encoding="utf-8"))
    seats = np.array(D["seats"]); steps = D["steps"]; n = len(steps); RW, RD = 7.5, 9.0; empty = set(D["empty"]); cam = D["cam"]; seat = D["seat"]
    xs = np.unique(np.round(seats[:, 0], 3)); ys = np.unique(np.round(seats[:, 1], 3))
    lab = lambda k: "%d째 줄 %d번" % (int(np.argmin(np.abs(ys - seats[k, 1]))) + 1, int(np.argmin(np.abs(xs - seats[k, 0]))) + 1)
    per = T_MOVE + T_SCAN + T_UP; total = T_IN + n * per + T_OUT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)

    def draw(t):
        fig.clf(); fig.patch.set_facecolor(BG)
        k = int(np.clip((t - T_IN) // per, -1, n - 1)) if t >= T_IN else -1; tt = t - T_IN - k * per if k >= 0 else 0.0
        done = t >= T_IN + n * per
        phase = "in" if k < 0 else ("out" if done else ("move" if tt < T_MOVE else ("scan" if tt < T_MOVE + T_SCAN else "up")))
        shown = (n - 1) if done else (k if phase == "up" else k - 1)                    # 지금 화면에 반영된 곳 (마지막 인덱스)
        post = np.array(steps[shown]["post"]) if shown >= 0 else np.full(len(seats), 1.0 / len(seats))
        if phase == "up" and k >= 0:                                                   # 앞 곳의 확률에서 새 확률로 부드럽게
            prev = np.array(steps[k - 1]["post"]) if k > 0 else np.full(len(seats), 1.0 / len(seats)); u = (tt - T_MOVE - T_SCAN) / T_UP; post = prev + (np.array(steps[k]["post"]) - prev) * u
        ax = fig.add_axes([0.035, 0.05, 0.47, 0.86]); ax.set_xlim(-1.05, RW + 1.25); ax.set_ylim(RD + 0.75, -0.95); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), RW, RD, fc="white", ec=NAVY, lw=2.2)); ax.add_patch(Rectangle((1.35, 0.03), 4.8, 0.1, fc="#2E5E4E", ec="none")); ax.text(RW / 2, -0.18, "칠판", ha="center", fontsize=9, color=GRAY)
        ax.add_patch(Rectangle((RW / 2 - 0.4, 0.9), 0.8, 0.5, fc="#D8C3A5", ec="none")); ax.text(RW / 2, 1.15, "교탁", ha="center", va="center", fontsize=8, color="#6B5B45")
        for i, (x, y) in enumerate(seats):
            hot = post[i] > 0.02 and shown >= 0
            ax.add_patch(Rectangle((x - 0.3, y - 0.2), 0.6, 0.4, fc=heat(post[i]) if shown >= 0 or phase == "up" else "white", ec="#B36B00" if hot else "#9AA7B2", lw=1.3 if hot else 0.7, zorder=3))
            if i not in empty:
                ax.add_patch(Circle((x, y + 0.45), 0.16, fc="#CBD3DA", ec="none", zorder=2))
        ax.add_patch(Circle((cam[0], cam[1]), 0.13, fc="none", ec=RED, lw=2.0, zorder=6)); ax.annotate("카메라 (실제 위치)", (cam[0], cam[1]), (cam[0] + 0.55, cam[1] - 0.62), fontsize=8.5, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="-", color=RED, lw=0.8), zorder=7)
        for p in D["spots"]:
            ax.add_patch(Circle((p[0], p[1]), 0.09, fc="none", ec="#B8C2CC", lw=1.0, zorder=4))
        vis = (k + 1) if k >= 0 else 0
        for j in range(min(vis, n) if not done else n):
            if j == k and phase == "move" and not done:
                continue
            x, y = steps[j]["xy"]; ax.add_patch(Circle((x, y), 0.25, fc=NAVY, ec="white", lw=1.3, zorder=8)); ax.text(x, y, str(j + 1), ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=9)
        if k >= 0 and not done:                                                         # 지금 재는 수신기
            x1, y1 = steps[k]["xy"]; x0, y0 = steps[k - 1]["xy"] if k > 0 else (RW / 2, -0.6)
            u = min(1.0, tt / T_MOVE); u = u * u * (3 - 2 * u); x, y = (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) if phase == "move" else (x1, y1)
            ax.add_patch(Circle((x, y), 0.36, fc="none", ec=TEAL, lw=2.4, zorder=9))
            if phase == "scan":
                ang = 2 * np.pi * (tt - T_MOVE) / T_SCAN * 3; px, py = x + 0.36 * np.cos(ang), y + 0.36 * np.sin(ang); dx, dy = -np.sin(ang) * 0.2, np.cos(ang) * 0.2
                ax.plot([px - dx, px + dx], [py - dy, py + dy], color=INK, lw=3.2, solid_capstyle="round", zorder=10)
                ax.plot([x, cam[0]], [y, cam[1]], color=TEAL, lw=1.0, ls=(0, (4, 4)), alpha=0.7, zorder=5)
            if phase == "move":
                ax.add_patch(Circle((x, y), 0.22, fc=TEAL, ec="white", lw=1.2, zorder=10))
        # 치수
        ax.annotate("", (0, -0.55), (RW, -0.55), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0)); ax.text(RW / 2 + 2.9, -0.62, "7.5 m", ha="center", va="bottom", fontsize=9.5, color=GRAY)
        ax.annotate("", (RW + 0.5, 0), (RW + 0.5, RD), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0)); ax.text(RW + 0.62, RD / 2, "9.0 m", rotation=-90, ha="left", va="center", fontsize=9.5, color=GRAY)
        ax.annotate("", (xs[0], ys[-1] + 0.95), (xs[1], ys[-1] + 0.95), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8)); ax.text((xs[0] + xs[1]) / 2, ys[-1] + 1.02, "1.1 m", ha="center", va="top", fontsize=8, color=GRAY)
        ax.annotate("", (-0.5, ys[0]), (-0.5, ys[1]), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8)); ax.text(-0.58, (ys[0] + ys[1]) / 2, "0.95 m", ha="right", va="center", fontsize=8, color=GRAY)
        ax.text(0, RD + 0.38, "교실 40석 · 천장 2.7 m · 수신기 안테나 높이 2.4 m (스탠드) · 벽에서 0.25~0.5 m", fontsize=8.8, color=GRAY, va="center")
        # 오른쪽
        fig.text(0.55, 0.905, "교실 동작 시뮬레이션", fontsize=21, color=INK, fontweight="bold"); fig.text(0.55, 0.862, "시험 중 Wi-Fi 카메라의 자리를 좁혀 가는 과정 (전파 경로 계산)", fontsize=10.5, color=GRAY)
        if phase == "in":
            msg, sub = "수신기를 2.4 m 스탠드에 올린다", "교탁 옆에서 시작한다. 학생 사이로는 들어가지 않는다"
        elif done:
            ok = steps[-1]["top3"][0] == seat; msg = "후보 3자리를 눈으로 확인한다"; sub = "1순위 %s%s · 여섯 곳, 약 %d분" % (lab(steps[-1]["top3"][0]), " = 실제 카메라 자리" if ok else "", round(n * 2.3 + (n - 1) * 0.8))
        elif phase == "move":
            msg, sub = "%s 곳째로 옮긴다" % ORD[k], ("교탁 옆" if k == 0 else "앱이 고른 곳: 지금 후보들이 가장 잘 갈라져 보이는 벽 쪽 자리")
        elif phase == "scan":
            msg, sub = "%s 판을 돌려 잰다" % ORD[k], "머리 숙임 3가지 × 받침 6방향 = %d번 (약 2분 20초)" % D["scans"]
        else:
            msg, sub = "%s 결과를 겹친다" % ORD[k], "곳마다의 가능성을 곱해 자리 확률을 고친다"
        fig.text(0.55, 0.785, msg, fontsize=17, color=NAVY, fontweight="bold"); fig.text(0.55, 0.745, sub, fontsize=10.5, color=INK)
        el = 0 if k < 0 else (n * 2.3 + (n - 1) * 0.8 if done else k * 3.1 + (2.3 * min(1.0, max(0.0, (tt - T_MOVE) / T_SCAN)) if phase != "move" else 0))
        fig.text(0.955, 0.79, "지난 시간 약 %d분" % round(el), fontsize=10.5, color=GRAY, ha="right")
        ac = fig.add_axes([0.575, 0.455, 0.38, 0.235]); ac.set_xlim(0, 360); ac.set_xticks([0, 90, 180, 270, 360]); ac.set_xticklabels(["0°", "90°", "180°", "270°", "360°"], fontsize=8.5, color=GRAY); ac.set_yticks([])
        for s_ in ("top", "right", "left"):
            ac.spines[s_].set_visible(False)
        ac.spines["bottom"].set_color("#B8C2CC"); ac.set_title("신호 곡선 (판 각도마다의 세기) · 18번 가운데 3번", fontsize=9.5, color=GRAY, loc="left", pad=4)
        kc = k if (k >= 0 and not done and phase != "move") else (shown if shown >= 0 else None)
        if kc is not None:
            frac = min(1.0, (tt - T_MOVE) / T_SCAN * 1.15) if (kc == k and phase == "scan") else 1.0
            for ci, row in enumerate(steps[kc]["curves"]):
                v = np.array([np.nan if x is None else x for x in row], float); v = v - np.nanmean(v) + ci * 0.0; m = max(2, int(len(v) * frac)); xx = 2.5 + 5 * np.arange(len(v))
                ac.plot(xx[:m], v[:m], color=[NAVY, TEAL, AMBER][ci % 3], lw=2.0, alpha=0.95)
            ac.set_ylim(-14, 9)
        ab = fig.add_axes([0.575, 0.085, 0.38, 0.27]); ab.set_xlim(0, 1.0); ab.set_ylim(-0.6, 2.6); ab.invert_yaxis(); ab.axis("off"); ab.set_title("자리 후보 3개와 확률", fontsize=9.5, color=GRAY, loc="left", pad=4)
        if shown >= 0 or phase == "up":
            o = np.argsort(-post)[:3]
            for r_, i in enumerate(o):
                ab.add_patch(Rectangle((0.34, r_ - 0.3), 0.62, 0.6, fc="#E7ECF1", ec="none")); ab.add_patch(Rectangle((0.34, r_ - 0.3), 0.62 * float(post[i]), 0.6, fc=RED if i == seat else AMBER, ec="none"))
                ab.text(0.0, r_, lab(i), va="center", fontsize=11.5, color=INK, fontweight="bold" if r_ == 0 else "normal"); ab.text(0.97, r_, "%d%%" % round(float(post[i]) * 100), va="center", ha="right", fontsize=11, color=INK)
        else:
            ab.text(0.0, 1.0, "아직 재지 않았다 (40자리 모두 같은 확률)", va="center", fontsize=10.5, color=GRAY)
        fig.text(0.955, 0.03, "DeteCam · sim/rt/demo_video.py · 한 번의 예 (seed %d). 통계는 README" % D["seed"], fontsize=7.5, color=GRAY, ha="right")
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    if a.still is not None:
        from PIL import Image
        Image.fromarray(draw(a.still)).save("/tmp/demo_still_%04.1f.png" % a.still); print("still", a.still, "total", round(total, 1)); return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wtr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    nf = int(total * FPS)
    for i in range(nf):
        wtr.stdin.write(np.ascontiguousarray(draw(i / FPS)).tobytes())
    wtr.stdin.close(); wtr.wait(); print("영상", a.out, "%.1f초" % total)


if __name__ == "__main__":
    main()
