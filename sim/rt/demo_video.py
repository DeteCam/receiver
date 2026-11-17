"""demo_run.json (패널 둘이 '사진을 가끔 보내는 카메라'의 자리를 좁혀 가는 과정)을 동작 시뮬레이션 영상으로 만든다.

    python demo_video.py [--out ../../docs/videos/교실_동작_시뮬레이션.mp4] [--still 12.5]

위에서 본 교실(치수 표시)에 패널 둘을 걸어 두고, 카메라가 사진을 보낼 때마다 두 패널이 잰 방향과
자리 확률이 좁혀지는 모습을 보여 준다. 숫자는 모두 demo_run.json 의 계산값이다.
"""
import os, json, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Circle, Polygon

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, BG, INK = "#10304F", "#19C2B0", "#FFB020", "#E0483A", "#6B7780", "#F5F7FA", "#0E1B26"
HERE = os.path.dirname(os.path.abspath(__file__)); W, H, FPS = 1280, 720, 24
T_IN, T_WAIT, T_BURST, T_UP, T_OUT = 4.0, 1.6, 1.2, 1.2, 7.0
PCOL = (TEAL, "#7A5CFA")                               # 패널 A, B 의 색
# rt_study 의 통계 (CoreHW 간격 패널 둘, 창 유리 가운데 + 복도 창턱 앞쪽. 자리를 정확히 / 옆자리까지)
STAT = [("1번", (70, 90), (56, 81), (70, 93)), ("2번", (78, 94), (68, 89), (81, 97)), ("4번", (80, 95), (80, 94), (88, 99)), ("8번", (84, 97), (87, 97), (90, 100))]


def heat(p):
    t = min(1.0, float(p) ** 0.5); a = np.array([1, 1, 1.0]); b = np.array([1.0, 0.69, 0.125]); c = np.array([0.878, 0.282, 0.227])
    return tuple(a + (b - a) * (t / 0.5)) if t < 0.5 else tuple(b + (c - b) * ((t - 0.5) / 0.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs", "videos", "교실_동작_시뮬레이션.mp4")); ap.add_argument("--still", type=float, nargs="*")
    a = ap.parse_args(); D = json.load(open(os.path.join(HERE, "demo_run.json"), encoding="utf-8"))
    seats = np.array(D["seats"]); steps = D["steps"]; n = len(steps); RW, RD = D["room"][0], D["room"][1]; cam = D["cam"]; seat = D["seat"]; spots = np.array(D["spots"]); per_s = D["period"]
    xs = np.unique(np.round(seats[:, 0], 3)); ys = np.unique(np.round(seats[:, 1], 3))
    lab = lambda k: "%d째 줄 %d번" % (int(np.argmin(np.abs(ys - seats[k, 1]))) + 1, int(np.argmin(np.abs(xs - seats[k, 0]))) + 1)
    per = T_WAIT + T_BURST + T_UP; total = T_IN + n * per + T_OUT; uni = np.full(len(seats), 1.0 / len(seats))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)

    def draw(t):
        fig.clf(); fig.patch.set_facecolor(BG)
        k = int(np.clip((t - T_IN) // per, 0, n - 1)) if t >= T_IN else -1; tt = t - T_IN - k * per if k >= 0 else 0.0; done = t >= T_IN + n * per
        phase = "in" if k < 0 else ("out" if done else ("wait" if tt < T_WAIT else ("burst" if tt < T_WAIT + T_BURST else "up")))
        shown = (n - 1) if done else (k if phase == "up" else k - 1)                    # 자리 확률에 반영된 마지막 전송
        post = np.array(steps[shown]["p_cum"]) if shown >= 0 else uni
        if phase == "up":
            prev = np.array(steps[k - 1]["p_cum"]) if k > 0 else uni; u = (tt - T_WAIT - T_BURST) / T_UP; post = prev + (np.array(steps[k]["p_cum"]) - prev) * u
        pk = min(max(k, 0), n - 1); ppl = np.array(steps[pk]["people"])
        if phase == "wait" and k > 0:                                                   # 기다리는 동안 사람들이 자세를 바꾼다
            p0 = np.array(steps[k - 1]["people"]); u = min(1.0, tt / T_WAIT); ppl = p0 + (ppl - p0) * u
        clock = 0.0 if k < 0 else (n * per_s if done else (k * per_s + per_s * min(1.0, tt / T_WAIT)))
        # ---- 왼쪽: 위에서 본 교실 ----
        ax = fig.add_axes([0.03, 0.05, 0.47, 0.86]); ax.set_xlim(-1.25, RW + 1.45); ax.set_ylim(RD + 0.75, -0.95); ax.set_aspect("equal"); ax.axis("off")
        room = Rectangle((0, 0), RW, RD, fc="white", ec=NAVY, lw=2.2); ax.add_patch(room); ax.add_patch(Rectangle((1.95, 0.03), 3.6, 0.1, fc="#2E5E4E", ec="none")); ax.text(RW / 2, -0.2, "칠판 3.6 m", ha="center", fontsize=9, color=GRAY)
        ax.add_patch(Rectangle((RW / 2 - 0.4, 0.9), 0.8, 0.5, fc="#D8C3A5", ec="none")); ax.text(RW / 2, 1.15, "교탁 · 허브", ha="center", va="center", fontsize=7.5, color="#6B5B45")
        ax.plot([0, 0], [0.5, RD - 0.5], color="#8FC1E3", lw=5, solid_capstyle="butt", zorder=1); ax.text(-0.3, 1.6, "창", ha="center", va="center", fontsize=9, color="#4A86B4")
        ax.plot([RW, RW], [1.6, 7.4], color="#8FC1E3", lw=3, solid_capstyle="butt", zorder=1); ax.text(RW + 0.72, 6.6, "복도 쪽\n높은 창", ha="center", va="center", fontsize=8, color="#4A86B4")
        ax.annotate("", (0, RD + 0.4), (RW, RD + 0.4), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9)); ax.text(RW / 2, RD + 0.62, "%.1f m" % RW, ha="center", fontsize=8.5, color=GRAY)
        ax.annotate("", (-0.75, 0), (-0.75, RD), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9)); ax.text(-0.98, RD / 2, "%.1f m" % RD, rotation=90, va="center", ha="center", fontsize=8.5, color=GRAY)
        for i, (x, y) in enumerate(seats):
            hot = post[i] > 0.02 and (shown >= 0 or phase == "up")
            ax.add_patch(Rectangle((x - 0.3, y - 0.2), 0.6, 0.4, fc=heat(post[i]) if hot else "white", ec="#B36B00" if hot else "#9AA7B2", lw=1.3 if hot else 0.7, zorder=3))
            ax.add_patch(Circle((ppl[i][0], ppl[i][1]), 0.16, fc="#CBD3DA", ec="none", zorder=2))
        ax.add_patch(Circle((cam[0], cam[1]), 0.13, fc="none", ec=RED, lw=2.0, zorder=6))
        ax.annotate("카메라 (실제 자리)", (cam[0], cam[1]), (cam[0] + 0.5, cam[1] - 0.62), fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8), zorder=7)
        vk = k if (k >= 0 and phase in ("burst", "up")) else (n - 1 if done else None)
        for j, sp in enumerate(spots):                                                  # 패널과 잰 방향
            nx = 1.0 if sp[0] < RW / 2 else -1.0
            ax.add_patch(Rectangle((sp[0] - 0.05 if nx > 0 else sp[0] - 0.09, sp[1] - 0.18), 0.14, 0.36, fc=PCOL[j], ec="none", zorder=8))
            ax.text(sp[0] - 0.32 * nx, sp[1] - 0.42, "패널 %s" % "AB"[j], ha="center", fontsize=9, color=PCOL[j], fontweight="bold")
            ax.text(sp[0] - 0.42 * nx, sp[1] + 0.62, "높이 %s m" % ("%.2f" % sp[2]).rstrip("0").rstrip("."), ha="center", fontsize=7, color=PCOL[j])
            if vk is not None:
                v = steps[vk]["views"][j]; az = np.radians(v["az"]); L = 8.5 if not done else 0.0
                if phase == "burst":
                    L *= min(1.0, (tt - T_WAIT) / (T_BURST * 0.6))
                if L > 0:
                    ln, = ax.plot([sp[0], sp[0] + L * np.cos(az)], [sp[1], sp[1] + L * np.sin(az)], color=PCOL[j], lw=2.2 if v["los"] else 1.4, ls="-" if v["los"] else (0, (4, 3)), alpha=0.85, zorder=5)
                    ln.set_clip_path(room)
        if phase == "burst":                                                            # 전송하는 순간: 카메라에서 퍼지는 고리
            u = (tt - T_WAIT) / T_BURST
            for r0 in (0.0, 0.33, 0.66):
                r_ = ((u + r0) % 1.0) * 1.6; ax.add_patch(Circle((cam[0], cam[1]), r_, fc="none", ec=RED, lw=1.4, alpha=max(0.0, 0.8 - r_ / 2.0), zorder=6))
        if shown >= 0 or phase == "up":
            b = int(np.argmax(post)); ax.add_patch(Rectangle((seats[b, 0] - 0.36, seats[b, 1] - 0.26), 0.72, 0.52, fc="none", ec=INK, lw=2.0, zorder=7))
        ax.set_title("위에서 본 교실 (40석)", fontsize=10, color=GRAY, loc="left", pad=2)
        # ---- 오른쪽 위: 제목과 시간 줄 ----
        fig.text(0.535, 0.925, "패널 수신기 동작 시뮬레이션", fontsize=19, color=INK, fontweight="bold")
        sub = {"in": "패널 둘을 창 유리와 복도 쪽 창턱에 걸어 둔다. 옮기지 않는다. 모터가 없어 소리가 나지 않는다",
               "wait": "카메라가 쉬는 동안에는 잴 수 없다. 기다린다 (사람들은 자세를 조금씩 바꾼다)",
               "burst": "%d번째 전송: 사진 한 장을 보내는 0.5초 안에 두 패널이 방향을 잰다 (패킷 64개, 0.3초)" % (k + 1),
               "up": "두 방향이 만나는 자리의 확률을 올리고, 앞에서 잰 것과 곱한다",
               "out": "전송 %d번(%d초)으로 자리를 좁혔다. 허브가 앱에 자리 후보를 알린다" % (n, n * per_s)}[phase]
        fig.text(0.535, 0.885, sub, fontsize=10.5, color=NAVY)
        at = fig.add_axes([0.535, 0.775, 0.43, 0.075]); at.set_xlim(-6, n * per_s + 6); at.set_ylim(-1, 1.6); at.axis("off")
        at.plot([0, n * per_s], [0, 0], color="#C9D1D9", lw=3, solid_capstyle="round")
        for i in range(n):
            x = (i + 1) * per_s; on = (i < k) or done or (i == k and phase in ("burst", "up"))
            at.plot([x, x], [-0.45, 0.45], color=RED if on else "#C9D1D9", lw=5, solid_capstyle="round"); at.text(x, 0.75, "%d초" % x, ha="center", fontsize=8, color=INK if on else GRAY)
        at.plot([clock], [0], marker="v", color=NAVY, ms=9, zorder=5); at.text(0, 0.75, "0", ha="center", fontsize=8, color=GRAY)
        at.text(-5, -0.95, "카메라는 %d초마다 0.5초 동안만 전파를 낸다 (사진을 찍어 보내는 방식)" % per_s, fontsize=8.5, color=GRAY)
        # ---- 오른쪽 가운데: 이번 전송에서 잰 것 ----
        am = fig.add_axes([0.535, 0.455, 0.43, 0.275]); am.set_xlim(0, 1); am.set_ylim(0, 1); am.axis("off")
        am.text(0.0, 0.93, "패널이 잰 방향 (위에서 본 각도)", fontsize=9.5, color=GRAY)
        if vk is not None and not done:
            for j, v in enumerate(steps[vk]["views"]):
                y0 = 0.66 - 0.33 * j; am.add_patch(Rectangle((0.0, y0 - 0.12), 1.0, 0.27, fc="white", ec="#DCE2E8", lw=0.8))
                am.add_patch(Rectangle((0.0, y0 - 0.12), 0.012, 0.27, fc=PCOL[j], ec="none"))
                am.text(0.03, y0 + 0.055, "패널 %s  %s" % ("AB"[j], D["spot_names"][j]), fontsize=10.5, color=INK, fontweight="bold")
                am.text(0.03, y0 - 0.065, "직접 보인다 (실선)" if v["los"] else "몸에 가려 돌아온 전파 (점선)", fontsize=9.5, color=NAVY if v["los"] else RED)
                am.text(0.97, y0 + 0.055, "방향 오차 %.0f도" % abs(v["err"][0]), fontsize=10.5, color=INK, ha="right")
                am.text(0.97, y0 - 0.065, "높이각 오차 %.0f도" % abs(v["err"][1]), fontsize=9.5, color=GRAY, ha="right")
        elif done:
            am.text(0.0, 0.74, "같은 교실에서 뽑은 통계 (자리를 정확히 / 옆자리까지, %)", fontsize=9.5, color=INK)
            for c_, h_ in enumerate(("전송", "책상 위", "가슴", "안경")):
                am.text(0.02 + 0.25 * c_, 0.60, h_, fontsize=9.5, color=GRAY)
            for r_, row in enumerate(STAT):
                y0 = 0.47 - 0.135 * r_; am.text(0.02, y0, row[0], fontsize=10.5, color=INK, fontweight="bold")
                for c_, (e_, n_) in enumerate(row[1:]):
                    am.text(0.27 + 0.25 * c_, y0, "%d / %d" % (e_, n_), fontsize=10.5, color=INK)
        else:
            am.text(0.0, 0.62, "돌판 수신기는 한 곳에서 110초 동안 신호가 이어져야 방향을 잰다.", fontsize=10, color=GRAY)
            am.text(0.0, 0.47, "사진만 가끔 보내는 카메라는 그렇게 잴 수 없다.", fontsize=10, color=GRAY)
            am.text(0.0, 0.27, "패널은 안테나 16개를 전자 스위치로 바꿔 가며 0.3초에 잰다.", fontsize=10, color=INK)
        # ---- 오른쪽 아래: 자리 후보 ----
        ab = fig.add_axes([0.535, 0.085, 0.43, 0.30]); ab.set_xlim(0, 1.0); ab.set_ylim(-0.6, 2.6); ab.invert_yaxis(); ab.axis("off"); ab.set_title("자리 후보 3개와 확률 (전송마다 곱해 간다)", fontsize=9.5, color=GRAY, loc="left", pad=2)
        if shown >= 0 or phase == "up":
            o = np.argsort(-post)[:3]
            for r_, i in enumerate(o):
                ab.add_patch(Rectangle((0.34, r_ - 0.3), 0.62, 0.6, fc="#E7ECF1", ec="none")); ab.add_patch(Rectangle((0.34, r_ - 0.3), 0.62 * float(post[i]), 0.6, fc=RED if i == seat else AMBER, ec="none"))
                ab.text(0.0, r_, lab(i), va="center", fontsize=11.5, color=INK, fontweight="bold" if r_ == 0 else "normal"); ab.text(0.97, r_, "%d%%" % round(float(post[i]) * 100), va="center", ha="right", fontsize=11, color=INK)
        else:
            ab.text(0.0, 1.0, "아직 재지 않았다 (40자리 모두 같은 확률)", va="center", fontsize=10.5, color=GRAY)
        fig.text(0.965, 0.03, "DeteCam · sim/rt/demo_video.py · 한 번의 예 (seed %d, %s 카메라). 통계는 sim/rt/README" % (D["seed"], D["kind"]), fontsize=7.5, color=GRAY, ha="right")
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    if a.still is not None:
        from PIL import Image
        for ts in a.still:
            Image.fromarray(draw(ts)).save("/tmp/demo_still_%05.1f.png" % ts)
        print("still", a.still, "total", round(total, 1)); return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wtr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)):
        wtr.stdin.write(np.ascontiguousarray(draw(i / FPS)).tobytes())
    wtr.stdin.close(); wtr.wait(); print("영상", a.out, "%.1f초" % total)


if __name__ == "__main__":
    main()
