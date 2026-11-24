"""DeteCam 6.0 (Nano-2 AI) — 단 2기 초소형 벽면 노드 + 가상 거울 앵커 + AI 초해상도 동작 시뮬레이션 영상.

    python3 cad/figures/anim_operation.py

1280x720, 24fps, 24초:
  Part 1 (0~12초): L자형 꺾인 객실 (극한 코너 사각지대) — 단 2기 포켓 노드(9.6cm) + 거울 반사 가상 앵커로 벽 뒤 카메라 0.17m 정밀 포착!
  Part 2 (12~24초): 40석 대형 금속 칠판 교실 (미학습 공간) — 단 2기 벽면 스위블 + ReconCovNet-v4 + RAPS 90% AR (0.19m · 4석 압축)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "model")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib import font_manager
import paths

W, H, FPS = 1280, 720, 24
DUR = 24.0
OUT = os.path.join(paths.VID, "배치_동작_시뮬레이션.mp4")

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def draw_wall_node(ax, x, y, aim_deg, label, va="bottom", dy=0.35, color="#0EA5A4", ghost=False):
    al = 0.10 if ghost else 0.15
    ax.add_patch(Wedge((x, y), 1.05, aim_deg - 38, aim_deg + 38, facecolor=color, alpha=al, edgecolor=color, lw=1.2, ls="--", zorder=3))
    ax.scatter([x], [y], s=110, c="#38BDF8" if ghost else "#1E2D5A", marker="s", alpha=0.55 if ghost else 1.0, zorder=6)
    ax.text(x, y + dy, label, ha="center", va=va, fontsize=8.3,
            color="#0284C7" if ghost else "#1E2D5A", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="#E0F2FE" if ghost else "white", ec="#38BDF8" if ghost else "#CBD5E1", alpha=0.92), zorder=7)


def render_frame(fig, ax, t):
    ax.clear()
    ax.set_facecolor("#FFFFFF")
    if t < 12.0:
        # Part 1: L자형 꺾인 객실 — 단 2기 포켓 노드 + 가상 거울 앵커로 코너 사각지대 해결!
        L, W_r = 6.0, 4.8
        ax.set_xlim(-1.45, L + 0.65); ax.set_ylim(-0.65, W_r + 0.70); ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), L, W_r, fill=True, facecolor="#FBFBF8", edgecolor="#1E2D5A", lw=2.4))
        # L자 구조 콘크리트 벽체 (직접 전파 차단 사각지대!)
        ax.add_patch(Rectangle((3.1, 2.4), 2.9, 2.4, facecolor="#CBD5E1", edgecolor="#334155", lw=2.0, hatch="///", zorder=4))
        ax.text(4.55, 3.6, "콘크리트 L자 코너\n(직선 전파 차단벽)", ha="center", va="center", fontsize=9.2, color="#1E293B", fontweight="bold", zorder=5)
        # 침대 & 대형 복도 거울
        ax.add_patch(Rectangle((0.3, 3.0), 1.9, 1.5, facecolor="#E2D5C3", edgecolor="#7D6B58", lw=1.2, zorder=2))
        ax.text(1.25, 3.75, "침대", ha="center", va="center", fontsize=8.5, color="#3A2F24", zorder=3)
        ax.add_patch(Rectangle((-0.08, 1.0), 0.14, 1.7, facecolor="#38BDF8", edgecolor="#0284C7", lw=2.0, zorder=5))
        ax.text(-0.22, 1.85, "복도\n대형거울", ha="right", va="center", fontsize=8.5, color="#0284C7", fontweight="bold")

        # 숨겨진 카메라 (L자 코너 우측 구석!)
        gt = (4.85, 1.45)
        ax.scatter([gt[0]], [gt[1]], s=140, c="#EB6834", marker="*", zorder=8)
        ax.text(gt[0], gt[1] - 0.34, "은닉 카메라\n(코너 뒤 사각지대)", ha="center", va="top", fontsize=8.5, color="#C2410C", fontweight="bold", zorder=8)

        # 단 2기의 초소형 Nano-2 (9.6 cm) 벽면 노드!
        n1 = (0.10, 4.10)  # 왼벽 상단 (직접 시야는 L코너에 막힘!)
        n2 = (5.90, 0.45)  # 오른벽 하단 (직접 시야 확보)
        draw_wall_node(ax, n1[0], n1[1], -35, "노드 1 (9.6cm)\n왼벽 2.0m · -15°", va="bottom", dy=0.22)
        draw_wall_node(ax, n2[0], n2[1], 145, "노드 2 (9.6cm)\n오른벽 1.1m · +30°", va="top", dy=-0.22)

        u = t / 12.0
        # 1) 막힌 직선 경로 (회색 점선 X)
        ax.plot([n1[0], gt[0]], [n1[1], gt[1]], color="#94A3B8", lw=1.4, ls="--", alpha=0.65, zorder=3)
        if u > 0.15:
            ax.plot([n2[0], gt[0]], [n2[1], gt[1]], color="#0EA5A4", lw=2.2, alpha=0.80, zorder=5)
        if u > 0.30:
            # 2) 가상 거울 앵커 (왼벽 거울 바깥의 홀로그램 3번째 눈!) + 반사 광선!
            ax.plot([n1[0], 0.0, gt[0]], [n1[1], 2.05, gt[1]], color="#0284C7", lw=2.2, ls="-.", alpha=0.90, zorder=6)
            # 가상 거울 고스트 노드 (거울 뒤 대칭 위치)
            draw_wall_node(ax, -0.85, 2.95, -20, "가상 거울 앵커\n(0원 공짜 3번째 눈!)", va="bottom", dy=0.22, color="#0284C7", ghost=True)
            ax.plot([-0.85, 0.0], [2.95, 2.05], color="#38BDF8", lw=1.6, ls=":", alpha=0.85, zorder=5)

            rad = max(0.18, 1.35 * (1.0 - min(1.0, (u - 0.30) / 0.50)))
            ax.add_patch(Circle(gt, rad, facecolor="#EB6834", alpha=0.20, edgecolor="#EB6834", lw=2.2, zorder=6))
            ax.text(gt[0] - 0.10, gt[1] + rad + 0.15, f"MA-PINN AI 추정 오차 {rad:.2f} m", ha="center", va="bottom", fontsize=9.2, color="#B83B0E", fontweight="bold", zorder=8)

        stage = ("단 2기 포켓 노드 + MA-PINN 가상 거울 앵커 -> 코너 뒤 카메라 0.17 m 정밀 포착!" if u > 0.30
                 else "L자 코너 사각지대: 노드 1 직선 전파가 콘크리트 벽에 막힌 극한 상황")
        ax.set_title(f"[DeteCam 6.0 Nano-2 AI] L자형 꺾인 객실 — {stage}", fontsize=12.4, fontweight="bold", color="#1E2D5A", pad=12)
        ax.set_xlabel("가로 (m)", fontsize=10); ax.set_ylabel("세로 (m)", fontsize=10)
    else:
        # Part 2: 40석 대형 금속 칠판 교실 (미학습 공간!) — 단 2기만으로 3기 초월!
        tt = t - 12.0
        L, W_r = 8.5, 7.5
        ax.set_xlim(-0.7, L + 0.7); ax.set_ylim(-0.7, W_r + 0.8); ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), L, W_r, fill=True, facecolor="#FBFBF8", edgecolor="#1E2D5A", lw=2.4))
        ax.add_patch(Rectangle((2.45, 0.0), 3.6, 0.16, facecolor="#5C6670", edgecolor="#1E2D5A", lw=1.2))
        ax.text(4.25, -0.30, "3.6m 대형 금속 칠판 (MA-PINN이 칠판 반사도 가상 앵커로 활용!)", ha="center", va="top", fontsize=9.5, color="#0284C7", fontweight="bold")
        ax.add_patch(Rectangle((3.7, 0.75), 1.1, 0.55, facecolor="#D9C8B4", edgecolor="#7D6B58", lw=1.2))
        ax.text(4.25, 1.02, "교탁", ha="center", va="center", fontsize=8.5, color="#3A2F24")

        gt = (5.45, 4.75)
        xs = np.linspace(1.15, 7.35, 8)
        ys = np.linspace(2.05, 6.45, 5)
        u = tt / 12.0
        n_photos = 0 if u < 0.22 else min(4, int((u - 0.22) / 0.17) + 1)
        sigma = {0: 4.0, 1: 1.15, 2: 0.78, 3: 0.55, 4: 0.36}[n_photos]
        seats = [(x, y) for y in ys for x in xs]
        scores = np.array([np.exp(-((x - gt[0]) ** 2 + (y - gt[1]) ** 2) / (2 * sigma ** 2)) for x, y in seats])
        probs = scores / scores.sum()
        order = np.argsort(-probs)
        raps_set = set(order[: (10 if n_photos == 1 else (7 if n_photos == 2 else 4))]) if n_photos > 0 else set()

        for idx, (x, y) in enumerate(seats):
            in_raps = idx in raps_set
            fc = "#FFE2D4" if in_raps else "#EEF1F5"
            ec = "#EB6834" if in_raps else "#B5BECB"
            ax.add_patch(Rectangle((x - 0.30, y - 0.22), 0.60, 0.44, facecolor=fc, edgecolor=ec, lw=2.0 if in_raps else 0.9, zorder=2))
            if in_raps and n_photos >= 2:
                ax.text(x, y, f"{probs[idx]*100:.0f}%", ha="center", va="center", fontsize=7.5, color="#B83B0E", fontweight="bold", zorder=3)

        ax.scatter([gt[0]], [gt[1]], s=130, c="#EB6834", marker="*", zorder=6)

        # 단 2기 벽면 부착 (AR-OPA-2 최적 배치!)
        nodes = [
            (0.08, 2.2, 22, "노드 1 (9.6cm)\n왼벽 2.0m · +30°/-15°", "bottom", 0.55),
            (8.42, 6.1, 210, "노드 2 (9.6cm)\n오른벽 1.1m · -30°/0°", "top", -0.55),
        ]
        for nx, ny, ad, lbl, va, dy in nodes:
            draw_wall_node(ax, nx, ny, ad, lbl, va=va, dy=dy)
            if n_photos >= 1:
                ax.plot([nx, gt[0]], [ny, gt[1]], color="#0EA5A4", alpha=0.72, lw=2.0, zorder=4)
        if n_photos >= 2:
            # 칠판 경면 반사 가상 앵커 경로 표시!
            ax.plot([0.08, 3.2, gt[0]], [2.2, 0.16, gt[1]], color="#0284C7", lw=1.8, ls="-.", alpha=0.80, zorder=4)
            ax.text(3.2, 0.42, "칠판 반사 가상 앵커(3번째 눈)", ha="center", va="bottom", fontsize=8.2, color="#0284C7", fontweight="bold")

        stage = ("단 2기 벽면 부착 (AR-OPA-2 · 15초 설치 · 스탠드 0개)" if n_photos == 0
                 else f"사진 {n_photos}장 -> ReconCovNet-v4 + MA-PINN (0.19m · RAPS 90% AR {len(raps_set)}석 압축!)")
        ax.set_title(f"[미학습 공간] 40석 대형 금속 칠판 교실 — {stage}", fontsize=12.4, fontweight="bold", color="#1E2D5A", pad=12)
        ax.set_xlabel("가로 (m)", fontsize=10); ax.set_ylabel("세로 (m)", fontsize=10)

    ax.grid(alpha=0.18, linestyle="--")
    fig.tight_layout(pad=1.3)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def main():
    os.makedirs(paths.VID, exist_ok=True)
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100, facecolor="#F7F6F2")
    n_frames = int(round(DUR * FPS))
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n_frames):
        proc.stdin.write(render_frame(fig, ax, i / FPS).tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    print(f"Nano-2 AI operation video saved: {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
