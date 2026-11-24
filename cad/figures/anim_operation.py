"""DeteCam 5.0 (Palm-12) — 벽면 부착 AR-OPA + TriF v12 + RAPS 90% AR 동작 시뮬레이션 영상 생성기.

    python3 cad/figures/anim_operation.py

1280x720, 24fps, 24초:
  Part 1 (0~12초): 40석 대형 금속 칠판 교실 — 손 닿는 벽면 2.0m 부착 + 각도 딸깍(±30°/-15°) -> RAPS 90% AR 6석 압축
  Part 2 (12~24초): 금속 사물함 탈의실 — 자석·나노겔 벽면 부착 + 각도 조절 -> 10면 가상 거울 앵커 -> 0.28 m 정밀 수렴
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


def draw_wall_node(ax, x, y, aim_deg, label, va="bottom", dy=0.35):
    """벽면 부착 Palm-12 퍽 + 스위블 지향 빔 부채꼴 시각화."""
    ax.add_patch(Wedge((x, y), 1.05, aim_deg - 38, aim_deg + 38, facecolor="#19C2B0", alpha=0.14, edgecolor="#19C2B0", lw=1.2, ls="--", zorder=3))
    ax.scatter([x], [y], s=105, c="#1E2D5A", marker="s", zorder=6)
    ax.text(x, y + dy, label, ha="center", va=va, fontsize=8.3, color="#1E2D5A", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.92), zorder=7)


def render_frame(fig, ax, t):
    ax.clear()
    ax.set_facecolor("#FFFFFF")
    if t < 12.0:
        L, W_r = 8.5, 7.5
        ax.set_xlim(-0.7, L + 0.7); ax.set_ylim(-0.7, W_r + 0.8); ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), L, W_r, fill=True, facecolor="#FBFBF8", edgecolor="#1E2D5A", lw=2.4))
        ax.add_patch(Rectangle((2.45, 0.0), 3.6, 0.16, facecolor="#5C6670", edgecolor="#1E2D5A", lw=1.2))
        ax.text(4.25, -0.30, "1200×3600 mm 대형 금속 칠판", ha="center", va="top", fontsize=9.8, color="#4F5A6B", fontweight="bold")
        ax.add_patch(Rectangle((3.7, 0.75), 1.1, 0.55, facecolor="#D9C8B4", edgecolor="#7D6B58", lw=1.2))
        ax.text(4.25, 1.02, "교탁", ha="center", va="center", fontsize=8.5, color="#3A2F24")

        gt = (5.45, 4.75)
        xs = np.linspace(1.15, 7.35, 8)
        ys = np.linspace(2.05, 6.45, 5)
        u = t / 12.0
        n_photos = 0 if u < 0.25 else min(4, int((u - 0.25) / 0.16) + 1)
        sigma = {0: 4.0, 1: 1.35, 2: 0.95, 3: 0.70, 4: 0.50}[n_photos]
        seats = [(x, y) for y in ys for x in xs]
        scores = np.array([np.exp(-((x - gt[0]) ** 2 + (y - gt[1]) ** 2) / (2 * sigma ** 2)) for x, y in seats])
        probs = scores / scores.sum()
        order = np.argsort(-probs)
        raps_set = set(order[: (14 if n_photos == 1 else (9 if n_photos == 2 else 6))]) if n_photos > 0 else set()

        for idx, (x, y) in enumerate(seats):
            in_raps = idx in raps_set
            fc = "#FFE2D4" if in_raps else "#EEF1F5"
            ec = "#EB6834" if in_raps else "#B5BECB"
            ax.add_patch(Rectangle((x - 0.30, y - 0.22), 0.60, 0.44, facecolor=fc, edgecolor=ec, lw=2.0 if in_raps else 0.9, zorder=2))
            if in_raps and n_photos >= 2:
                ax.text(x, y, f"{probs[idx]*100:.0f}%", ha="center", va="center", fontsize=7.5, color="#B83B0E", fontweight="bold", zorder=3)

        ax.scatter([gt[0]], [gt[1]], s=120, c="#EB6834", marker="*", zorder=6)

        # 실제 ar_opa_results.json 벽면 부착 최적 3노드 (스탠드 0개!)
        nodes = [
            (0.08, 3.6, -20, "노드 1: 왼벽 2.0m\n좌우 -30° · 아래 -15°", "bottom", 0.55),
            (8.42, 3.6, 200, "노드 2: 오른벽 2.0m\n좌우 +30° · 아래 -15°", "bottom", 0.55),
            (4.25, 7.42, -80, "노드 3: 뒷벽 2.0m\n좌우 +15° · 아래 -15°", "bottom", 0.25),
        ]
        if t > 0.4:
            for nx, ny, ad, lbl, va, dy in nodes:
                draw_wall_node(ax, nx, ny, ad, lbl, va=va, dy=dy)
                if n_photos >= 1:
                    ax.plot([nx, gt[0]], [ny, gt[1]], color="#19C2B0", alpha=0.60, lw=1.8, zorder=4)

        stage = ("1단계: AR-OPA 벽면 부착 가이드 (손 닿는 2.0m · 스탠드 0개 · 각도 딸깍 1초)" if n_photos == 0
                 else f"2단계: 사진 {n_photos}장 포착 -> TriF v12(결합보정) + RAPS 90% 신뢰 AR 구역 ({len(raps_set)}석 압축)")
        ax.set_title(f"[시험장] 40석 대형 금속 칠판 교실 — {stage}", fontsize=13.0, fontweight="bold", color="#1E2D5A", pad=12)
        ax.set_xlabel("가로 (m)", fontsize=10); ax.set_ylabel("세로 (m)", fontsize=10)
    else:
        tt = t - 12.0
        L, W_r = 5.5, 4.0
        ax.set_xlim(-0.8, L + 0.8); ax.set_ylim(-0.7, W_r + 0.7); ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), L, W_r, fill=True, facecolor="#FBFBF8", edgecolor="#1E2D5A", lw=2.4))
        ax.add_patch(Rectangle((0.0, 0.5), 0.55, 3.0, facecolor="#8C95A0", edgecolor="#2E3640", lw=1.5))
        ax.text(0.27, 2.0, "금속\n사물함", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.add_patch(Rectangle((4.95, 0.5), 0.55, 3.0, facecolor="#8C95A0", edgecolor="#2E3640", lw=1.5))
        ax.text(5.22, 2.0, "금속\n사물함", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.add_patch(Rectangle((2.3, 1.1), 0.9, 1.8, facecolor="#D9C8B4", edgecolor="#7D6B58", lw=1.2))
        ax.text(2.75, 2.0, "벤치", ha="center", va="center", fontsize=8.5, color="#3A2F24")
        ax.add_patch(Rectangle((1.5, 3.88), 2.5, 0.12, facecolor="#78C2D4", edgecolor="#1E2D5A", lw=1.2))
        ax.text(2.75, 4.10, "대형 거울 (경면 반사벽)", ha="center", va="bottom", fontsize=9.2, color="#1E2D5A", fontweight="bold")

        gt = (0.68, 2.75)
        ax.scatter([gt[0]], [gt[1]], s=125, c="#EB6834", marker="*", zorder=6)
        ax.text(gt[0] + 0.14, gt[1] + 0.14, "은닉 몰래카메라", fontsize=9, color="#EB6834", fontweight="bold")

        # 실제 ar_opa_results.json 탈의실 벽면 부착 최적 3노드
        nodes = [
            (1.15, 3.92, -60, "노드 1: 뒷벽 2.0m\n+30° / -15°", "top", -0.35),
            (4.35, 3.92, -120, "노드 2: 뒷벽 2.0m\n-30° / -15°", "top", -0.35),
            (3.45, 0.08, 105, "노드 3: 앞벽 1.1m\n-15° / 0°", "top", -0.18),
        ]
        u = tt / 12.0
        for nx, ny, ad, lbl, va, dy in nodes:
            draw_wall_node(ax, nx, ny, ad, lbl, va=va, dy=dy)

        if u > 0.22:
            refl_pt = (2.2, 3.88)
            ax.plot([gt[0], refl_pt[0], 3.45], [gt[1], refl_pt[1], 0.08], color="#EB6834", lw=1.8, ls=":", alpha=0.82)
            ax.plot([1.15, gt[0]], [3.92, gt[1]], color="#19C2B0", lw=1.8, alpha=0.65)
            ax.plot([4.35, gt[0]], [3.92, gt[1]], color="#19C2B0", lw=1.8, alpha=0.65)
            rad = max(0.28, 1.30 * (1.0 - min(1.0, (u - 0.22) / 0.55)))
            ax.add_patch(Circle(gt, rad, facecolor="#EB6834", alpha=0.18, edgecolor="#EB6834", lw=2.2, zorder=3))
            ax.text(gt[0] + rad + 0.12, gt[1] - 0.15, f"90% AR 반경 {rad:.2f} m", fontsize=9.5, color="#B83B0E", fontweight="bold")

        stage = ("벽면 스위블 배치 + 10면 가상 거울 앵커 -> 중앙값 0.28 m 정밀 수렴! (P90 0.60 m)"
                 if u > 0.25 else "AR-OPA: 자석·나노겔 벽부착 + 각도 딸깍 3점 자동 계산 (0.1초 · 각도고정 대비 P90 -20%)")
        ax.set_title(f"[사적 공간] 금속 사물함 탈의실 — {stage}", fontsize=13.0, fontweight="bold", color="#1E2D5A", pad=12)
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
    print(f"Palm-12 operation video saved: {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
