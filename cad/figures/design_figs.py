"""설계도와 데모 환경 그림.  python3 design_figs.py   ->  docs/images/설계도_*.png, 도면_패널.png, 데모_환경_*.png

치수 그림(도면_패널)의 숫자는 cad/model/panel.py 에서, 교실과 방의 크기는 sim/rt 의 장면(scene_kr, scene_room)에서 가져온다.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Arc

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "images")
sys.path[:0] = [os.path.join(ROOT, "cad", "model"), os.path.join(ROOT, "sim", "rt")]
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, INK, BG = "#10304F", "#0E8F82", "#B36B00", "#E0483A", "#6B7780", "#0E1B26", "#F5F7FA"
MINT, WARM, PALE, LINE = "#E3F5F2", "#FFF4DF", "#EEF3F7", "#C9D3DC"


def canvas(w, h, title, sub=None):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150); fig.patch.set_facecolor(BG); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_position([0, 0, 1, 1])
    ax.text(0.35, h - 0.42, title, fontsize=17, fontweight="bold", color=INK, va="center")
    if sub: ax.text(0.35, h - 0.82, sub, fontsize=10.5, color=GRAY, va="center")
    return fig, ax


def box(ax, x, y, w, h, title=None, sub=None, fc="white", ec=NAVY, tc=INK, fs=11.5, lw=1.4, ls="-", r=0.1, ta="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=lw, ls=ls))
    n = 0 if not sub else sub.count("\n") + 1; tx = x + w / 2 if ta == "center" else x + 0.15
    if title: ax.text(tx, y + h / 2 + 0.11 * n, title, ha=ta, va="center", fontsize=fs, fontweight="bold", color=tc)
    if sub: ax.text(tx, y + h / 2 - 0.16 - 0.09 * (n - 1) if title else y + h / 2, sub, ha=ta, va="center", fontsize=fs - 2.5, color=GRAY if tc == INK else tc, linespacing=1.45)


def arrow(ax, p0, p1, text=None, color=NAVY, ls="-", both=False, lw=1.6, off=(0, 0.14), fs=9, ha="center"):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=13))
    if text: ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, fontsize=fs, color=color, ha=ha, va="center")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor=fig.get_facecolor()); plt.close(fig); print("ok", name)


def block(ax, x, y, title, lines, fs=9.5, lh=0.27, color=INK):
    """제목 한 줄과 그 아래의 줄들. 다음에 쓸 y 를 돌려준다"""
    ax.text(x, y, title, fontsize=11.5, fontweight="bold", color=NAVY, va="center"); y -= 0.36
    for ln in lines:
        ax.text(x, y, ln, fontsize=fs, color=color, va="center"); y -= lh
    return y - 0.22


def system():
    fig, ax = canvas(12.6, 6.6, "시스템 구성도", "카메라가 낸 Wi-Fi 전파를 패널이 받고, 허브가 계산하고, 폰이 보여 준다")
    box(ax, 0.35, 2.55, 2.3, 1.7, "찾을 대상", "Wi-Fi 카메라 (2.4GHz)\n숨긴 카메라 · 시험장의\n초소형 카메라", fc="#FDECEA", ec=RED)
    for k, rr in enumerate((0.35, 0.6, 0.85)):
        ax.add_patch(Arc((2.72, 3.4), 2 * rr, 2 * rr, theta1=-32, theta2=32, color=RED, lw=2, alpha=0.9 - 0.25 * k))
    box(ax, 3.75, 0.45, 5.0, 5.0, fc="white", ec=NAVY, lw=1.8, r=0.18); ax.text(3.98, 5.18, "수신기  receiver", fontsize=12.5, fontweight="bold", color=NAVY, va="center")
    box(ax, 4.0, 3.55, 2.05, 1.3, "패널 A", "안테나 16개\n+ ESP32-C5", fc=MINT, ec=TEAL); box(ax, 4.0, 1.95, 2.05, 1.3, "패널 B", "둘째 시점\n(같은 것 하나 더)", fc=MINT, ec=TEAL)
    box(ax, 6.55, 1.95, 2.0, 2.9, None, None, fc=NAVY, ec=NAVY); ax.text(7.55, 4.52, "허브", fontsize=12.5, fontweight="bold", color="white", ha="center", va="center")
    ax.text(7.55, 4.22, "라즈베리파이 3B+", fontsize=9, color="#B9C7D3", ha="center", va="center")
    ax.text(6.72, 3.2, "· 탐지 (의심 점수)\n· 세기 → 위상 → 방향\n· 두 패널, 전송마다 겹치기\n· 폰이 부르는 API", fontsize=9, color="white", va="center", linespacing=1.6)
    box(ax, 4.0, 0.7, 4.55, 0.9, None, "탐지 동글 (모든 채널을 엿듣는다)  ·  통신 동글 (폰과 잇는 Wi-Fi)\n보조배터리 (콘센트 없이 동작)", fc=WARM, ec=AMBER)
    arrow(ax, (6.05, 4.2), (6.55, 4.2)); arrow(ax, (6.05, 2.6), (6.55, 2.6)); ax.text(6.3, 3.4, "세기 값\nUSB ·\nESP-NOW", fontsize=6.8, color=NAVY, ha="center", va="center", linespacing=1.3)
    arrow(ax, (3.1, 3.75), (4.0, 4.2), color=RED, lw=1.2); arrow(ax, (3.1, 3.05), (4.0, 2.6), color=RED, lw=1.2)
    box(ax, 9.95, 3.3, 2.35, 1.9, "스마트폰 앱  app", "수신기 조종\n패널 건 자리 등록 (AR 마커)\n의심 자리 3곳을 AR 로 표시", fc="white", ec=TEAL, tc=TEAL)
    arrow(ax, (8.55, 4.0), (9.95, 4.0), "Wi-Fi · HTTP", both=True, off=(0.1, 0.2))
    box(ax, 9.95, 0.75, 2.35, 1.75, "분석  ai  (오프라인)", "전파 시뮬레이션과 측정 자료로\n의심 점수 · 자리 모델을 학습", fc="white", ec=GRAY, ls="--")
    arrow(ax, (9.95, 1.75), (8.55, 2.3), "모델 파일", color=GRAY, ls="--", off=(0.2, -0.3))
    ax.text(0.35, 0.22, "첫 시제품은 허브(가진 라즈베리파이)를 쓴다. 그다음에는 폰이 허브의 일을 맡도록 줄일 수 있다 (패널 ↔ 폰 블루투스).", fontsize=9, color=GRAY)
    save(fig, "설계도_시스템_구성.png")


def wiring():
    fig, ax = canvas(12.9, 7.3, "패널 배선도", "수신 길이 하나뿐인 ESP32 로 안테나 16개의 위상을 얻는 길: 기준 안테나와 합쳐 세기만 잰다")
    box(ax, 0.35, 0.95, 3.25, 4.75, fc="white", ec=NAVY, lw=1.6, r=0.14); ax.text(1.97, 5.42, "안테나 배열 보드", fontsize=11.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(1.97, 5.14, "CHW1010-ANT2 · 150 × 150 mm", fontsize=8.5, color=GRAY, ha="center")
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((0.72 + 0.66 * i, 2.25 + 0.66 * j), 0.5, 0.5, fc="#F5A623", ec="#9A6400", lw=0.8))
    ax.text(1.97, 2.02, "패치 안테나 16개 (간격 37.5mm)", fontsize=8.5, color=GRAY, ha="center")
    box(ax, 0.72, 1.15, 2.5, 0.6, "16갈래 스위치", None, fc=PALE, fs=9.5)
    box(ax, 4.35, 4.75, 1.55, 0.8, "기준 안테나", "ESP32 에 딸려 온 것", fc=WARM, ec=AMBER, fs=10)
    box(ax, 6.45, 4.75, 1.3, 0.8, "스위치 A", "1 → 4", fc=PALE, fs=10); box(ax, 9.25, 4.75, 1.3, 0.8, "스위치 B", "4 → 1", fc=PALE, fs=10)
    arrow(ax, (5.9, 5.15), (6.45, 5.15))
    for dy in (0.27, 0.09, -0.09, -0.27):
        ax.plot([7.75, 9.25], [5.15 + dy, 5.15 + dy], color=NAVY, lw=1.2)
    ax.text(8.5, 5.72, "길이가 다른 선 넷 (약 21mm 씩)", fontsize=8.5, color=NAVY, ha="center"); ax.text(8.5, 4.6, "0° · 90° · 180° · 270°", fontsize=8.5, color=GRAY, ha="center")
    box(ax, 9.25, 3.05, 1.3, 0.8, "합침기", "둘을 더한다", fc=PALE, fs=10); arrow(ax, (9.9, 4.75), (9.9, 3.85))
    arrow(ax, (3.6, 3.45), (9.25, 3.45), "고른 안테나 하나의 신호", off=(-1.2, 0.17))
    box(ax, 8.95, 1.05, 1.9, 1.15, "XIAO ESP32-C5", "세기만 잰다 (CSI 의 크기)", fc=WARM, ec=AMBER, fs=10.5); arrow(ax, (9.9, 3.05), (9.9, 2.2), "합친 신호", off=(0.55, 0.0))
    ax.plot([8.95, 6.0, 6.0, 3.6], [1.75, 1.75, 1.45, 1.45], color=TEAL, lw=1.4, ls="--"); ax.text(6.25, 1.92, "안테나 고르기 4가닥 + 켜기·제어 2가닥 (FPC 10핀)", fontsize=8.5, color=TEAL)
    ax.plot([10.85, 11.45, 11.45, 10.55], [1.6, 1.6, 5.15, 5.15], color=TEAL, lw=1.4, ls="--"); ax.plot([7.1, 7.1, 11.45, 11.45], [5.55, 6.02, 6.02, 5.15], color=TEAL, lw=1.4, ls="--")
    ax.text(11.6, 3.4, "위상 고르기\n2가닥\n(A·B 를 같이)", fontsize=8.5, color=TEAL, va="center", linespacing=1.4)
    box(ax, 4.2, 0.42, 4.3, 0.8, None, "전원: USB-C (허브나 충전기) 또는 18650 한 개\n한 번 재기 = 안테나 16 × 위상 4 = 패킷 64개 (약 0.3초)", fc="white", ec=LINE, fs=11)
    ax.text(0.35, 0.18, "실선: 전파가 지나는 길 (동축선)   점선: 스위치를 고르는 신호선", fontsize=9, color=GRAY)
    save(fig, "설계도_패널_배선.png")


def sequence():
    fig, ax = canvas(12, 5.7, "동작 순서", "찾기 버튼을 누른 뒤 의심 자리가 화면에 뜰 때까지")
    S = [("1  엿듣기", "모든 채널을 돌며\n누가 얼마나 보내는지 센다", PALE, NAVY), ("2  의심 점수", "카메라처럼 보내는 기기를\n고른다 (MAC 주소)", PALE, NAVY), ("3  찌르기", "그 기기에 빈 프레임을 보내\n응답(ACK)을 받는다", WARM, AMBER),
         ("4  세기 64번", "안테나 16 × 위상 4\n(약 0.3초)", WARM, AMBER), ("5  방향", "세기 → 위상 → 패널마다\n방향의 점수", MINT, TEAL), ("6  겹치기 · 쌓기", "두 패널의 점수를 곱하고\n전송이 올 때마다 다시", MINT, TEAL), ("7  표시", "의심 자리 3곳과 확률을\n폰 화면(AR)에", "white", TEAL)]
    xs = [0.35 + 2.92 * i for i in range(4)] + [0.35 + 2.92 * i + 1.46 for i in range(3)]; ys = [3.45] * 4 + [1.85] * 3
    for (t, s, fc, ec), x, y in zip(S, xs, ys):
        box(ax, x, y, 2.55, 1.15, t, s, fc=fc, ec=ec, fs=11)
    for i in range(3): arrow(ax, (xs[i] + 2.55, 4.02), (xs[i + 1], 4.02))
    xa, xb = xs[3] + 1.27, xs[4] + 1.27
    ax.plot([xa, xa, xb], [3.45, 3.22, 3.22], color=NAVY, lw=1.6, solid_capstyle="round"); arrow(ax, (xb, 3.22), (xb, 3.0))
    for i in (4, 5): arrow(ax, (xs[i] + 2.55, 2.42), (xs[i + 1], 2.42))
    ax.text(xs[4], 1.62, "전송 한 번마다 3~7 을 되풀이한다 (사진 한 장을 보내는 0.2~1초 안에 한 번 잰다)", fontsize=9.5, color=AMBER, va="center")
    ax.text(0.35, 3.22, "1~2: 약 30초", fontsize=9.5, color=GRAY, va="center")
    ax.text(xs[1] + 1.275, 3.335, "5 GHz 기기는 여기까지 (위치 재기는 막음)", fontsize=8.5, color=RED, ha="center", va="center")
    ax.plot([0.6, 11.4], [0.95, 0.95], color=LINE, lw=2)
    for t, lab in ((0, "첫 전송\n카메라를 알아챔"), (30, "둘째 전송\n첫 자리 후보"), (60, "셋째"), (90, "넷째"), (120, "다섯째\n네 번 쌓은 결과")):
        x = 0.9 + t * 0.076; ax.add_patch(Rectangle((x - 0.05, 0.8), 0.1, 0.3, fc=RED, ec="none")); ax.text(x, 0.6, "%d초" % t, fontsize=8.5, color=GRAY, ha="center"); ax.text(x + 0.12, 1.32, lab, fontsize=8.5, color=INK, va="center", linespacing=1.3)
    ax.text(0.6, 0.22, "보기: 30초마다 사진 한 장을 보내는 카메라. 계속 영상을 보내는 카메라는 1초에 세 번까지 잰다", fontsize=9, color=GRAY)
    save(fig, "설계도_동작_순서.png")


def software():
    fig, ax = canvas(12.4, 7.0, "소프트웨어 구조", "저장소 셋이 나눠 맡는 일과 지금 상태 (2026-11-17)")
    COLS = [("패널 장치 코드", "receiver · ESP-IDF (C)", [("엿듣기", "채널을 돌며 프레임의 머리만 읽는다", 2), ("찌르기와 응답 받기", "빈 프레임 → ACK 의 CSI", 2), ("스위치 순서", "안테나 16 × 위상 4 를 패킷에 맞춘다", 2), ("이득 보정", "패킷마다의 수신 이득을 되돌린다", 2), ("허브로 보내기", "세기 묶음 (USB · ESP-NOW)", 2)]),
            ("허브 프로그램", "receiver · Python", [("탐지기", "특징을 뽑아 의심 점수를 매긴다", 0), ("패널 계산", "세기 → 위상 → 방향의 점수", 0), ("위치 추정", "두 패널, 전송마다 겹쳐 자리 확률", 1), ("보정", "스위치·선의 위상, 패널의 자세", 1), ("API 와 폰 화면", "폰이 부르는 주소, 측정 기록", 1)]),
            ("스마트폰 앱", "app · Kotlin · ARCore", [("연결과 조종", "찾기 · 재기 · 멈춤", 2), ("패널 자리 등록", "패널의 마커를 비춰 자리와 기울기를", 2), ("자리표 · 방 훑기", "교실의 자리, 방의 벽과 가구", 2), ("AR 표시", "의심 자리 3곳과 확률", 2), ("기록 보기", "언제, 어느 기기, 어느 자리", 2)]),
            ("분석과 학습", "ai · Python · PyTorch", [("학습 자료 만들기", "전파 시뮬레이션 (Sionna RT)", 0), ("의심 점수 모델", "카메라인지 가려낸다", 1), ("자리 모델", "물리 계산 × 배운 모델", 1), ("실측으로 고치기", "시뮬레이션과 실물의 차이를 줄인다", 2), ("평가", "정확도 표와 그림", 1)])]
    ST = [(MINT, TEAL, "만들어 검사함 (장치 없이)"), (WARM, AMBER, "일부 만듦"), ("white", GRAY, "아직")]
    for c, (name, sub, mods) in enumerate(COLS):
        x = 0.35 + 3.0 * c; box(ax, x, 0.75, 2.75, 5.25, fc="white", ec=LINE, lw=1.2, r=0.14)
        ax.text(x + 0.2, 5.7, name, fontsize=12.5, fontweight="bold", color=NAVY, va="center"); ax.text(x + 0.2, 5.42, sub, fontsize=8.5, color=GRAY, va="center")
        for k, (t, s, st) in enumerate(mods):
            fc, ec, _ = ST[st]; box(ax, x + 0.15, 4.35 - 0.86 * k, 2.45, 0.72, t, s, fc=fc, ec=ec, fs=10, lw=1.1, ta="left")
    xx = 6.3
    for fc, ec, lab in ST:
        ax.add_patch(Rectangle((xx, 0.28), 0.3, 0.2, fc=fc, ec=ec, lw=1.1)); ax.text(xx + 0.38, 0.38, lab, fontsize=9, color=INK, va="center"); xx += 0.6 + 0.135 * len(lab)
    ax.text(0.35, 0.38, "패널 → 허브: 세기 묶음.  허브 → 앱: HTTP + JSON.  분석 → 허브: 모델 파일.", fontsize=9, color=GRAY, va="center")
    save(fig, "설계도_소프트웨어_구조.png")


def drawing():
    import panel as P
    fig, ax = canvas(13.4, 7.4, "패널과 허브의 치수 (mm)", "cad/model/panel.py 의 값. 보드에 맞춘 값([실측])은 부품을 받은 뒤 다시 잰다")
    k = 0.019                                                                    # mm -> 그림 단위
    def dim(p0, p1, t, off=(0, 0), color=RED, bg=None):
        ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<->", color=color, lw=1.0, shrinkA=0, shrinkB=0))
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], t, fontsize=9.5, color=color, ha="center", va="center", fontweight="bold", bbox=dict(fc=bg, ec="none", pad=1.5) if bg else None)
    W, y0, y1 = 2 * P.OX, P.OY0, P.OY1; H = y1 - y0; top = 5.95; cy = top - y1 * k; Y = lambda y: cy + y * k; BLUE, DARK = "#7A88B5", "#2B3A8C"
    # 앞
    cx = 2.75; X = lambda x: cx + x * k
    ax.add_patch(FancyBboxPatch((X(-P.OX), Y(y0)), W * k, H * k, boxstyle="round,pad=0,rounding_size=0.12", fc=BLUE, ec=NAVY, lw=1.4))
    ax.add_patch(Rectangle((X(-75), Y(-75)), 150 * k, 150 * k, fc="none", ec="white", lw=1.0, ls="--"))
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((X(-56.25 + 37.5 * i - 13), Y(-56.25 + 37.5 * j - 13)), 26 * k, 26 * k, fc="none", ec="#F5D36B", lw=0.9))
    mx, my = P.MARK_C; ax.add_patch(Rectangle((X(mx - P.MARK[0] / 2), Y(my - P.MARK[1] / 2)), P.MARK[0] * k, P.MARK[1] * k, fc="white", ec=NAVY, lw=0.8))
    for dx in (-P.MARK_DX, P.MARK_DX):
        ax.add_patch(Rectangle((X(mx + dx - P.MARK_SIZE / 2), Y(my - P.MARK_SIZE / 2)), P.MARK_SIZE * k, P.MARK_SIZE * k, fc=INK, ec="none"))
    for (bx, by) in P.BOSS: ax.add_patch(Circle((X(bx), Y(by)), 0.035, fc="white", ec=NAVY, lw=0.6))
    dim((X(-P.OX), Y(y1) + 0.2), (X(P.OX), Y(y1) + 0.2), "%.1f" % W, (0, 0.15)); dim((X(-P.OX) - 0.22, Y(y0)), (X(-P.OX) - 0.22, Y(y1)), "%.1f" % H, (-0.38, 0))
    dim((X(-75), Y(0)), (X(75), Y(0)), "보드 150", color="white", bg=BLUE); dim((X(-56.25), Y(56.25) + 0.36), (X(-18.75), Y(56.25) + 0.36), "37.5", (0, 0.13), color="#F5D36B")
    ax.text(cx, Y(y0) - 0.3, "앞 (안테나 쪽). 아래 흰 띠는 AR 마커 둘", fontsize=10, color=INK, ha="center")
    # 옆
    sx = 6.1; zf, zb, zm = P.ZC, P.ZO, P.ZO - P.MT_T
    ax.add_patch(Rectangle((sx + zb * k, Y(y0)), (zf - zb) * k, H * k, fc=DARK, ec=NAVY, lw=1.2)); ax.add_patch(Rectangle((sx + zm * k, Y(-P.MT_W / 2)), P.MT_T * k, P.MT_W * k, fc="#E0703A", ec="#8A3C12", lw=1.0))
    ax.add_patch(Rectangle((sx - 3.6 * k, Y(-75)), 3.6 * k, 150 * k, fc="#F5D36B", ec="none"))
    dim((sx + zm * k, Y(y1) + 0.2), (sx + zf * k, Y(y1) + 0.2), "%.1f" % (zf - zm), (0, 0.15)); dim((sx + zb * k, Y(y0) - 0.14), (sx + zf * k, Y(y0) - 0.14), "%.1f" % (zf - zb), (0, -0.15))
    ax.text(sx + (zf + zm) / 2 * k, Y(y0) - 0.58, "옆. 통 + 덮개 %.1f,\n자석 걸이판(주황) %d" % (zf - zb, P.MT_T), fontsize=10, color=INK, ha="center", va="center", linespacing=1.4)
    # 뒤
    bx0 = 8.75; XB = lambda x: bx0 + x * k
    ax.add_patch(FancyBboxPatch((XB(-P.OX), Y(y0)), W * k, H * k, boxstyle="round,pad=0,rounding_size=0.12", fc=DARK, ec=NAVY, lw=1.4))
    ax.add_patch(FancyBboxPatch((XB(-P.MT_W / 2), Y(-P.MT_W / 2)), P.MT_W * k, P.MT_W * k, boxstyle="round,pad=0,rounding_size=0.08", fc="#E0703A", ec="#8A3C12", lw=1.0))
    for (qx, qy) in P.MAGS: ax.add_patch(Circle((XB(qx), Y(qy)), P.MAG_D / 2 * k, fc="#C8CDD2", ec="#6B7780", lw=0.8))
    for (qx, qy) in P.MOUNT: ax.add_patch(Circle((XB(qx), Y(qy)), 0.035, fc="white", ec=NAVY, lw=0.6))
    ax.add_patch(Rectangle((XB(P.OX) - 0.02, Y(P.XIAO_C[1]) - 0.045), 0.07, 0.09, fc="#C8CDD2", ec=NAVY, lw=0.6)); ax.text(XB(P.OX) + 0.1, Y(P.XIAO_C[1]), "USB-C", fontsize=8.5, color=INK, va="center")
    dim((XB(-P.MT_W / 2), Y(-P.MT_W / 2) - 0.2), (XB(P.MT_W / 2), Y(-P.MT_W / 2) - 0.2), "걸이판 %d" % P.MT_W, (0, -0.15), color="white")
    dim((XB(P.MOUNT[0][0]), Y(P.MT_W / 2) + 0.2), (XB(-P.MOUNT[0][0]), Y(P.MT_W / 2) + 0.2), "나사 사이 %d" % (2 * abs(P.MOUNT[0][0])), (0, 0.15), color="white")
    ax.text(bx0, Y(y0) - 0.3, "뒤. 회색 원: 자석 지름 %d 넷" % P.MAG_D, fontsize=10, color=INK, ha="center")
    # 허브
    q = 0.55; hl, hw, hh = (P.HL + 2 * P.HWALL), (P.HW_ + 2 * P.HWALL), P.HFLOOR + P.HH + 2.4; hx = 11.05; hy = 4.75
    ax.add_patch(FancyBboxPatch((hx, hy), hl * k * q, hw * k * q, boxstyle="round,pad=0,rounding_size=0.06", fc=DARK, ec=NAVY, lw=1.2)); ax.add_patch(Rectangle((hx, hy - 1.45), hl * k * q, hh * k * q, fc=DARK, ec=NAVY, lw=1.2))
    dim((hx, hy + hw * k * q + 0.18), (hx + hl * k * q, hy + hw * k * q + 0.18), "%.1f" % hl, (0, 0.15)); dim((hx + hl * k * q + 0.14, hy), (hx + hl * k * q + 0.14, hy + hw * k * q), "%.1f" % hw, (0.3, 0))
    dim((hx + hl * k * q + 0.14, hy - 1.45), (hx + hl * k * q + 0.14, hy - 1.45 + hh * k * q), "%.1f" % hh, (0.3, 0))
    ax.text(hx + hl * k * q / 2, hy - 1.78, "허브 (위, 옆)\n줄여 그림", fontsize=10, color=INK, ha="center", va="center", linespacing=1.4)
    ax.text(0.35, 0.3, "패널 약 0.6kg.  출력 부품 8종, 약 22시간, 필라멘트 0.8kg.  거는 법: 자석(칠판) · 흡착 걸이(유리) · 받침 발(창턱, 교탁, 삼각대).", fontsize=9, color=GRAY)
    save(fig, "도면_패널.png")


def _scene_kr():
    """교실 장면 모듈에서 자리 좌표만 쓴다. Sionna 가 없는 곳에서는 빈 모듈로 대신한다 (장면을 만들지는 않는다)."""
    import importlib, types
    for m in ("trimesh", "mitsuba", "sionna", "sionna.rt"):
        try: importlib.import_module(m)
        except Exception: sys.modules[m] = types.ModuleType(m)
    import scene_kr
    return scene_kr


def demo_class():
    K = _scene_kr()
    K.set_room("kr40"); fig, ax = canvas(12.6, 7.6, "데모 환경 1: 강의실 (시험장)", "40석 교실 7.5 × 9.0m. 패널 둘을 마주 보는 두 옆벽의 2m 높이에 걸고, 자리에 둔 카메라를 찾는다")
    k = 0.63; ox, oy = 1.25, 0.62; X = lambda x: ox + x * k; Y = lambda y: oy + (K.D - y) * k
    ax.add_patch(Rectangle((X(0), Y(K.D)), K.W * k, K.D * k, fc="white", ec=NAVY, lw=1.8)); ax.add_patch(Rectangle((X(1.35), Y(0) - 0.05), 4.8 * k, 0.1, fc=TEAL, ec="none")); ax.text(X(3.75), Y(0) + 0.15, "칠판", fontsize=9, color=TEAL, ha="center")
    ax.add_patch(Rectangle((X(3.35), Y(1.4)), 0.8 * k, 0.5 * k, fc=PALE, ec=NAVY, lw=1)); ax.text(X(3.75), Y(1.15), "교탁", fontsize=8, color=INK, ha="center", va="center")
    cams = {8: ("①", "책상 위"), 21: ("②", "가슴"), 33: ("③", "안경")}
    for i, (x, y) in enumerate(K.SEATS):
        c = i in cams; ax.add_patch(Rectangle((X(x) - 0.19, Y(y) - 0.12), 0.38, 0.24, fc=RED if c else "#DCE4EC", ec="none"))
        if c: ax.text(X(x), Y(y), cams[i][0], fontsize=9, color="white", ha="center", va="center", fontweight="bold")
    for nm, (px, py) in (("패널 A", (0.08, 2.50)), ("패널 B", (7.42, 2.50))):          # 2026-11-18: A 를 창 유리 앞쪽으로 (docs/배치 계산.md)
        ax.add_patch(Rectangle((X(px) - 0.07, Y(py) - 0.2), 0.14, 0.4, fc=TEAL, ec="none"))
        ax.text(X(px) + (-0.18 if px < 1 else 0.18), Y(py), nm + "\n높이 2m", fontsize=9, color=TEAL, ha="right" if px < 1 else "left", va="center", fontweight="bold", linespacing=1.3)
        for i in cams: ax.plot([X(px), X(K.SEATS[i][0])], [Y(py), Y(K.SEATS[i][1])], color=TEAL, lw=0.8, alpha=0.5)
    ax.add_patch(Rectangle((X(4.25), Y(1.35)), 0.3, 0.2, fc=NAVY, ec="none")); ax.text(X(4.8), Y(1.19), "허브", fontsize=9, color=NAVY, va="center", fontweight="bold")
    ax.add_patch(Circle((X(5.6), Y(0.75)), 0.11, fc=AMBER, ec="none")); ax.text(X(5.82), Y(0.75), "감독관 (폰)", fontsize=9, color=AMBER, va="center", fontweight="bold")
    ax.text(X(0) - 0.14, Y(7.6), "창", fontsize=9, color=GRAY, ha="center", va="center"); ax.text(X(K.W) + 0.2, Y(6.5), "복도", fontsize=9, color=GRAY, ha="center", va="center")
    ax.annotate("", (X(0), Y(K.D) - 0.2), (X(K.W), Y(K.D) - 0.2), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1)); ax.text(X(3.75), Y(K.D) - 0.36, "7.5 m", fontsize=9, color=GRAY, ha="center")
    tx = 7.35; y = 6.25
    y = block(ax, tx, y, "준비물", ["· 수신기 한 벌 (패널 2, 허브 1), 스마트폰", "· 찾을 카메라: 책상 위 필통 속 ①, 가슴 단추 자리 ②,", "  안경 자리 ③ (한 번에 하나씩)", "· 카메라가 붙을 폰 핫스팟 (교실 안 가방 속이나 교실 밖)", "· 자리에 앉아 줄 사람 (없으면 빈 교실에서 먼저)"])
    y = block(ax, tx, y, "순서", ["1. 패널을 걸고 폰으로 패널의 마커를 비춰 자리를 등록한다", "2. 카메라가 영상(또는 30초마다 사진)을 보낸다", "3. 앱에서 찾기: 약 30초 뒤 의심 기기가 뜬다", "4. 패널이 전송마다 재고, 의심 자리 3곳이 뜬다", "5. 카메라를 다른 자리로 옮겨 되풀이한다 (20자리)"])
    y = block(ax, tx, y, "통과 기준", ["· 30초 안에 의심 기기로 올린 비율 90% 이상", "· 사진 4장이면 의심 좌석 3곳 안 90% 이상, 16장이면 95% 이상", "· 찾기부터 자리 후보까지 3분 이하", "· 재는 동안 소리가 나지 않는다"])
    ax.text(tx, y + 0.08, "계산으로는 의심 좌석 3곳 안에 사진 4장 91~99%, 16장 97~100%.\n실물 값은 이 데모에서 처음 나온다.", fontsize=9, color=GRAY, va="center", linespacing=1.5)
    save(fig, "데모_환경_교실.png")


def demo_room():
    import scene_room as R
    fig, ax = canvas(12.6, 7.3, "데모 환경 2: 숙소 방 (몰카)", "방 3.6 × 5.4m, 현관 옆에 욕실. 패널 하나로 네 자리에서 재서 숨긴 카메라의 자리를 찾는다")
    k = 0.97; ox, oy = 0.75, 0.75; X = lambda x: ox + x * k; Y = lambda y: oy + y * k
    ax.add_patch(Rectangle((X(0), Y(0)), R.W * k, R.D * k, fc="white", ec=NAVY, lw=1.8))
    ax.add_patch(Rectangle((X(0), Y(0)), R.BX * k, R.BY * k, fc="#EAF1F6", ec=NAVY, lw=1.4)); ax.text(X(0.85), Y(1.8), "욕실", fontsize=9.5, color=GRAY, ha="center")
    ax.plot([X(R.BX), X(R.BX)], [Y(1.2), Y(1.9)], color="white", lw=3); ax.plot([X(2.4), X(3.3)], [Y(0), Y(0)], color="white", lw=3.5); ax.text(X(2.85), Y(0) - 0.17, "현관문", fontsize=9, color=GRAY, ha="center")
    ax.plot([X(0.9), X(2.9)], [Y(R.D), Y(R.D)], color="#7FC8F8", lw=4); ax.text(X(3.3), Y(R.D) + 0.15, "창", fontsize=9, color=GRAY, ha="center")
    for (x0, y0, x1, y1, nm) in ((0.15, 3.0, 2.15, 4.6, "침대"), (0.10, 2.45, 0.50, 2.90, ""), (3.05, 2.6, 3.6, 4.2, "책상\n·TV"), (3.10, 4.3, 3.58, 4.8, ""), (1.80, 0.05, 2.30, 0.60, "옷장")):
        ax.add_patch(Rectangle((X(x0), Y(y0)), (x1 - x0) * k, (y1 - y0) * k, fc="#F0EBE1", ec="#B9AE98", lw=1)); ax.text(X((x0 + x1) / 2), Y((y0 + y1) / 2 - (0.45 if nm.startswith("책상") else 0)), nm, fontsize=8.5, color="#7A6F5A", ha="center", va="center", linespacing=1.2)
    HIDE = [("①", 2.6, 3.9, "천장 화재감지기"), ("②", 0.07, 4.95, "벽시계·액자"), ("③", 3.38, 3.85, "TV·셋톱박스 둘레"), ("④", 1.15, 5.3, "에어컨·커튼 상자"), ("⑤", 0.13, 3.3, "침대 머리맡 콘센트"), ("⑥", 3.0, 5.12, "창가 화분·휴지통"),
            ("⑦", 2.05, 0.33, "옷장 위·현관문 위"), ("⑧", 0.85, 1.1, "욕실 환풍기"), ("⑨", 0.1, 0.7, "욕실 선반·거울")]
    for s, x, y, _ in HIDE:
        ax.add_patch(Circle((X(x), Y(y)), 0.15, fc=RED, ec="white", lw=1)); ax.text(X(x), Y(y), s, fontsize=9.5, color="white", ha="center", va="center", fontweight="bold")
    mx, my = R.MID[0], R.MID[1]
    for (nm, pos, aim) in R.PLACES:
        if nm.startswith("가운데") and "천장" not in nm:
            d = np.array(aim[:2]) - np.array(pos[:2]); d = 0.55 * d / np.linalg.norm(d); ax.annotate("", (X(mx + d[0]), Y(my + d[1])), (X(mx), Y(my)), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6, mutation_scale=12))
    for lab, (x, y), ha, dx, dy in (("가  방 가운데\n(네 쪽 + 천장)", (mx, my), "left", 0.2, -0.45), ("나  창턱", (1.9, R.D - 0.1), "left", 0.2, -0.02), ("다  현관 통로", (2.75, 2.0), "right", -0.2, 0.0), ("라  욕실 안", (1.45, 0.35), "right", -0.2, 0.0)):
        ax.add_patch(Rectangle((X(x) - 0.11, Y(y) - 0.11), 0.22, 0.22, fc=TEAL, ec="white", lw=1)); ax.text(X(x) + dx, Y(y) + dy, lab, fontsize=9, color=TEAL, ha=ha, va="center", fontweight="bold", linespacing=1.3)
    ax.annotate("", (X(0), Y(0) - 0.42), (X(R.W), Y(0) - 0.42), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1)); ax.text(X(1.0), Y(0) - 0.28, "3.6 m", fontsize=9, color=GRAY, ha="center")
    tx = 5.0; y = 6.2
    ax.text(tx, y, "숨겨 볼 곳 (빨강)", fontsize=11.5, fontweight="bold", color=NAVY, va="center")
    for i, (s, _, _, n) in enumerate(HIDE):
        ax.text(tx + (0 if i < 5 else 3.1), y - 0.36 - 0.27 * (i if i < 5 else i - 5), "%s %s" % (s, n), fontsize=9.5, color=INK, va="center")
    y -= 0.36 + 0.27 * 5 + 0.22
    y = block(ax, tx, y, "재는 자리 (청록)", ["가. 방 가운데에서 패널을 네 쪽과 천장으로 돌려 가며 (삼각대)", "나. 창턱에 세워 방 안쪽을 본다", "다. 현관 통로에서 현관 쪽을 본다", "라. 욕실 안 세면대 위에서 욕실 안쪽을 본다"])
    y = block(ax, tx, y, "순서와 통과 기준", ["1. 카메라를 ①~⑨ 가운데 한 곳에 숨기고 영상을 보내게 한다", "2. 앱에서 찾기 → 의심 기기 → 가·나·다·라에서 잰다", "   (자리마다 폰으로 패널의 마커를 비춰 자리와 방향을 등록)", "3. 화면에 뜬 자리와 참 자리 사이를 줄자로 잰다", "통과: 가운데값 0.5m 이하, 1m 안 80% 이상 (곳마다 10번)"])
    ax.text(tx, y + 0.08, "계산으로는 가운데값 0.2m, 50cm 안 89%.  기숙사 방이나 세미나실처럼\n허락받은 곳에서, 내 카메라만 쓴다.", fontsize=9, color=GRAY, va="center", linespacing=1.5)
    save(fig, "데모_환경_방.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (system, wiring, sequence, software, drawing, demo_class, demo_room):
        f()
