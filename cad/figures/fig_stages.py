"""네 단계 그림 (13_네_단계.png): 탐지 -> 판별 -> 위치 측정 -> 표시. 단계마다 하는 일, 쓰는 장치, 안 될 때의 길.   python3 fig_stages.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import paths
from PIL import Image, ImageDraw, ImageFont

OUT = paths.IMG + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG, CARD = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE", "#FFFFFF"
BLUE, ORANGE, GREEN, NAVY = "#2A78D6", "#EB6834", "#1E8E5A", "#1B2A57"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)

W, H = 2360, 1560; m = 40
im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "카메라를 찾는 네 단계: 탐지 → 판별 → 위치 측정 → 표시", font=font(50, True), fill=INK)
d.text((m, 100), "네트워크를 몰라도 된다. 수신기는 카메라가 내보내는 전파만 듣는다. 앞 단계가 찾은 것을 뒤 단계가 받아 쓴다.", font=font(28), fill=SUB)

STAGES = [
    ("1", "탐지", "약 30초", "둘레의 Wi-Fi 기기를 훑어\n카메라 후보를 추린다",
     ["채널 13개를 한 번 훑고, 바쁜 채널", "3개를 5초씩 자세히 듣는다", "큰 프레임(평균 300바이트 이상)을", "초당 30개 넘게 내보내면 후보"],
     "탐지 동글 (모니터 모드)", "동글이 없으면 내장 Wi-Fi 의\nCSI 수로 '초당 몇 개'만 본다"),
    ("2", "판별", "55초", "이 방을 찍는 카메라가\n맞는지 확인한다",
     ["화면·소리·진동이 시키는 대로", "8초 멈춤, 8초 움직임을 3번", "전송량이 박자를 따라 오르면", "'카메라', 아니면 '의심 기기'"],
     "탐지 동글 또는 내장 Wi-Fi + 폰", "반응이 없으면: 불을 껐다 켜며,\n또는 방을 나갔다 들어오며 다시"),
    ("3", "위치 측정", "한 자리 약 1분 × 보통 3자리", "어느 방향, 어느 높이에\n있는지 잰다",
     ["판을 세 방향으로 돌려 신호가 꺼지는", "각(골)을 찾는다 → 방향과 높이", "골이 한 번만 보이면 그쪽을 가로질러", "한 번 더 잰다 (집중 스캔)"],
     "라즈베리파이 + 모터 3개 + 판", "좁은 칸에서는 '여섯 번 재기'.\n자리를 옮겨 2~4곳에서 잰다"),
    ("4", "표시", "바로", "카메라가 있을 만한 범위를\n방 안에 보여 준다",
     ["방향 선들이 만나는 범위 =", "추정 구간 (3D 상자)", "폰이 수신기 등판의 마커를 보고", "수신기가 놓인 자리를 안다"],
     "폰 (브라우저 → AR 앱) + 종이 마커", "지금은 브라우저의 평면 그림.\nAR 앱은 만들 것"),
]
PASS = ["채널과\n기기 주소", "카메라 /\n의심 기기", "추정 구간\n(상자)"]
bw, bh, gap = 500, 760, 86; x0 = m; y0 = 190
for i, (n, name, tm, goal, lines, dev, alt) in enumerate(STAGES):
    x = x0 + i * (bw + gap); col = [BLUE, ORANGE, GREEN, NAVY][i]
    d.rounded_rectangle([x, y0, x + bw, y0 + bh], 22, fill=CARD, outline=col, width=5)
    d.ellipse([x + 26, y0 + 26, x + 106, y0 + 106], fill=col); d.text((x + 66, y0 + 64), n, font=font(46, True), fill="#FFFFFF", anchor="mm")
    d.text((x + 124, y0 + 22), name, font=font(46, True), fill=col); d.text((x + 126, y0 + 82), tm, font=font(25), fill=SUB)
    for k, l in enumerate(goal.split("\n")):
        d.text((x + 28, y0 + 136 + k * 44), l, font=font(32, True), fill=INK)
    d.line([(x + 28, y0 + 242), (x + bw - 28, y0 + 242)], fill=LINE, width=2)
    for k, l in enumerate(lines):
        d.text((x + 28, y0 + 262 + k * 40), l, font=font(26), fill=INK)
    d.text((x + 28, y0 + 446), "쓰는 장치", font=font(23, True), fill=SUB)
    d.rounded_rectangle([x + 24, y0 + 484, x + bw - 24, y0 + 548], 12, fill="#EEF1F6", outline=None)
    d.text((x + bw / 2, y0 + 516), dev, font=font(26, True), fill=INK, anchor="mm")
    d.text((x + 28, y0 + 574), "안 될 때", font=font(23, True), fill=SUB)
    for k, l in enumerate(alt.split("\n")):
        d.text((x + 28, y0 + 610 + k * 38), l, font=font(25), fill=INK)
    if i < 3:                                              # 다음 단계로 넘기는 것
        ax0, ax1 = x + bw + 8, x + bw + gap - 8; ay = y0 + 64
        d.line([(ax0, ay), (ax1 - 14, ay)], fill=SUB, width=6); d.polygon([(ax1, ay), (ax1 - 22, ay - 13), (ax1 - 22, ay + 13)], fill=SUB)
        for k, l in enumerate(PASS[i].split("\n")):
            d.text(((ax0 + ax1) / 2, y0 + 112 + k * 28), l, font=font(20, True), fill=SUB, anchor="mm")

# ---- 검토에서 정한 것
ys = y0 + bh + 44
d.text((m, ys), "처음 생각에서 검토하며 정한 것", font=font(36, True), fill=INK)
NOTES = [
    (BLUE, "탐지", "내장 Wi-Fi 는 CSI 를 받는 펌웨어라 프레임의 길이를 알려 주지 않는다. 그래서 모니터 모드가 되는 USB 동글을 하나 더 꽂는다 (없어도 돌지만 덜 정확하다)."),
    (ORANGE, "판별", "'보행 박자'는 폰 화면·소리·진동으로 알려 준다. 반응이 1.2배보다 약한 카메라는 놓치므로, 확정이 안 돼도 '의심 기기'로 두고 위치 측정으로 넘어간다."),
    (GREEN, "위치", "'신호 판독 AI'는 지금은 골 모양을 수식에 맞추는 계산이다. 실물로 잰 자료가 쌓이면 학습한 모델로 바꿀 수 있게 그 부분만 따로 떼어 두었다."),
    (NAVY, "표시", "수신기는 '수신기에서 본' 방향만 안다. 방 안에 그리려면 수신기가 어디에 어느 쪽을 보고 놓였는지 알아야 해서, 등판에 종이 마커(ArUco 2개)를 붙인다."),
    (SUB, "곳", "모텔 방은 잘 된다. 욕실(타일·거울)은 반사가 세서 '여섯 번 재기'와 자리 옮기기로. 강의실은 방향은 맞지만 멀어서 구간이 넓다. 변기 칸은 '여섯 번 재기'로. 철판 칸막이 칸은 반사가 너무 세서 위치는 믿기 어렵다."),
]
for k, (col, tag, text) in enumerate(NOTES):
    y = ys + 64 + k * 92
    d.rounded_rectangle([m, y, m + 130, y + 56], 12, fill=col); d.text((m + 65, y + 27), tag, font=font(28, True), fill="#FFFFFF", anchor="mm")
    words = text.split(" "); line = ""; yy = y - 6
    for wd in words:
        t = (line + " " + wd).strip()
        if d.textlength(t, font=font(27)) > W - 2 * m - 170:
            d.text((m + 156, yy), line, font=font(27), fill=INK); yy += 38; line = wd
        else:
            line = t
    d.text((m + 156, yy), line, font=font(27), fill=INK)
im.save(OUT + '13_네_단계.png'); print("saved", OUT + '13_네_단계.png', im.size)
