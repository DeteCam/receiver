"""수신기 화면 그림 (29_수신기_화면.png): 폰 브라우저로 본 탐지·판별 화면 석 장을 나란히.   python3 fig_screens.py <화면 그림이 있는 폴더>
화면 그림은 수신기 프로그램을 가짜 모드(node.py --dry)로 띄우고 찍은 것이다."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "asset", "screens")
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
SHOTS = [("web_1_detect.png", "1 탐지", "둘레 기기를 훑은 표. 큰 프레임을 많이 내보내는\n기기에 ‘카메라 후보’ 표시가 붙는다. 줄을 누르면 고른다."),
         ("web_2_prompt.png", "2 판별 (하는 중)", "화면이 ‘걷기 / 멈춤’과 남은 시간을 알려 준다.\n바뀔 때마다 소리와 진동이 난다."),
         ("web_3_result.png", "2 판별 (결과)", "막대 = 0.5초마다의 전송량, 주황 띠 = 움직이라고 한 때.\n박자를 따라 오르면 ‘카메라로 보임’.")]
W, H = 2360, 1560; m = 40
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "수신기 화면: 폰 브라우저에서 탐지와 판별을 하는 모습", font=font(50, True), fill=INK)
d.text((m, 100), "수신기의 Wi-Fi 에 붙어 http://192.168.4.1 을 열면 나온다. 아래는 장치 없이 가짜 모드로 띄워 찍은 것이다 (기기 주소와 숫자는 가짜).", font=font(28), fill=SUB)
pw = 640; gap = (W - 2 * m - 3 * pw) // 2
for i, (fn, title, cap) in enumerate(SHOTS):
    s = Image.open(os.path.join(SRC, fn)).convert("RGB"); ph = int(s.size[1] * pw / s.size[0]); s = s.resize((pw, ph), Image.LANCZOS)
    ph = min(ph, 1130); s = s.crop((0, 0, pw, ph)); x = m + i * (pw + gap); y = 300
    d.text((x, 190), title, font=font(36, True), fill=INK)
    for k, l in enumerate(cap.split("\n")):
        d.text((x, 240 + k * 32), l, font=font(23), fill=SUB)
    d.rounded_rectangle([x - 8, y + 12, x + pw + 8, y + ph + 28], 26, fill="#FFFFFF", outline=LINE, width=3)
    im.paste(s, (x, y + 20))
im.save(os.path.join(paths.IMG, "29_수신기_화면.png")); print("saved 29_수신기_화면.png")
