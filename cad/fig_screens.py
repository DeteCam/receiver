"""수신기 화면 그림: 폰 브라우저로 본 화면을 나란히.   python3 fig_screens.py <화면 그림이 있는 폴더>
   29_수신기_화면.png   탐지·판별
   30_폰_조종.png       위치 재기, 추정 구간, 손으로 움직이기, 멈춤
화면 그림은 수신기 프로그램을 가짜 모드(node.py --dry)로 띄우고 찍은 것이다 (firmware/tests/screens.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "asset", "screens")
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
W, H = 2360, 1560; m = 40


def load(fn, pw, hmax):
    s = Image.open(os.path.join(SRC, fn)).convert("RGB"); ph = int(s.size[1] * pw / s.size[0]); s = s.resize((pw, ph), Image.LANCZOS)
    return s.crop((0, 0, pw, min(ph, hmax)))


def sheet(out, title, sub, cols, pw):
    """cols = [(제목, 설명, [화면 파일 ...])]. 화면 파일이 여럿이면 그 칸에 위아래로 쌓는다"""
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((m, 28), title, font=font(50, True), fill=INK); d.text((m, 100), sub, font=font(28), fill=SUB)
    gap = (W - 2 * m - len(cols) * pw) // max(1, len(cols) - 1)
    for i, (head, cap, files) in enumerate(cols):
        x = m + i * (pw + gap); y = 300
        d.text((x, 190), head, font=font(36, True), fill=INK)
        for k, l in enumerate(cap.split("\n")):
            d.text((x, 240 + k * 32), l, font=font(23), fill=SUB)
        room = (1130 - 40 * (len(files) - 1)) // len(files)
        for fn in files:
            s = load(fn, pw, room)
            d.rounded_rectangle([x - 8, y + 12, x + pw + 8, y + s.size[1] + 28], 26, fill="#FFFFFF", outline=LINE, width=3)
            im.paste(s, (x, y + 20)); y += s.size[1] + 40
    im.save(os.path.join(paths.IMG, out)); print("saved", out)


NOTE = "수신기의 Wi-Fi 에 붙어 http://192.168.4.1 을 열면 나온다. 아래는 장치 없이 가짜 모드로 띄워 찍은 것이다 (기기 주소와 숫자는 가짜)."
sheet("29_수신기_화면.png", "수신기 화면: 폰 브라우저에서 탐지와 판별을 하는 모습", NOTE, [
    ("1 탐지", "둘레 기기를 훑은 표. 큰 프레임을 많이 내보내는\n기기에 ‘카메라 후보’ 표시가 붙는다. 줄을 누르면 고른다.", ["web_1_detect.png"]),
    ("2 판별 (하는 중)", "화면이 ‘걷기 / 멈춤’과 남은 시간을 알려 준다.\n바뀔 때마다 소리와 진동이 난다.", ["web_2_prompt.png"]),
    ("2 판별 (결과)", "막대 = 0.5초마다의 전송량, 주황 띠 = 움직이라고 한 때.\n박자를 따라 오르면 ‘카메라로 보임’.", ["web_3_result.png"])], 640)
sheet("30_폰_조종.png", "폰으로 조종하기: 재기, 추정 구간, 손으로 움직이기, 멈춤", NOTE, [
    ("3 위치 재기 · 4 잰 자리", "놓은 자리를 적고 ‘이 자리에서 재기’.\n자리마다 찾은 방향이 표에 쌓인다.", ["web_4_measure.png"]),
    ("5 추정 구간", "주황 = 카메라가 있을 만한 곳.\n초록 동그라미 = 다음에 놓으면 좋은 자리.", ["web_5_zone.png"]),
    ("손으로 움직이기", "관절 셋을 조금씩 돌리고, 판 0° 맞추기,\n모터 시험, 다시 켜기, 끄기까지 여기서 한다.", ["web_6_control.png"]),
    ("멈춤", "무엇을 하는 중이든 맨 위에 ‘멈춤’이 뜬다.\n누르면 모터 전류까지 끄고 선다.", ["web_7_moving.png", "web_8_stopped.png"])], 530)
