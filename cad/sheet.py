"""부품을 출력 방향 그대로 한 장에 늘어놓은 그림 (확인용)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, render, export2 as E
from p2 import MIDNIGHT
from PIL import Image, ImageDraw, ImageFont
import paths
T = paths.TMP + '/'
want = sys.argv[1:]
spec = [s for s in E.SPEC if (not want and "fit1" not in s[0] and "fit3" not in s[0] and not s[0].endswith("_L")) or s[0] in want]
h = 520; cols = 5; rows = (len(spec) + cols - 1) // cols
sheet = Image.new('RGB', (h * cols, h * rows), "#F4F2EE"); d = ImageDraw.Draw(sheet)
f = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22, index=1)
for i, (name, fn, rot, qty, ko, pick, infill) in enumerate(spec):
    m = E.oriented(fn, rot); bb = m.bounding_box()
    size = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
    az = float(os.environ.get("AZ", -55)); el = float(os.environ.get("EL", 32))
    render.render({"p": (m, "X", MIDNIGHT if os.environ.get("COL") is None else os.environ["COL"], True)}, T + 'p_%s.png' % name, cam_az=az, cam_el=el, size=(h, h),
                  dist=size * 3.1 + 60, focus=(0, 0, (bb[5] - bb[2]) / 2), ground=False)
    sheet.paste(Image.open(T + 'p_%s.png' % name), ((i % cols) * h, (i // cols) * h))
    d.text(((i % cols) * h + 10, (i // cols) * h + 6), "%s  %s" % (name, ko), font=f, fill="#22262B")
out = T + (os.environ.get("NAME", "parts_sheet") + '.png'); sheet.save(out); print(out)
