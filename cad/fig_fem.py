"""회전 막대의 유한요소 해석 그림 (18_회전_막대_해석.png).   xvfb-run -a python3 fig_fem.py      (먼저 fem_rotor.py)
fem/ 폴더에 남은 그물(rotor_ccx.inp)과 풀이(rotor_ccx.frd)를 읽어, 판 아래끝을 5N 으로 밀 때의 응력을 겉면에 색으로 칠한다."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import paths
import fem_rotor as FR

W = FR.W; OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
RAMP = ["#f1ede6", "#f6cfae", "#f0a06a", "#e06d2c", "#b3410f", "#7a2706"]          # 한 색(주황)의 옅음 -> 짙음
CMAP = LinearSegmentedColormap.from_list("stress", RAMP)


def read_inp(path):
    """ccx 입력 파일에서 절점과 사면체(모서리 4개만)를 읽는다"""
    nodes = {}; tets = []; mode = None
    for line in open(path):
        if line.startswith("*"):
            mode = "N" if line.startswith("*NODE,") else ("E" if line.startswith("*ELEMENT,") else None); continue
        if mode == "N":
            a = line.split(","); nodes[int(a[0])] = [float(a[1]), float(a[2]), float(a[3])]
        elif mode == "E":
            a = line.split(","); tets.append([int(a[1]), int(a[2]), int(a[3]), int(a[4])])
    return nodes, np.array(tets)


R = json.load(open(os.path.join(paths.RES, "fem_rotor.json"), encoding="utf-8")); name = "판 아래끝을 머리 쪽으로 5N"; C = R["cases"][name]
nodes, tets = read_inp(os.path.join(W, "rotor_ccx.inp")); disp, stress = FR.read_frd(os.path.join(W, "rotor_ccx.frd"), 1)
ids = np.array(sorted(nodes)); idx = {k: i for i, k in enumerate(ids)}; X = np.array([nodes[k] for k in ids])
S = np.array([stress[0].get(k, [0.0] * 6) for k in ids]); V = FR.vm(S)
cells = np.column_stack([np.full(len(tets), 4), np.vectorize(idx.get)(tets)]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA), X); grid["vm"] = V
surf = grid.extract_surface().clean()
top = float(np.ceil(C["vm_max"]))


def view(fn, pos, focus, up, size, half):
    p = pv.Plotter(off_screen=True, window_size=size); p.set_background(BG)
    p.add_mesh(surf, scalars="vm", cmap=CMAP, clim=(0.0, top), show_scalar_bar=False, smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.05)
    p.camera.position = pos; p.camera.focal_point = focus; p.camera.up = up; p.camera.parallel_projection = True; p.camera.parallel_scale = half          # 그림 높이의 절반이 몇 mm 인가
    p.screenshot(TMP + fn); p.close()
    return Image.open(TMP + fn).convert("RGB")


a = view("fem_a.png", (60, -330, 250), (2, 0, 16), (0, 0, 1), (1500, 760), 55.0)           # 위·앞에서: 막대 전체
b = view("fem_b.png", (250, -200, 120), (62, 0, 22), (0, 0, 1), (900, 760), 42.0)          # 판 받이 쪽을 가까이
b = b.resize((int(b.size[0] * 1.0), b.size[1]))
Wd, Hd = 2360, 1320; m = 40
im = Image.new("RGB", (Wd, Hd), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "회전 막대를 유한요소법으로 풀어 본 것: 판 아래끝을 5N 으로 밀 때", font=font(50, True), fill=INK)
d.text((m, 100), "그물은 fTetWild, 풀이는 CalculiX. 사면체 %s개, 절점 %s개 (크기 %.1fmm). 재료는 PLA, 모터 축 구멍을 붙잡고 알루미늄 판에 건 힘을 볼트 4개 자리로 옮겨 걸었다." % (
    format(R["mesh"]["elements"], ","), format(R["mesh"]["nodes"], ","), R["mesh"]["size_mm"]), font=font(28), fill=SUB)
im.paste(a, (m, 180)); im.paste(b.crop((60, 0, 840, 760)), (m + 1500 + 20, 180))
d.text((m + 20, 190), "막대 전체 (가운데가 모터 축, 오른쪽 끝이 판 받이, 왼쪽 끝이 동전 통)", font=font(26, True), fill=INK)
d.text((m + 1500 + 40, 190), "판 받이 쪽", font=font(26, True), fill=INK)
# 색 띠
x0, y0, w, h = m + 20, 980, 900, 30
for i in range(w):
    d.line([(x0 + i, y0), (x0 + i, y0 + h)], fill=tuple(int(255 * c) for c in CMAP(i / (w - 1))[:3]))
d.rectangle([x0, y0, x0 + w, y0 + h], outline=LINE, width=2)
for t in range(0, int(top) + 1, 3):
    xx = x0 + w * t / top; d.line([(xx, y0 + h), (xx, y0 + h + 10)], fill=SUB, width=2); d.text((xx, y0 + h + 14), "%d" % t, font=font(24), fill=SUB, anchor="ma")
d.text((x0 + w + 24, y0 + h / 2), "응력 (폰 미세스, MPa)", font=font(26), fill=INK, anchor="lm")
rows = [("가장 큰 응력", "%.1f MPa" % C["vm_max"], "팔이 판 받이와 만나는 곳. 재료가 버티는 값 40 의 %.1f분의 1" % C["sf_vm"]),
        ("층 사이를 벌리는 응력", "%.1f MPa" % C["szz_max"], "출력할 때 층이 쌓이는 방향. 버티는 값 15 의 %.1f분의 1" % C["sf_layer"]),
        ("판 아래끝이 밀리는 양", "%.1f mm" % abs(C["plate_end_mm"]), "부러지지는 않지만 팔이 휜다. 판을 잡고 들지 않는다")]
for i, (k, v, note) in enumerate(rows):
    y = 1090 + i * 62
    d.text((m + 20, y), k, font=font(30, True), fill=INK); d.text((m + 420, y), v, font=font(30, True), fill="#b3410f"); d.text((m + 640, y), note, font=font(27), fill=SUB)
others = [(k, v) for k, v in R["cases"].items() if k != name]
d.text((m + 1500 + 40, 1090), "다른 경우의 여유", font=font(30, True), fill=INK)
for i, (k, v) in enumerate(others):
    d.text((m + 1500 + 40, 1090 + 46 * (i + 1)), "%s: %.1f배" % (k, min(v["sf_vm"], v["sf_layer"])), font=font(25), fill=SUB)
im.save(OUT + "18_회전_막대_해석.png"); print("saved", OUT + "18_회전_막대_해석.png", "max", round(float(V.max()), 2), "at", X[int(np.argmax(V))].round(1))
