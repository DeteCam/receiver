"""DeteCam receiver assembly animation (design values, simulation drawing - not a photo of a real unit).
Usage: python3 make_video.py test          -> a few PNG frames in out/test_*.png
       python3 make_video.py scene <n>     -> out/scene_<n>.mp4
       python3 make_video.py all           -> all scenes, joined into out/assembly.mp4"""
import os
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
from r3d import Renderer, Mesh, orbit, transform  # noqa: E402

W, H, FPS, SS = 1280, 720, 30, 2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc"


def font(size, weight="Medium"):
    return ImageFont.truetype(FONT % weight, size, index=1)      # index 1 = Korean


def ease(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


# ------------------------------------------------------------------ parts and when they come in
# (name, mesh builder, scene, start [s] inside the scene, group)   group: "head" nods, "base" stays
PARTS = []


def add(name, mesh, scene, t0, group="head", drop=(0.0, 0.0, 90.0)):
    V, F, C, E = mesh.arrays()
    PARTS.append(dict(name=name, V=V, F=F, C=C, E=E, scene=scene, t0=t0, group=group, drop=np.asarray(drop, dtype=float)))


add("tray", M.box_tray(), 1, -1.0)
add("carrier", M.carrier(), 1, 0.3)
add("ax210", M.ax210(), 1, 1.0)
add("cm4", M.cm4(), 1, 1.7)
add("heatsink", M.heatsink(), 1, 2.4)
add("wilkinsons", M.wilkinsons(), 2, 0.2)
add("att2", M.attenuator2(), 2, 0.7)
add("pico", M.pico_buck(), 2, 1.1)
add("sp16t", M.sp16t(), 2, 1.8)
for i in range(16):
    p, d = M.sp16t_ports()[i]
    add(f"adapter{i}", M.adapter(i), 2, 2.7 + 0.07 * i, drop=(40.0 * d[0], 40.0 * d[1], 0.0))
add("att1", M.attenuator1(), 2, 3.9, drop=(40.0, 0.0, 0.0))
add("walls", M.box_front_walls(), 3, 0.2, drop=(0.0, 0.0, 70.0))
add("lid", M.box_lid(), 3, 0.8)
add("column", M.column(), 3, 1.5, drop=(0.0, 0.0, 110.0))
add("post", M.cable_post(), 3, 1.9, drop=(0.0, 0.0, 110.0))
add("plate", M.plate(), 4, 0.2)
for k in list(range(16)) + [16]:
    add(f"fin{k}", M.fin(k), 4, 1.0 + 0.10 * (k if k < 16 else 17), drop=(0.0, 0.0, 60.0))
    add(f"film{k}", M.film(k), 4, 1.5 + 0.10 * (k if k < 16 else 17), drop=(0.0, 0.0, 60.0))
add("base", M.base(), 6, 0.2, group="base", drop=(0.0, 0.0, -80.0))
add("motor", M.motor(), 6, 1.0, group="base", drop=(0.0, 90.0, 0.0))
add("bank", M.power_bank(), 6, 1.8, group="base", drop=(0.0, -90.0, 0.0))

# cables: (polyline, scene, start, duration, colour, width)
CABLES = []
for k in range(16):
    CABLES.append((M.rib(k), 5, 0.3 + 0.12 * k, 0.5, M.CABLE, 2.0))
CABLES.append((M.ref_cable(), 5, 2.4, 0.4, M.CABLE, 2.0))
CABLES.append((M.spine(), 5, 2.9, 0.9, M.BUNDLE, 4.0))
CABLES.append((M.trunk(), 5, 3.9, 0.9, M.BUNDLE, 5.0))

# scene -> (duration [s], caption, camera (target, dist, az, el))
SCENES = {
    1: (3.6, "1. 호스트: CM4 캐리어 보드에 AX210(무선랜 카드)과 CM4 · 방열판을 꽂는다", ((M.DX, -10, -128), 500, -62, 40)),
    2: (5.2, "2. 무선 부품: 윌킨슨 분배기 2개 · 6 dB 감쇠기 2개 · SP16T 스위치, 포트마다 SMA–U.FL 어댑터", ((M.DX, 0, -118), 520, -58, 44)),
    3: (3.4, "3. 상자를 닫고 고리를 받칠 기둥(플라스틱)과 뒤쪽 케이블 기둥을 세운다", ((M.DX / 2, 0, -80), 660, -52, 26)),
    4: (4.6, "4. 고리 판(지름 12 cm): 필름 안테나 16개를 날개에 세로로, 기준 안테나 1개는 고리 면 가운데", ((0, 0, -30), 520, -58, 40)),
    5: (5.6, ("5. 케이블은 고리 면 안에서만 모아(등뼈 배선) 고리 안쪽 뒤에서 한 묶음으로 상자에 내린다" if M.LAYOUT == "k2i" else "5. 케이블은 고리 면 안에서만 모아(등뼈 배선) 고리 바로 뒤에서 한 묶음으로 상자에 내린다"), ((-8, 0, -20), 490, -74, 52)),
    6: (3.6, "6. 받침판: 끄덕이는 축, 웜기어 스텝 모터, 보조배터리(따로 둠)", ((M.DX / 2, 20, -100), 900, -50, 22)),
    7: (5.4, "7. 자리마다 −20° · 0° · +20° 세 자세로 잰다 (재는 동안 모터는 끈다)", ((M.DX / 2, 20, -95), 900, -86, 10)),
    8: (4.6, "완성: 가로 · 세로 20 cm 내외, 수신기 1기를 여러 자리로 옮겨 가며 잰다", ((M.DX / 2, 20, -95), 950, -40, 20)),
}
ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
T_START = {}
_t = 0.0
for _s in ORDER:
    T_START[_s] = _t
    _t += SCENES[_s][0]
T_TOTAL = _t


def nod_angle(scene, ts):
    if scene != 7:
        return 0.0
    keys = [(0.0, 0.0), (0.6, 0.0), (1.5, 20.0), (2.3, 20.0), (3.5, -20.0), (4.3, -20.0), (5.1, 0.0), (9.0, 0.0)]
    for (t0, a0), (t1, a1) in zip(keys[:-1], keys[1:]):
        if ts <= t1:
            u = (ts - t0) / max(1e-9, t1 - t0)
            u = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))
            return a0 + (a1 - a0) * u
    return 0.0


def camera(scene, ts):
    dur, cap, (tg, dist, az, el) = SCENES[scene]
    i = ORDER.index(scene)
    if scene == 8:
        az = az + 40.0 * (ts / dur)
    elif scene == 4:
        az = az + 10.0 * (ts / dur)
    elif scene == 5 and ts > 3.7:          # swing round to the back to show the bundle going down
        u = ease((ts - 3.7) / 1.3)
        tg = tuple(np.asarray(tg) + u * (np.array([-30.0, 0.0, -50.0]) - np.asarray(tg)))
        dist, az, el = dist + u * (600 - dist), az + u * (-150.0 - az), el + u * (20.0 - el)
    if i > 0 and ts < 0.9:
        _, _, (tg0, d0, az0, el0) = SCENES[ORDER[i - 1]]
        if ORDER[i - 1] == 4:
            az0 += 10.0
        if ORDER[i - 1] == 5:
            tg0, d0, az0, el0 = (-30.0, 0.0, -50.0), 600, -150.0, 20.0
        u = ease(ts / 0.9)
        tg = tuple(np.asarray(tg0) + u * (np.asarray(tg) - np.asarray(tg0)))
        dist, az, el = d0 + u * (dist - d0), az0 + u * (az - az0), el0 + u * (el - el0)
    return orbit(tg, dist, az, el, fov_deg=24.0)


def frame(rend, scene, ts, caption=True):
    cam = camera(scene, ts)
    rend.begin(cam)
    ang = nod_angle(scene, ts)
    idx = ORDER.index(scene)
    Vs, Fs, Cs, Es, n = [], [], [], [], 0
    for p in PARTS:
        pi = ORDER.index(p["scene"])
        if pi > idx:
            continue
        if pi == idx:
            if ts < p["t0"]:
                continue
            u = ease((ts - p["t0"]) / 0.55)
            sh = (1.0 - u) * p["drop"]
        else:
            sh = np.zeros(3)
        V = p["V"] + sh
        if p["group"] == "head" and ang:
            V = transform(V, rot_y_deg=ang, pivot=M.PIVOT)
        Vs.append(V)
        Fs.append(p["F"] + n)
        Cs.append(p["C"])
        if len(p["E"]):
            Es.append(p["E"] + n)
        n += len(V)
    if Vs:
        rend.draw(np.vstack(Vs), np.vstack(Fs), np.vstack(Cs), np.vstack(Es) if Es else None)
    for pts, sc, t0, d, col, wd in CABLES:
        ci = ORDER.index(sc)
        if ci > idx or (ci == idx and ts < t0):
            continue
        f = 1.0 if ci < idx else ease((ts - t0) / d)
        a, b = pts[0], pts[0] + f * (pts[1] - pts[0])
        if ang:
            a, b = transform(np.array([a, b]), rot_y_deg=ang, pivot=M.PIVOT)
        rend.lines(a[None], b[None], col, width=wd, bias=2.5)
    img = rend.image()
    if caption:
        overlay(img, rend, scene, ts, ang)
    return img


def label(d, xy, text, anchor_xy=None, size=22):
    f = font(size)
    x, y = xy
    tw = d.textlength(text, font=f)
    if anchor_xy is not None:
        d.line([anchor_xy, (x + (0 if anchor_xy[0] < x else tw), y + size * 0.7)], fill=(40, 44, 52), width=2)
        r = 4
        d.ellipse([anchor_xy[0] - r, anchor_xy[1] - r, anchor_xy[0] + r, anchor_xy[1] + r], fill=(200, 60, 30))
    d.rounded_rectangle([x - 8, y - 4, x + tw + 8, y + size * 1.45], radius=6, fill=(255, 255, 255), outline=(60, 66, 76), width=1)
    d.text((x, y), text, font=f, fill=(25, 28, 34))


def overlay(img, rend, scene, ts, ang):
    d = ImageDraw.Draw(img, "RGBA")
    dur, cap, _ = SCENES[scene]
    d.rectangle([0, H - 92, W, H], fill=(22, 26, 34, 225))
    f = font(27)
    d.text((36, H - 72), cap, font=f, fill=(245, 246, 248))
    d.text((36, 22), "DeteCam 수신기 조립 순서", font=font(26, "Bold"), fill=(25, 28, 34))
    d.text((W - 36 - d.textlength("시뮬레이션 그림 · 치수는 설계값(실물 조립 전)", font=font(18, "Regular")), 28),
           "시뮬레이션 그림 · 치수는 설계값(실물 조립 전)", font=font(18, "Regular"), fill=(90, 96, 108))
    # progress dots
    for i, s in enumerate(ORDER):
        x = 40 + i * 26
        d.ellipse([x, H - 22, x + 12, H - 10], fill=(240, 170, 60) if s == scene else ((150, 156, 168) if ORDER.index(s) < ORDER.index(scene) else (70, 76, 88)))

    def sp(p):
        x, y = rend.screen(np.asarray(p, dtype=float))
        return float(x[0]), float(y[0])

    if scene == 2 and ts > 2.4:
        label(d, (760, 120), "SP16T 스위치 (기판 60 × 100 mm)", sp((M.DX + M.SP_DX, 20, M.Z_TOP - 20)))
    if scene == 2 and ts > 3.6:
        label(d, (120, 150), "SMA–U.FL 어댑터 16개", sp(M.plug_point(1)))
    if scene == 2 and 0.8 < ts < 2.4:
        label(d, (120, 130), "윌킨슨 분배기 2개", sp((M.DX - 44, 42, M.Z_BOT + 16)))
        label(d, (760, 150), "Pico · 5 V 벅 · PD 트리거", sp((M.DX + 36, 44, M.Z_BOT + 8)))
    if scene == 4 and ts > 3.2:
        label(d, (800, 130), "기준 안테나 (고리 면 가운데)", sp((0, 0, 18)))
        label(d, (90, 150), "필름 안테나 16개", sp((M.ring_xy(11)[0], M.ring_xy(11)[1], 16)))
    if scene == 5 and 3.0 < ts < 3.7:
        label(d, (820, 150), "등뼈(고리 면 안 묶음)", sp((10, M.YS, 0)))
    if scene == 5 and ts > 4.9:
        label(d, (120, 430), ("고리 안쪽 뒤에서 한 묶음으로 내림" if M.LAYOUT == "k2i" else "고리 바로 뒤에서 한 묶음으로 내림"), sp((M.G[0], M.YS, -45)))
    if scene == 6 and ts > 1.6:
        label(d, (860, 200), "웜기어 스텝 모터", sp((M.DX + 30, 112, M.PIVOT[2] + 20)))
    if scene == 6 and ts > 2.4:
        label(d, (110, 470), "보조배터리 (크기에 넣지 않음)", sp((M.DX, -150, -161)))
    if scene == 7:
        label(d, (W // 2 - 90, 96), f"끄덕임 {ang:+.0f}°", None, size=30)
    if scene == 8 and ts > 0.8:
        label(d, (860, 150), "고리 지름 12 cm", sp((0, 60, 18)))
        label(d, (860, 330), "상자 윗면은 고리 면에서 9 cm 아래", sp((M.BOX[1] - 8, 70, M.Z_TOP)))
        label(d, (90, 200), "상자 13.5 × 15.8 × 5.6 cm", sp((M.DX, -78, M.Z_TOP - 28)))


def title_card():
    img = Image.new("RGB", (W, H), (22, 26, 34))
    d = ImageDraw.Draw(img)
    d.text((80, 190), "DeteCam", font=font(84, "Bold"), fill=(245, 246, 248))
    d.text((84, 310), "숨은 Wi‑Fi 카메라의 방향 · 위치를 찾는 수신기", font=font(38), fill=(240, 170, 60))
    d.text((84, 392), "16소자 스위치 고리 수신기 1기 · 조립 순서 (시뮬레이션)", font=font(30, "Regular"), fill=(200, 205, 214))
    d.text((84, 560), "이 영상은 설계값으로 그린 그림이다. 실물 사진이 아니며, 부품은 아직 사지 않았다.", font=font(22, "Regular"), fill=(150, 156, 168))
    d.text((84, 596), "한국공학대학교 컴퓨터공학 졸업작품 · 2026-11", font=font(22, "Regular"), fill=(150, 156, 168))
    return img


def encode_scene(scene, rend):
    os.makedirs(OUT, exist_ok=True)
    dur = SCENES[scene][0]
    n = int(round(dur * FPS))
    tmp = os.path.join(OUT, f"tmp_scene_{scene}.mp4")
    dst = os.path.join(OUT, f"scene_{scene}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        pr.stdin.write(frame(rend, scene, i / FPS).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    os.replace(tmp, dst)
    return dst


def encode_title():
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "title.png")
    title_card().save(png)
    tmp = os.path.join(OUT, "tmp_scene_0.mp4")
    dst = os.path.join(OUT, "scene_0.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", png, "-vf", "format=yuv420p,loop=-1:1,trim=duration=2.6",
                           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS), tmp])
    os.replace(tmp, dst)
    return dst


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    rend = Renderer(W, H, ss=SS)
    os.makedirs(OUT, exist_ok=True)
    if mode == "test":
        import time
        which = [tuple(map(float, a.split(":"))) for a in sys.argv[2:]] or [(1, 3.4), (2, 5.0), (3, 3.2), (4, 4.4), (5, 5.4), (6, 3.4), (7, 1.9), (8, 2.5)]
        for scene, ts in which:
            scene = int(scene)
            t0 = time.time()
            frame(rend, scene, ts).save(os.path.join(OUT, f"test_{scene}.png"))
            print(scene, ts, f"{time.time() - t0:.2f}s", flush=True)
        title_card().save(os.path.join(OUT, "test_0.png"))
    elif mode == "scene":
        print(encode_scene(int(sys.argv[2]), rend))
    else:
        files = [encode_title()] + [encode_scene(s, rend) for s in ORDER]
        lst = os.path.join(OUT, "list.txt")
        with open(lst, "w") as f:
            for p in files:
                f.write(f"file '{p}'\n")
        tmp = os.path.join(OUT, "tmp_assembly.mp4")
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp])
        os.replace(tmp, os.path.join(OUT, "assembly.mp4"))
        print("done", T_TOTAL + 2.6)


if __name__ == "__main__":
    main()
