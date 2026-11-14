"""AR 마커: 폰이 수신기의 자리와 방향을 알아내는 데 쓰는 종이 띠를 만든다.
    python3 marker.py        ->  마커 폴더(저장소에서는 ../마커)에 AR_마커.pdf (실제 크기로 인쇄), AR_마커.png, marker.json
                                 marker.json 은 수신기 코드의 firmware/pi/marker.json 과 같아야 한다 (바뀌면 그쪽에도 복사한다)

왜 필요한가: 수신기는 '수신기에서 본' 방향과 구간을 계산한다. 폰이 그것을 방 안에 그리려면 수신기가 방 어디에, 어느 쪽을 보고 놓였는지 알아야 한다.
어떻게: 머리 등판(뒷벽) 아래쪽에 ArUco 마커 2개가 그려진 종이 띠(108 x 44mm)를 대고, 출력한 마커 틀로 덮어 M3 나사 4개로 조인다 (풀·테이프를 쓰지 않는다). 폰 카메라가 마커의 네 귀를 찾아 자리와 방향을 계산한다 (OpenCV solvePnP).
        마커는 머리에 붙어 있으므로, 수신기가 그때의 관절 각도로 '받침에서 본 마커 귀의 좌표'를 계산해 폰에 준다 (GET /api/marker). 폰은 그것으로 받침의 자리를 바로 얻는다.
붙이는 곳: 머리를 재는 자세(54.7도)로 숙이면 등판이 위를 55도쯤 올려다본다. 서서 내려다보는 폰에 잘 보인다. 통풍 틈(등판 위쪽)은 가리지 않는다.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import cv2
import paths
from p2 import *

DICT_NAME = "DICT_4X4_50"
IDS = (0, 1)
SIZE = 34.0                       # 마커 한 변 (검은 테두리까지)
STRIP_W, STRIP_H = 108.0, 44.0    # 종이 띠
CX = 27.0                         # 마커 가운데가 머리 한가운데에서 옆으로 떨어진 거리
Z_BOTTOM = HZ0 + 1.5              # 띠 아래 변 (머리 치마 아래끝에서 1.5mm 위)
ZC = Z_BOTTOM + STRIP_H / 2       # 마커 가운데 높이 (머리 좌표)
Y_FACE = HY1 + 0.1                # 등판 겉면 + 종이 두께
# 마커 틀(parts2.marker_frame)이 종이를 덮는 곳: 양옆 6mm, 가운데 12mm, 위아래 1mm. 나사 4개는 띠의 귀 근처(옆 변에서 2.5, 위·아래 변에서 4)를 지나므로 그 자리를 오려 낸다
FRAME_SIDE, FRAME_MID, FRAME_EDGE = 6.0, 6.0, 1.0
NOTCH = (5.0, 1.6, 6.4)           # 나사 자리 오려 내기: 옆 변에서 들어가는 깊이, 위(아래) 변에서 시작, 끝


def corners_head():
    """마커 귀의 머리 좌표 (mm). 순서는 OpenCV ArUco 와 같다: 마커를 바로 보고 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래.
    등판을 뒤에서 바라보면 머리의 +X 가 왼쪽이다. 그래서 0번 마커(왼쪽)는 x = +27, 1번(오른쪽)은 x = -27."""
    out = {}; h = SIZE / 2
    for i, cx_ in zip(IDS, (CX, -CX)):
        out[i] = [[cx_ + h, Y_FACE, ZC + h], [cx_ - h, Y_FACE, ZC + h], [cx_ - h, Y_FACE, ZC - h], [cx_ + h, Y_FACE, ZC - h]]
    return out


def head_to_base(p, yaw, pitch):
    """머리 좌표의 점(mm) -> 받침 좌표(mm). 받침 좌표: 원점 = 받침 바닥 한가운데, +X = 받침 0도(앞 표시), +Z = 위. parts2.pose_head 와 같은 계산"""
    x, y, z = p; t = np.radians(pitch); a = np.radians(yaw)
    xy = -(y - Y_T) * np.cos(t) + (z - S_T) * np.sin(t); zy = H_T + (y - Y_T) * np.sin(t) + (z - S_T) * np.cos(t)
    return [xy * np.cos(a) - x * np.sin(a), xy * np.sin(a) + x * np.cos(a), zy]


def bits(i):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    img = cv2.aruco.generateImageMarker(d, i, 6)          # 6 x 6 칸 (테두리 1칸 + 4 x 4)
    return (img < 128)


def draw_png(path, dpi=600):
    s = dpi / 25.4; W, H = int(round(STRIP_W * s)), int(round(STRIP_H * s))
    im = np.full((H, W), 255, np.uint8)
    for i, cx_ in zip(IDS, (-CX, CX)):                    # 그림에서는 0번이 왼쪽
        b = bits(i); c = SIZE / 6
        for r in range(6):
            for q in range(6):
                if b[r, q]:
                    x0 = (STRIP_W / 2 + cx_ - SIZE / 2 + q * c) * s; y0 = (STRIP_H / 2 - SIZE / 2 + r * c) * s
                    im[int(round(y0)):int(round(y0 + c * s)), int(round(x0)):int(round(x0 + c * s))] = 0
    cv2.rectangle(im, (0, 0), (W - 1, H - 1), 0, 2)
    cv2.imwrite(path, im)
    return im


def draw_pdf(path, dpi=600):
    """A4 한 장: 인쇄 안내 + 50mm 자 + 띠 2개. 글자가 깨지지 않게 그림으로 그려 PDF 로 저장한다 (600dpi, 실제 크기)."""
    from PIL import Image, ImageDraw, ImageFont
    s = dpi / 25.4; PW, PH = int(round(210 * s)), int(round(297 * s))
    im = Image.new("L", (PW, PH), 255); d = ImageDraw.Draw(im)
    F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; FB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font = lambda pt, bold=False: ImageFont.truetype(FB if bold else F, int(round(pt / 72.0 * dpi)), index=1)
    X = lambda mm_: int(round(mm_ * s))

    def strip(x0, y0):                                    # x0, y0 = 띠의 왼쪽 위 (mm)
        n = 60
        for k in range(n):                                # 자르는 선 (점선)
            for (a, b, c, e) in ((x0 + STRIP_W * k / n, y0, x0 + STRIP_W * (k + 0.5) / n, y0), (x0 + STRIP_W * k / n, y0 + STRIP_H, x0 + STRIP_W * (k + 0.5) / n, y0 + STRIP_H)):
                d.line([(X(a), X(b)), (X(c), X(e))], fill=90, width=max(2, X(0.12)))
        n = 24
        for k in range(n):
            for (a, b, c, e) in ((x0, y0 + STRIP_H * k / n, x0, y0 + STRIP_H * (k + 0.5) / n), (x0 + STRIP_W, y0 + STRIP_H * k / n, x0 + STRIP_W, y0 + STRIP_H * (k + 0.5) / n)):
                d.line([(X(a), X(b)), (X(c), X(e))], fill=90, width=max(2, X(0.12)))
        for i, cx_ in zip(IDS, (-CX, CX)):
            b = bits(i); cell = SIZE / 6
            for r in range(6):
                for q in range(6):
                    if b[r, q]:
                        d.rectangle([X(x0 + STRIP_W / 2 + cx_ - SIZE / 2 + q * cell), X(y0 + STRIP_H / 2 - SIZE / 2 + r * cell),
                                     X(x0 + STRIP_W / 2 + cx_ - SIZE / 2 + (q + 1) * cell) - 1, X(y0 + STRIP_H / 2 - SIZE / 2 + (r + 1) * cell) - 1], fill=0)
        nd, n0, n1 = NOTCH                                # 나사 자리 4곳 (오려 낸다): 빗금 친 네모
        for sx_ in (0, 1):
            for sy_ in (0, 1):
                xa = x0 + (STRIP_W - nd if sx_ else 0.0); ya = y0 + (STRIP_H - n1 if sy_ else n0)
                d.rectangle([X(xa), X(ya), X(xa + nd), X(ya + n1 - n0)], outline=0, width=max(2, X(0.12)))
                for k in range(1, 6):
                    d.line([(X(xa), X(ya + (n1 - n0) * k / 6)), (X(xa + nd), X(ya + (n1 - n0) * k / 6))], fill=120, width=max(1, X(0.08)))
        cxm = X(x0 + STRIP_W / 2)
        d.polygon([(cxm, X(y0 + 14)), (cxm - X(2.2), X(y0 + 18.5)), (cxm + X(2.2), X(y0 + 18.5))], fill=0)      # 위쪽 화살표
        d.text((cxm, X(y0 + 22.5)), "위", font=font(9, True), fill=0, anchor="mm")
        d.text((cxm, X(y0 + 29.5)), "DeteCam", font=font(5), fill=60, anchor="mm")

    d.text((X(20), X(16)), "DeteCam AR 마커 (머리 등판에 대고 마커 틀로 조이는 띠)", font=font(17, True), fill=0)
    lines = ["1. 이 종이를 ‘실제 크기(100%)’로 인쇄한다. ‘용지에 맞춤’이 켜져 있으면 크기가 달라져 위치가 틀어진다. 두꺼운 종이면 더 좋다.",
             "2. 아래 50mm 자를 자로 재서 정확히 50mm 인지 확인한다. 띠는 가로 108mm, 세로 44mm, 검은 네모는 한 변 34mm 다.",
             "3. 점선을 따라 띠 하나를 오린다 (두 개는 같은 것이다. 하나는 예비). 네 귀의 빗금 친 네모(나사 자리)도 오려 낸다.",
             "4. 머리 등판(파이 뒤쪽의 평평한 면)의 아래쪽에 댄다. 띠의 아래 변을 등판 아래 모서리에서 1.5mm 위에,",
             "   양옆은 2mm 씩 남게 가운데에 맞춘다. ▲ 표시가 위(판 모터 쪽)를 보게 한다. 통풍 틈은 가리지 않는다.",
             "5. 그 위에 마커 틀(15_marker_frame)을 덮고 M3 x 6 나사 4개로 조인다. 풀이나 테이프는 쓰지 않는다. 번쩍이는 것으로 덮지 않는다."]
    for k, t in enumerate(lines):
        d.text((X(20), X(29 + 6.2 * k)), t, font=font(9.5), fill=0)
    y = 74.0
    d.line([(X(20), X(y)), (X(70), X(y))], fill=0, width=max(2, X(0.2)))
    for xx in (20, 70):
        d.line([(X(xx), X(y - 1.8)), (X(xx), X(y + 1.8))], fill=0, width=max(2, X(0.2)))
    d.text((X(73), X(y)), "50mm", font=font(9), fill=0, anchor="lm")
    strip(20, 88); strip(20, 148)
    d.text((X(20), X(199)), "ArUco %s, 번호 0 (왼쪽), 1 (오른쪽). 귀의 좌표는 marker.json 에 있다. 수신기가 GET /api/marker 로 알려 준다." % DICT_NAME, font=font(8.5), fill=60)
    im.save(path, "PDF", resolution=dpi)


def with_frame(im, dark=60):
    """마커 틀을 덮은 모습: 틀(어두운 파랑)이 종이를 가리는 곳을 어둡게 칠한다. 틀의 창으로는 마커와 그 둘레 흰 여백 4mm 가 보인다."""
    im = im.copy(); h, w = im.shape; sx = w / STRIP_W; sy = h / STRIP_H
    X = lambda v: int(round(v * sx)); Y = lambda v: int(round(v * sy))
    im[:, :X(FRAME_SIDE)] = dark; im[:, X(STRIP_W - FRAME_SIDE):] = dark
    im[:, X(STRIP_W / 2 - FRAME_MID):X(STRIP_W / 2 + FRAME_MID)] = dark
    im[:Y(FRAME_EDGE), :] = dark; im[Y(STRIP_H - FRAME_EDGE):, :] = dark
    return im


def selftest(png):
    """그린 그림을 OpenCV 로 다시 읽어 마커 2개가 잡히는지, 귀 순서가 좌표와 맞는지 확인한다. 비스듬히 본 그림으로도 해 본다.
    마커 틀을 덮은 모습(둘레가 어두운 틀에 4mm 까지 가려진다)으로 본다."""
    im = with_frame(cv2.imread(png, cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(png.replace(".png", "_틀을_덮은_모습.png"), im)
    pad = cv2.copyMakeBorder(im, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=60)       # 둘레는 어두운 파랑(출력물)이라고 본다
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME)), cv2.aruco.DetectorParameters())
    out = {}
    for name, img in (("정면", pad), ("작게(멀리서)", cv2.resize(pad, None, fx=0.12, fy=0.12, interpolation=cv2.INTER_AREA))):
        cs, ids, _ = det.detectMarkers(img)
        out[name] = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    # 55도 비스듬히 본 모습
    h, w = pad.shape; src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); k = 0.45
    dst = np.float32([[w * 0.18, h * (1 - k) / 2], [w * 0.82, h * (1 - k) / 2], [w, h * (1 + k) / 2], [0, h * (1 + k) / 2]])
    warp = cv2.warpPerspective(pad, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=60)
    warp = cv2.resize(warp, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    cs, ids, _ = det.detectMarkers(warp); out["비스듬히"] = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    # 귀 순서: 정면 그림에서 0번 마커의 첫 귀(왼쪽 위)가 그림의 왼쪽 위인지
    cs, ids, _ = det.detectMarkers(pad); order_ok = False
    if ids is not None:
        for c_, i in zip(cs, ids.ravel()):
            if int(i) == 0:
                p = c_[0]; order_ok = bool(p[0][0] < p[1][0] and p[0][1] < p[3][1] and p[0][0] < pad.shape[1] / 2)
    out["귀 순서"] = order_ok
    return out


if __name__ == "__main__":
    out = paths.MARKER
    png = os.path.join(out, "AR_마커.png"); draw_png(png); draw_pdf(os.path.join(out, "AR_마커.pdf"))
    ch = corners_head()
    J = dict(dictionary=DICT_NAME, ids=list(IDS), size_mm=SIZE, strip_mm=[STRIP_W, STRIP_H],
             where="머리 등판(뒷벽) 아래쪽. 띠 아래 변이 등판 아래 모서리에서 1.5mm 위, 가운데 맞춤. 마커 틀로 덮어 나사 4개로 조인다",
             head_frame="원점 = 파이 안테나, X = 끄덕임 축 방향(등판을 뒤에서 보면 왼쪽이 +), Y = 등판 쪽이 +, Z = 위(판 회전축)",
             corners_head_mm={str(k): v for k, v in ch.items()}, corner_order="왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 (마커를 바로 볼 때. OpenCV ArUco 와 같다)",
             kinematics=dict(Y_T=Y_T, S_T=S_T, H_T=H_T, formula="받침 좌표 = Rz(yaw) · [ -(y - Y_T)cos p + (z - S_T)sin p,  x,  H_T + (y - Y_T)sin p + (z - S_T)cos p ]  (p = 끄덕임 각)"),
             register_pose=dict(pitch=TILT3, spin=0.0, note="자리 맞출 때의 자세: 재는 자세(54.7도)에 판을 옆(0도)으로. 등판이 위를 55도쯤 올려다본다"),
             corners_base_mm_at_register={str(k): [[round(float(u), 2) for u in head_to_base(p, 0.0, TILT3)] for p in v] for k, v in ch.items()},
             api="GET /api/marker -> 지금 관절 각도에서의 귀 좌표 (받침 좌표, m)")
    json.dump(J, open(os.path.join(out, "marker.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    t = selftest(png)
    print("마커 띠 %g x %g mm, 마커 %g mm, 가운데 높이 z = %.1f (머리 좌표)" % (STRIP_W, STRIP_H, SIZE, ZC))
    print("다시 읽기:", t)
    c0 = np.array(J["corners_base_mm_at_register"]["0"]); n = np.cross(c0[3] - c0[0], c0[1] - c0[0]); n = n / np.linalg.norm(n)      # 바깥쪽(종이가 보이는 쪽)을 향하는 방향
    print("재는 자세에서 0번 마커: 가운데 높이 %.0f mm, 면이 향한 쪽 (받침 좌표) = (%.2f, %.2f, %.2f) -> 올려다보는 각 %.1f도" % (c0[:, 2].mean(), n[0], n[1], n[2], np.degrees(np.arcsin(abs(n[2])))))
    json.dump(t, open(os.path.join(paths.RES, "marker_selftest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
