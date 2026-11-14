"""조립 시뮬레이션 영상: 조립 안내의 순서대로 부품을 하나씩 제자리에 넣고, 나사 55개를 하나하나 드라이버로 돌려 조인다.
    xvfb-run -a python3 anim_assembly.py plan                 # 장면 목록과 길이 -> 결과 폴더의 anim_assembly.json (장의 시작 시각)
    xvfb-run -a python3 anim_assembly.py still 12.5 30        # 그 시각(초)의 화면을 그림으로 (확인용, 임시 폴더)
    xvfb-run -a python3 anim_assembly.py render K N           # 영상을 N 토막으로 나눠 K 번째 토막을 그린다 (토막마다 따로 돌릴 수 있다)
    python3 anim_assembly.py join N                           # 토막을 이어 영상 폴더의 조립_시뮬레이션.mp4 (H.264)
    python3 anim_assembly.py gif                              # 영상에서 판을 넣고 누름쇠를 조이는 장면을 잘라 그림 35 (GIF)
부품이 들어오는 방향은 audit_assembly.py 가 걸리는 곳이 없다고 확인한 방향이고, 나사의 자리·방향·길이는 audit_screws.py 의 목록(S) 그대로다.
나사는 드라이버가 다가와 시계 방향으로 돌리며 들어간다 (오른나사). 너트는 나사보다 먼저 육각 자리에 넣는다."""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import parts2 as P
import paths
from cadlib import *
from p2 import *

W, H, FPS = 1280, 720, 24
OUTV = os.path.join(paths.VID, "조립_시뮬레이션.mp4")
TMP = os.path.join(paths.TMP, "asm") + "/"
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, BG, HOT, GOLD = "#22262B", "#5B6470", "#F4F2EE", "#EB6834", "#E3A23B"
STEEL = "#C3C8CF"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)


# ------------------------------------------------------------------ 행렬
def Tr(v):
    T = np.eye(4); T[:3, 3] = v; return T


def Rax(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a); T = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    T[i, i] = c; T[i, j] = -s; T[j, i] = s; T[j, j] = c
    return T


def about(p, R):
    return Tr(p) @ R @ Tr(-np.asarray(p, float))


def frame(p, d):
    """나사 좌표 (Z = 들어가는 방향 d, 원점 = p) -> 세계"""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    x = np.cross([0, 0, 1.0], d)
    x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-9 else x / np.linalg.norm(x)
    y = np.cross(d, x); T = np.eye(4); T[:3, 0] = x; T[:3, 1] = y; T[:3, 2] = d; T[:3, 3] = p
    return T


W0 = Tr([0, 0, H_T]) @ Rax("z", 90) @ Tr([0, -Y_T, -S_T])            # 머리 좌표 -> 세계 (받침 0, 끄덕임 0, 판 0). parts2.pose_head 와 같다
W0i = np.linalg.inv(W0)
hv = lambda v: W0[:3, :3] @ np.asarray(v, float)                       # 머리 좌표의 방향 -> 세계
NodT = lambda p: about([0, 0, H_T], Rax("y", p))
SpinT = lambda s: W0 @ Rax("z", s) @ W0i
ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def seg(u, a, b):
    """u 가 a~b 사이에서 0 -> 1 (부드럽게)"""
    return ease((u - a) / (b - a)) if b > a else float(u >= b)


# ------------------------------------------------------------------ 부품
def build_parts():
    """이름 -> (형상(세계, 조립된 자리), 무리, 색). 동전은 한 닢씩, 너트 7개, 나사 55개는 따로 만든다"""
    A = P.assembly(0.0, 0.0, 0.0)
    out = {}
    for k, (m, g, c, pr) in A.items():
        if k in ("coins", "rotor_hw"):
            continue
        out[k] = (m, g, c)
    for k in range(N_COIN):                                                        # 동전: 통의 막힌 쪽(위)부터
        out["coin%02d" % k] = (P.pose_head(cz(COIN_D / 2, POD_Z1 - COIN_T * (k + 1) + 0.03, POD_Z1 - COIN_T * k - 0.03, POD_XC, 0), 0, 0, 0.0), "R", "#B9B3A6")
    xf = FL_XH - FL_T
    for i, (y, z) in enumerate(PLATE_BOLTS):                                       # 판 받이 안쪽의 너트 4개
        out["nut_fl%d" % i] = (P.pose_head(prism_xy_to_yz(P.hex_cs(5.5), xf - 0.2, xf + 2.2).translate([0, y, z]) - cx(1.5, xf - 1, xf + 3, y, z), 0, 0, 0.0), "R", STEEL)
    for i, (y, z) in enumerate(P.clamp_screws()):                                  # 판 틀 뒷면 위쪽의 너트 2개
        out["nut_cl%d" % i] = (P.pose_head(prism_xy_to_yz(P.hex_cs(5.5), FL_XH + NUT_POCKET - 2.4, FL_XH + NUT_POCKET).translate([0, y, z]) - cx(1.5, FL_XH - 2, FL_XH + 3, y, z), 0, 0, 0.0), "R", STEEL)
    zs = M1_ZT + 7.0                                                               # 축 고정 나사의 너트 (윗면 홈, 꼭짓점이 바닥에)
    out["nut_sh"] = (P.pose_head(prism_xy_to_yz(P.hex_cs(5.5).rotate(90), 0.0, 2.4).rotate([0, 0, 90]).translate([0, 6.4, zs]) - cy(1.5, 3.0, 7.0, 0, zs), 0, 0, 0.0), "R", STEEL)
    return out


THREAD = {3.0: (1.5, 1.22, 0.5), 2.5: (1.25, 1.02, 0.45), 2.0: (1.0, 0.8, 0.4)}            # 바깥 반지름, 골 반지름, 피치


def screw_mesh(m, L, head, hh, hr):
    """나사 좌표 (Z = 들어가는 쪽, 원점 = 둥근머리는 머리 밑면, 접시머리는 머리 윗면)의 나사. 머리에 십자 홈"""
    ro, ri, pt = THREAD[m]
    z0 = 0.0 if head == "둥근" else 1.7 * m / 3.0
    prof = [(0.0, z0)]; z = z0; k = 0
    while z < L - 0.6:
        prof.append((ro if k % 2 == 0 else ri, z)); z += pt / 2; k += 1
    prof += [(ri * 0.9, L - 0.3), (ri * 0.5, L), (0.0, L)]
    shank = M.revolve(poly(prof), 18)
    if head == "둥근":
        hd = M.cylinder(hh * 0.7, hr, hr, 24).translate([0, 0, -hh * 0.7]) + M.cylinder(hh * 0.3, hr * 0.8, hr, 24).translate([0, 0, -hh])
        top = -hh
    else:
        hd = M.cylinder(z0, hr, ro, 24)
        top = 0.0
    w, ln, dp = 0.32 * m, 1.25 * hr, 0.45 * hh if head == "둥근" else 0.9 * m / 3.0
    cross = box(-w / 2, w / 2, -ln / 2, ln / 2, top - 1.0, top + dp) + box(-ln / 2, ln / 2, -w / 2, w / 2, top - 1.0, top + dp)
    return (hd + shank) - cross


def driver_mesh(m):
    """드라이버 (나사 좌표: 끝이 원점, 손잡이가 -Z 쪽). 축이 길고(110mm) 한 면에 노란 줄이 있어 도는 것이 보인다. 손잡이는 반투명으로 그린다"""
    rs = 2.0 if m >= 3.0 else 1.5
    tip = M.cylinder(4.0, rs * 0.45, rs, 6).translate([0, 0, -4.0])
    shaft = M.cylinder(106.0, rs, rs, 6).translate([0, 0, -110.0])
    handle = M.cylinder(90.0, 8.5, 9.5, 8).translate([0, 0, -204.0]) + M.cylinder(6.0, rs + 1.0, 8.5, 8).translate([0, 0, -116.0])
    stripe = box(rs * 0.80, rs * 0.80 + 0.5, -rs * 0.42, rs * 0.42, -60.0, -7.0) + box(9.3, 10.2, -1.8, 1.8, -196.0, -122.0)
    return (tip + shaft) - box(-0.22, 0.22, -rs, rs, -1.0, 0.1), handle, stripe


# ------------------------------------------------------------------ 나사 목록 (audit_screws.py)
def screw_list():
    import audit_screws as AS
    out = []
    for name, p, d, L, through, thread, stage, head, opt in AS.S:
        m = opt.get("m", 3.0); q = AS.SPEC[m]
        p = np.asarray(p, float); d = np.asarray(d, float); d /= np.linalg.norm(d)
        hh = q["head_h"] if head == "둥근" else 0.0
        out.append(dict(name=name, p=p, d=d, L=float(L), head=head, m=m, hh=q["head_h"], hr=q["head_r"], top=p - d * hh, nut=thread is None))
    return out


def screw_group(name):
    if name.startswith("받침 덮개"):
        return "B"
    if name.startswith(("모터 1 귀", "받침 회전", "끄덕임 멈춤", "고정 나사", "핀 막이")):
        return "Y"
    if name.startswith("모터 2 귀"):
        return "M2"
    if name.startswith("모터 3 귀"):
        return "C"
    if name.startswith(("판 틀 나사", "누름쇠 나사", "축 고정", "균형추 마개")):
        return "R"
    return "H"


def part_group(name, g):
    if name in ("motor2", "motor2_cap", "pinion2", "m2_ring"):
        return "M2"
    if name in ("crown", "motor3", "motor3_cap"):
        return "C"
    if name.startswith(("coin", "nut_")):
        return "R"
    return {"B": "B", "Y": "Y", "H": "H", "R": "R"}[g]


# ------------------------------------------------------------------ 무리(받침·요크·머리·회전부)의 자리
G0 = dict(yaw=0.0, pitch=0.0, spin=0.0, lift_Y=150.0, off_M2=np.array([0.0, 80.0, 0.0]), lift_H=230.0, lift_C=70.0, off_R=np.array([40.0, -420.0, -120.0]), flip=0.0, show_main=1.0)


def group_T(G, bar_c):
    T = {"B": np.eye(4)}
    T["Y"] = Rax("z", G["yaw"]) @ Tr([0, 0, G["lift_Y"]])
    T["M2"] = T["Y"] @ Tr(G["off_M2"])
    T["H"] = Rax("z", G["yaw"]) @ Tr([0, 0, G["lift_H"]]) @ NodT(G["pitch"])
    T["C"] = T["H"] @ Tr([0, 0, G["lift_C"]])
    T["R"] = T["H"] @ Tr(G["off_R"]) @ about(bar_c, Rax("y", G["flip"])) @ SpinT(G["spin"])
    return T


# ------------------------------------------------------------------ 장면 (동작 목록)
class Plan:
    def __init__(self):
        self.parts = build_parts()
        self.screws = screw_list()
        self.sidx = {s["name"]: i for i, s in enumerate(self.screws)}
        bb = np.array(self.parts["rotor_bar"][0].bounding_box()).reshape(2, 3); self.bar_c = bb.mean(0)
        self.acts = []; self.t = 0.0; self.step = ""; self.n_screw = 0; self.n_nut = 0
        self.chapters = []

    def add(self, kind, dur, text, **kw):
        a = dict(kind=kind, t0=self.t, t1=self.t + dur, text=text, step=self.step, **kw)
        self.acts.append(a); self.t += dur
        return a

    def chapter(self, title, sub=""):
        self.step = title; self.chapters.append(dict(t=round(self.t, 1), title=title, sub=sub))
        self.add("card", 1.6, sub, title=title)

    def part(self, names, d, dist, text, dur=1.4, cam=None, frame_dir="w", ghost=()):
        """부품을 d 방향(세계, 또는 머리 좌표 frame_dir='h')으로 dist 만큼 떨어진 곳에서 제자리로. ghost: 그동안 반투명으로 그릴 부품 (속이 보이게)"""
        names = [names] if isinstance(names, str) else list(names)
        dv = hv(d) if frame_dir == "h" else np.asarray(d, float)
        return self.add("part", dur, text, names=names, off=dv / max(np.linalg.norm(dv), 1e-9) * dist, cam=cam, ghost=tuple(ghost))

    def group(self, key, to, text, dur=1.6, cam=None, wiggle=0.0):
        return self.add("group", dur, text, key=key, to=to, cam=cam, wiggle=wiggle)

    def setg(self, key, value):
        """시간이 들지 않는 값 바꾸기 (예: 회전부만 보이기)"""
        return self.add("group", 0.0, "", key=key, to=value, cam=None, wiggle=0.0)

    def screws_(self, names, text, cam=None, dur=2.1, reverse=False, ghost=()):
        for k, nm in enumerate(names):
            i = self.sidx[nm]; s = self.screws[i]
            if not reverse:
                self.n_screw += 1
            hd = "접시머리" if s["head"] == "접시" else "둥근머리"
            tt = "%s%s · M%g × %d %s%s" % (text, " (%d/%d)" % (k + 1, len(names)) if len(names) > 1 else "", s["m"], s["L"], hd, " + 너트" if s["nut"] else "")
            self.add("screw", dur, tt, idx=i, cam=cam, group=names, reverse=reverse, count=self.n_screw, ghost=tuple(ghost))

    def nut(self, name, d, dist, text, dur=1.0, cam=None, frame_dir="h"):
        self.n_nut += 1
        a = self.part(name, d, dist, text, dur=dur, cam=cam, frame_dir=frame_dir); a["nut_count"] = self.n_nut
        return a


def make_plan():
    pl = Plan()
    F = lambda *v: hv(v)                                                           # 머리 좌표 방향
    pl.add("intro", 4.5, "")
    # ① 받침
    pl.chapter("① 받침", "받침 통에 랜선 홈 마개를 끼우고 덮개를 덮는다. 배터리는 옆 창으로")
    pl.part("tub", (0, 0, 1), 40, "받침 통(01)을 놓는다", cam=dict(az=35, el=32, dist=430))
    pl.part("lan_cap", (0, 0, 1), 30, "랜선 홈 마개(17)를 통 뒤쪽 홈에 위에서 끼운다", cam=dict(az=-150, el=30, dist=330))
    pl.part("cover", (0, 0, 1), 45, "덮개(02)의 세모 표시를 앞에 맞춰 얹는다 (둥근 테 홈에 비비탄 45개)", cam=dict(az=35, el=40, dist=430))
    pl.screws_(["받침 덮개 %d" % i for i in range(1, 5)], "덮개 나사")
    pl.add("battery", 2.2, "배터리를 옆 창으로 밀어 넣는다 (바닥 턱을 넘으면 걸려서 빠지지 않는다)", names=["battery", "usb_a"], cam=dict(az=-55, el=22, dist=520))
    # ② 요크
    pl.chapter("② 요크", "모터 1과 작은 기어를 달고, 요크를 받침에 끼워 고리로 잠근다")
    pl.part("yoke", (0, 0, 1), 40, "요크(04)", cam=dict(az=30, el=24, dist=560))
    pl.part(["motor1", "motor1_cap"], (0, 0, 1), 40, "모터 1을 전선 상자 속 자리에 내린다 (축이 아래 구멍으로)", cam=dict(az=-150, el=45, dist=330), ghost=["yoke"])
    pl.screws_(["모터 1 귀 %d" % i for i in range(1, 3)], "모터 1 귀 나사 (꼭 6mm)", cam=dict(alpha=45, az_pref=-150), ghost=["yoke"])
    pl.part("pinion1", (0, 0, -1), 25, "작은 기어(05)를 아래에서 모터 1 축에 끼운다", cam=dict(az=-140, el=-25, dist=300))
    pl.screws_(["받침 회전 멈춤"], "받침 회전 멈춤 나사 (아래로 6mm 나온다)", cam=dict(az_pref=-130, alpha=45), ghost=["yoke"])
    pl.group("lift_Y", 0.0, "요크를 받침 가운데 통에 끼워 내린다 (작은 기어를 가운데 기어에 물린다)", dur=2.0, cam=dict(focus=(0, 0, 60), az=30, el=18, dist=620))
    pl.part("clip", (-1, 0, 0), 45, "고리(03)를 상자 안에서 뒤에서 앞으로 밀어 잠근다 (딸깍)", cam=dict(az=-160, el=45, dist=360), ghost=["yoke"])
    # ③ 머리 속
    pl.chapter("③ 머리 속", "파이와 드라이버 3장을 기둥에 조이고, 젠더·버튼·동글을 꽂는다")
    pl.part("head_shell", (0, 0, 1), 40, "머리 몸통(07)", cam=dict(az=20, el=16, dist=420))
    pl.part(["pi_board", "pi_parts", "antenna"], F(0, -1, 0), 40, "파이를 뒷벽 기둥 4개에 얹는다 (SD카드를 먼저 꽂는다)", cam=dict(az=12, el=8, dist=330))
    pl.screws_(["파이 %d" % i for i in range(1, 5)], "파이 나사", cam=dict(alpha=42))
    for k in (3, 2, 1):
        pl.part(["driver%d_board" % k, "driver%d_parts" % k], F(0, -1, 0), 40, "드라이버 %d (위에서부터 3, 2, 1. 흰 소켓이 아래)" % k, cam=dict(az=-10, el=10, dist=300))
        pl.screws_(["드라이버 %d-%d" % (k, i) for i in range(1, 5)], "드라이버 %d 나사" % k, cam=dict(alpha=42))
    pl.add("note", 1.8, "점퍼선 18가닥을 꽂는다 (배선 표는 조립 안내 5절)", cam=dict(az=0, el=10, dist=380))
    pl.part("usb_plug", F(-1, 0, 0), 25, "ㄱ자 젠더를 파이 전원 단자에 꽂는다 (왼쪽 창으로)", cam=dict(az=-55, el=5, dist=300))
    pl.part(["button_board", "button_parts"], F(0, -1, 0), 30, "전원 버튼 모듈(KY-004)을 왼쪽 받침 살에 끼운다", cam=dict(az=-20, el=12, dist=280))
    pl.part("dongle", F(0, 0, -1), 40, "무선 동글을 파이 아래 USB에 꽂는다", cam=dict(az=25, el=-25, dist=330))
    pl.part("dongle2", F(0, 0, -1), 40, "탐지 동글을 그 옆 USB에 꽂는다", cam=dict(az=25, el=-25, dist=330))
    # ④ 머리 겉
    pl.chapter("④ 머리 겉", "모터 3을 넣은 모터 자리를 얹고, 부채꼴 기어와 마커 틀을 단다. 뚜껑은 아직")
    pl.part("crown", (0, 0, 1), 40, "모터 자리(09)", cam=dict(az=30, el=35, dist=360))
    pl.part(["motor3", "motor3_cap"], (0, 0, 1), 40, "모터 3을 넣는다 (흰 플러그를 먼저 네모 구멍으로)", cam=dict(az=30, el=45, dist=300))
    pl.screws_(["모터 3 귀 %d" % i for i in range(1, 3)], "모터 3 귀 나사", cam=dict(az_pref=30))
    pl.group("lift_C", 0.0, "모터 자리를 머리 몸통 위에 얹는다 (모터 3 전선을 드라이버 3에)", dur=1.6, cam=dict(az=30, el=30, dist=420))
    pl.screws_(["모터 자리 %d" % i for i in range(1, 5)], "모터 자리 나사", cam=dict(az_pref=30))
    pl.part("sector", F(1, 0, 0), 35, "부채꼴 기어(06)를 오른쪽 옆면에 댄다", cam=dict(az=75, el=15, dist=360))
    pl.screws_(["부채꼴 기어 %d" % i for i in range(1, 3)], "부채꼴 기어 나사")
    pl.part("marker_frame", F(0, 1, 0), 30, "AR 마커 종이를 등판에 대고 마커 틀(15)로 덮는다", cam=dict(az=-160, el=10, dist=360))
    pl.screws_(["마커 틀 %d" % i for i in range(1, 5)], "마커 틀 나사 (꼭 6mm)")
    # ⑤ 머리 올리기
    pl.chapter("⑤ 머리 올리기", "모터 2를 오른쪽 팔에 달고, 머리를 내려 핀으로 잇는다. 마지막에 뚜껑")
    pl.part(["motor2", "motor2_cap"], (0, 1, 0), 40, "모터 2", cam=dict(az=60, el=15, dist=420))
    pl.part("m2_ring", (0, 1, 0), 35, "맞춤 링(16)을 축 둘레 턱에 씌운다 (넓은 구멍이 모터 쪽)", cam=dict(az=75, el=10, dist=300))
    pl.part("pinion2", (0, 1, 0), 35, "작은 기어를 축 끝에서 턱까지 민다", cam=dict(az=75, el=10, dist=300))
    pl.group("off_M2", np.zeros(3), "모터 2를 오른쪽 팔 바깥에서 구멍에 넣는다", dur=1.6, cam=dict(az=70, el=12, dist=420))
    pl.screws_(["모터 2 귀 %d" % i for i in range(1, 3)], "모터 2 귀 나사 (꼭 6mm)")
    pl.group("lift_H", 0.0, "머리를 두 팔 사이로 내린다 (조금 끄덕여 기어 이를 맞물린다)", dur=3.0, wiggle=4.0, cam=dict(focus=(0, 0, 150), az=40, el=14, dist=720))
    pl.part("pin_r", (0, 1, 0), 30, "오른쪽 핀을 바깥에서 민다 (바늘이 아래)", cam=dict(az=75, el=10, dist=330))
    pl.part("pin_l", (0, -1, 0), 30, "왼쪽 핀", cam=dict(az=-75, el=10, dist=330))
    pl.part("keeper_r", (0, 1, 0), 25, "핀 막이(14)를 핀 머리 위에 덮는다", cam=dict(az=80, el=10, dist=280))
    pl.screws_(["핀 막이 (오른쪽)"], "핀 막이 나사")
    pl.part("keeper_l", (0, -1, 0), 25, "왼쪽 핀 막이", cam=dict(az=-80, el=10, dist=280))
    pl.screws_(["핀 막이 (왼쪽)"], "핀 막이 나사")
    pl.screws_(["끄덕임 멈춤 %d" % i for i in range(1, 3)], "끄덕임 멈춤 나사 (꼭 10mm. 머리 옆 돌기가 두 나사 사이에)")
    pl.add("note", 1.8, "모터 1·2 전선과 전원선을 머리 아래로 올려 꽂고, 케이블 타이로 묶는다", cam=dict(az=20, el=-10, dist=480))
    pl.part("head_lid", F(0, -1, 0), 45, "뚜껑(08)을 닫는다", cam=dict(az=25, el=12, dist=420))
    pl.screws_(["머리 뚜껑 %d" % i for i in range(1, 5)], "뚜껑 나사")
    pl.screws_(["고정 나사(옮길 때)"], "옮길 때만: 머리를 0°에 묶는 고정 나사 (재기 전에 뺀다)")
    # ⑥ 판과 균형추
    pl.chapter("⑥ 판과 균형추", "동전 12개를 넣고, 판 틀·알루미늄 판·누름쇠를 단 뒤 모터 3 축에 끼운다")
    pl.setg("show_main", 0.0)
    pl.part("rotor_bar", (0, 0, 1), 40, "회전 막대(10)", cam="R")
    pl.group("flip", 180.0, "막대를 뒤집어 균형추 통 입구가 위를 보게 한다", dur=1.4, cam="R")
    for k in range(N_COIN):
        pl.part("coin%02d" % k, F(0, 0, -1), 40, "100원 동전 %d/%d" % (k + 1, N_COIN), dur=0.45, cam="POD")
    pl.part("coin_cap", F(0, 0, -1), 30, "마개(11)를 덮는다 (동전과 마개 사이 틈은 나사가 눌러 잡는다)", cam="POD")
    pl.screws_(["균형추 마개 %d" % i for i in range(1, 3)], "마개 나사", cam="POD")
    pl.group("flip", 0.0, "막대를 바로 놓는다", dur=1.4, cam="R")
    for i in range(4):
        pl.nut("nut_fl%d" % i, (-1, 0, 0), 18, "판 받이 안쪽 육각 자리에 M3 너트 (%d/4)" % (i + 1), cam="FLANGE")
    pl.part("plate_frame", F(1, 0, 0), 30, "판 틀(18)을 판 받이에 댄다 (평평한 뒷면이 판 받이에)", cam="RF")
    pl.screws_(["판 틀 나사 %d" % i for i in range(1, 5)], "판 틀 나사 (판이 덮는다)", cam="RF")
    pl.add("plate", 3.6, "알루미늄 판(100 × 150 × 1T)의 아래 변을 아래 턱 밑으로 비스듬히 밀어 넣고 눕힌다", names=["plate"], cam="RP")
    for i in range(2):
        pl.nut("nut_cl%d" % i, (-1, 0, 0), 18, "판 틀 뒷면 위쪽 육각 자리에 M3 너트 (%d/2)" % (i + 1), cam="RB")
    pl.part("plate_clamp", F(1, 0, 0), 30, "누름쇠(19)를 판 위 변에 댄다", cam="RT")
    pl.screws_(["누름쇠 나사 %d" % i for i in range(1, 3)], "누름쇠 나사 (누름쇠가 판에 닿아 멈출 때까지)", cam="RT")
    pl.setg("show_main", 1.0)
    pl.group("off_R", np.array([0.0, 0.0, 70.0]), "회전부를 머리 위로 옮긴다", dur=2.4, cam=dict(focus=(0, -150, 180), az=-25, el=18, dist=950))
    pl.group("off_R", np.zeros(3), "납작한 면을 맞춰 모터 3 축에 끼운다 (막대 가운데를 누른다)", dur=1.6, cam=dict(focus=(-20, 0, 250), az=-30, el=25, dist=520))
    pl.nut("nut_sh", (0, 0, 1), 20, "축 고정 너트를 막대 윗면 홈에 떨어뜨린다", cam=dict(focus=(-20, 0, 260), az=-60, el=40, dist=260))
    pl.screws_(["축 고정"], "축 고정 나사 (손끝에 걸릴 때까지만)")
    pl.screws_(["고정 나사(옮길 때)"], "재기 전에 고정 나사를 뺀다", reverse=True)
    pl.step = "다 됐다"; pl.chapters.append(dict(t=round(pl.t, 1), title="다 됐다", sub="재는 자세"))
    pl.add("outro", 8.0, "재는 자세(54.7°)로 숙이고 판을 돌린다", cam=dict(focus=(0, 0, 175), az=30, el=15, dist=720))
    pl.add("end", 2.5, "")
    return pl


# ------------------------------------------------------------------ 시각 t 의 상태
def state(pl, t):
    """보이는 것, 부품마다의 행렬(무리 행렬 곱하기 전), 강조, 무리 값, 지금 동작"""
    vis = {}; loc = {}; hi = set(); G = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in G0.items()}
    scr = {}                                                                       # 나사 번호 -> (보임, 뺀 거리, 돈 각, 드라이버 보임, 드라이버 뺀 거리)
    cur = pl.acts[-1]
    for a in pl.acts:
        if t < a["t0"]:
            break
        u = 1.0 if t >= a["t1"] else (t - a["t0"]) / (a["t1"] - a["t0"])
        if t < a["t1"]:
            cur = a
        k = a["kind"]
        if k == "part" or k == "battery":
            for n in a["names"]:
                vis[n] = True
                if k == "part":
                    loc[n] = Tr(a["off"] * (1.0 - ease(u)))
                else:                                                              # 배터리: 문턱 위로 3.2mm 들고 옆 창으로 민 뒤 내려앉는다
                    y = -130.0 * (1.0 - seg(u, 0.0, 0.8)); z = 3.2 * (1.0 - seg(u, 0.8, 1.0))
                    loc[n] = Tr([0, y, z])
                if u < 1.0:
                    hi.add(n)
        elif k == "plate":
            vis["plate"] = True
            th = 18.0 * (1.0 - seg(u, 0.45, 1.0)); s = 26.0 * (1.0 - seg(u, 0.0, 0.45))
            r = np.radians(th); zp = max(BL_Z0, BL_Z0 - FR_GAP + np.sin(r) + 0.05)
            Tp = Tr([FL_X1 + s * np.sin(r), 0, zp + s * np.cos(r)]) @ Rax("y", th) @ Tr([-FL_X1, 0, -BL_Z0])
            loc["plate"] = W0 @ Tp @ W0i
            if u < 1.0:
                hi.add("plate")
        elif k == "group":
            v0 = G[a["key"]]
            G[a["key"]] = v0 + (a["to"] - v0) * ease(u)
            if a.get("wiggle") and u < 1.0:
                G["pitch"] = a["wiggle"] * np.sin(np.pi * 3.0 * seg(u, 0.55, 1.0)) * (1.0 - seg(u, 0.85, 1.0))
            elif a.get("wiggle"):
                G["pitch"] = 0.0
        elif k == "screw":
            i = a["idx"]; s = pl.screws[i]
            if not a["reverse"]:
                back = s["L"] + 14.0; turns = min(9.0, s["L"] * 0.75)
                if u < 0.22:                                                       # 다가온다
                    bk = back - (back - s["L"]) * ease(u / 0.22); ang = 0.0; dshow = True; dback = 0.0
                elif u < 0.78:                                                     # 돌리며 들어간다
                    w = (u - 0.22) / 0.56; bk = s["L"] * (1.0 - w); ang = 360.0 * turns * w; dshow = True; dback = 0.0
                elif u < 0.88:                                                     # 마지막으로 조인다
                    bk = 0.0; ang = 360.0 * (turns + 0.25 * (u - 0.78) / 0.10); dshow = True; dback = 0.0
                else:                                                              # 드라이버가 물러난다
                    bk = 0.0; ang = 360.0 * (turns + 0.25); dshow = True; dback = 60.0 * ease((u - 0.88) / 0.12)
                scr[i] = (True, bk, ang, dshow and u < 1.0, dback)
                if u < 1.0:
                    hi.add("screw%d" % i)
            else:                                                                  # 풀어서 뺀다
                turns = min(9.0, s["L"] * 0.75)
                if u < 0.15:
                    bk = 0.0; ang = 0.0; dback = 60.0 * (1.0 - ease(u / 0.15))
                elif u < 0.75:
                    w = (u - 0.15) / 0.6; bk = s["L"] * w; ang = -360.0 * turns * w; dback = 0.0
                else:
                    w = ease((u - 0.75) / 0.25); bk = s["L"] + 30.0 * w; ang = -360.0 * turns; dback = 30.0 * w
                scr[i] = (u < 1.0, bk, ang, u < 1.0, dback)
                if u < 1.0:
                    hi.add("screw%d" % i)
        elif k == "outro":
            G["yaw"] = 35.0 * seg(u, 0.0, 0.35); G["pitch"] = TILT3 * seg(u, 0.05, 0.45); G["spin"] = 540.0 * seg(u, 0.35, 1.0)
    return vis, loc, hi, G, scr, cur


# ------------------------------------------------------------------ 그리기
class Renderer:
    def __init__(self, pl):
        import pyvista as pv
        import render as RD
        self.pv = pv; self.pl = pl
        self.p = pv.Plotter(off_screen=True, window_size=(W, H))
        self.p.set_background(BG)
        self.p.enable_depth_peeling(number_of_peels=6)
        try:
            self.p.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        self.act = {}; self.col = {}; self.grp = {}
        for n, (m, g, c) in pl.parts.items():
            if m.is_empty():
                continue
            kw = dict(color=c, smooth_shading=True, specular=0.25, specular_power=18, ambient=0.25, diffuse=0.8)
            if n in ("plate",) or n.startswith(("coin", "nut_")) or (n.startswith("motor") and not n.endswith("cap")):
                kw.update(specular=0.7, specular_power=40)
            self.act[n] = self.p.add_mesh(RD.to_pv(m), **kw); self.col[n] = c; self.grp[n] = part_group(n, g)
        self.base = {}
        for i, s in enumerate(pl.screws):
            mesh = screw_mesh(s["m"], s["L"], s["head"], s["hh"], s["hr"]).transform(frame(s["p"], s["d"])[:3, :])
            n = "screw%d" % i
            self.act[n] = self.p.add_mesh(RD.to_pv(mesh), color=STEEL, smooth_shading=True, specular=0.8, specular_power=50, ambient=0.25, diffuse=0.75)
            self.col[n] = STEEL; self.grp[n] = screw_group(s["name"])
        self.drv = {}
        for m in (3.0, 2.5, 2.0):
            shaft, handle, stripe = driver_mesh(m)
            self.drv[m] = [self.p.add_mesh(RD.to_pv(shaft), color="#7E858E", specular=0.8, specular_power=50, smooth_shading=False),
                           self.p.add_mesh(RD.to_pv(handle), color="#D94A3A", specular=0.3, smooth_shading=False, opacity=0.55),
                           self.p.add_mesh(RD.to_pv(stripe), color="#F3C623", smooth_shading=False)]
        self.p.camera.up = (0, 0, 1); self.p.camera.view_angle = 30.0
        self.cams = {}

    def matrices(self, t):
        pl = self.pl
        vis, loc, hi, G, scr, cur = state(pl, t)
        if cur["kind"] == "intro":                                                # 처음: 다 조립한 모습 (받침 0, 끄덕임 0)
            t_done = [a for a in pl.acts if a["kind"] == "outro"][0]["t0"]
            vis, loc, hi, G, scr, _ = state(pl, t_done)
        GT = group_T(G, pl.bar_c)
        out = {}
        for n in self.act:
            if n.startswith("screw"):
                i = int(n[5:]); s = pl.screws[i]
                if i in scr and scr[i][0]:
                    _, bk, ang, _, _ = scr[i]; F_ = frame(s["p"], s["d"])
                    out[n] = (True, GT[self.grp[n]] @ F_ @ Tr([0, 0, -bk]) @ Rax("z", ang) @ np.linalg.inv(F_), n in hi)
                else:
                    out[n] = (False, None, False)
            else:
                v = vis.get(n, False)
                out[n] = (v, GT[self.grp[n]] @ loc.get(n, np.eye(4)), n in hi)
        drv = None
        for i, (sh, bk, ang, dshow, dback) in scr.items():
            if dshow:
                s = pl.screws[i]; F_ = frame(s["top"], s["d"])
                drv = (s["m"], GT[screw_group(s["name"])] @ F_ @ Tr([0, 0, -bk - dback - 0.05]) @ Rax("z", ang))
        return out, drv, G, cur, GT

    def camera(self, a, GT=None):
        """동작 a 의 카메라 (초점, 방위각, 높이각, 거리). 동작이 끝날 때의 자리로 잡는다"""
        key = id(a)
        if key in self.cams:
            return self.cams[key]
        pl = self.pl; c = a.get("cam")
        out, drv, G, cur, GTe = self.matrices(a["t1"] - 1e-3)
        def center(names):
            pts = []
            for n in names:
                if n in self.act and not n.startswith("screw") and out[n][0]:
                    b = np.array(pl.parts[n][0].bounding_box()).reshape(2, 3)
                    q = np.array([[x, y, z, 1.0] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]]) @ out[n][1].T
                    pts.append(q[:, :3].mean(0))
            return np.mean(pts, 0) if pts else None
        rotor_pt = lambda hp_: (GTe["R"] @ W0 @ np.r_[hp_, 1.0])[:3]
        if isinstance(c, str):
            presets = {"R": ([0, 0, 40], -150, 28, 460), "POD": ([POD_XC, 0, 25], -150, 35, 230), "FLANGE": ([70, 0, 20], -130, 30, 170),
                       "RF": ([75, 0, 20], 40, 22, 330), "RP": ([75, 0, 15], 35, 15, 560), "RB": ([75, 0, 96], -120, 30, 200), "RT": ([80, 0, 94], 45, 30, 240)}
            hp_, az, el, dist = presets[c]; f = rotor_pt(hp_)
        elif a["kind"] == "screw":
            idx = [pl.sidx[n] for n in a["group"]]
            pts = np.array([(GTe[screw_group(pl.screws[i]["name"])] @ np.r_[pl.screws[i]["top"], 1.0])[:3] for i in idx])
            dd = np.array([(GTe[screw_group(pl.screws[i]["name"])][:3, :3] @ -pl.screws[i]["d"]) for i in idx]).mean(0)
            dd /= max(np.linalg.norm(dd), 1e-9)
            f = pts.mean(0); spread = float(np.max(np.linalg.norm(pts - f, axis=1))) if len(pts) > 1 else 0.0
            cc = c if isinstance(c, dict) else {}
            al = np.radians(cc.get("alpha", 50.0)); ap = np.radians(cc.get("az_pref", 35.0))
            if abs(dd[2]) < 0.75:                                                 # 옆으로 들어가는 나사: 위쪽으로 비켜서
                w_ = np.array([0, 0, 1.0]) - dd[2] * dd
            else:                                                                  # 위아래로 들어가는 나사: az_pref 쪽으로 비켜서
                w_ = np.array([np.cos(ap), np.sin(ap), 0.0]); w_ = w_ - (w_ @ dd) * dd
            w_ /= np.linalg.norm(w_); cv = np.cos(al) * dd + np.sin(al) * w_
            az = float(np.degrees(np.arctan2(cv[1], cv[0]))); el = float(np.degrees(np.arcsin(np.clip(cv[2], -1, 1))))
            dist = float(np.clip(1.5 * spread + (250.0 if pl.screws[idx[0]]["m"] >= 3.0 else 210.0), 200.0, 480.0))
            if isinstance(c, dict):
                az, el, dist = c.get("az", az), c.get("el", el), c.get("dist", dist)
                if "focus" in c:
                    f = np.asarray(c["focus"], float)
        else:
            c = c or {}
            names = a.get("names") or {"lift_C": ["crown"], "off_M2": ["motor2"], "lift_Y": ["yoke"], "lift_H": ["head_shell"], "flip": ["rotor_bar"], "off_R": ["rotor_bar"]}.get(a.get("key"), [])
            f = np.asarray(c["focus"], float) if "focus" in c else center(names)
            if f is None:                                                         # 부품이 없는 동작(설명)은 앞 동작의 초점을 쓴다
                prev = [b for b in pl.acts if b is not a and b["t1"] <= a["t0"] and b["t1"] > b["t0"] and b["kind"] not in ("card", "intro", "end")]
                f = self.camera(prev[-1])[0] if prev else np.array([0, 0, 150.0])
            az, el, dist = c.get("az", 30.0), c.get("el", 20.0), c.get("dist", 500.0)
        self.cams[key] = (np.asarray(f, float), float(az), float(el), float(dist))
        return self.cams[key]

    def cam_at(self, t, GT):
        pl = self.pl; acts = [a for a in pl.acts if a["kind"] not in ("card", "end") and a["t1"] > a["t0"]]
        i = max([k for k, a in enumerate(acts) if a["t0"] <= t] or [0]); a = acts[i]
        if a["kind"] == "intro":
            return np.array([0, 0, 175.0]), 30.0 + 40.0 * (t / a["t1"]), 15.0, 700.0
        c1 = self.camera(a)
        if a["kind"] == "outro":
            f, az, el, d = c1; u = (t - a["t0"]) / (a["t1"] - a["t0"])
            return f, az + 50.0 * u, el, d
        c0 = self.camera(acts[i - 1]) if i > 0 and acts[i - 1]["kind"] not in ("intro",) else (np.array([0, 0, 175.0]), 70.0, 15.0, 700.0)
        tc = min(0.8, 0.45 * (a["t1"] - a["t0"]))
        same = np.allclose(c0[0], c1[0]) and abs(c0[1] - c1[1]) < 1e-6 and abs(c0[3] - c1[3]) < 1e-6
        w = 1.0 if same else ease((t - a["t0"]) / tc)
        f = c0[0] + (c1[0] - c0[0]) * w
        daz = ((c1[1] - c0[1] + 180.0) % 360.0) - 180.0
        return f, c0[1] + daz * w, c0[2] + (c1[2] - c0[2]) * w, c0[3] + (c1[3] - c0[3]) * w

    def frame(self, t):
        out, drv, G, cur, GT = self.matrices(t)
        ghost = set(cur.get("ghost", ()))
        for n, (v, T, h) in out.items():
            a = self.act[n]
            if G["show_main"] < 0.5 and self.grp[n] != "R":
                v = False
            a.SetVisibility(bool(v))
            if not n.startswith("screw"):
                a.prop.opacity = 0.28 if n in ghost else 1.0
            if v:
                a.user_matrix = T
                a.prop.color = (GOLD if n.startswith("screw") else HOT) if h else self.col[n]
        for m, acts in self.drv.items():
            show = drv is not None and drv[0] == m
            for a in acts:
                a.SetVisibility(show)
                if show:
                    a.user_matrix = drv[1]
        f, az, el, dist = self.cam_at(t, GT)
        ar, er = np.radians(az), np.radians(el)
        self.p.camera.focal_point = tuple(f)
        self.p.camera.position = tuple(f + dist * np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)]))
        self.p.camera.up = (0, 0, 1)
        self.p.reset_camera_clipping_range()
        self.p.render()
        img = Image.fromarray(np.asarray(self.p.screenshot(return_img=True)))
        return overlay(img, self.pl, t, cur)


def overlay(img, pl, t, cur):
    d = ImageDraw.Draw(img, "RGBA")
    k = cur["kind"]
    if k in ("intro", "end"):
        u = (t - cur["t0"]) / (cur["t1"] - cur["t0"])
        if k == "end":
            img = Image.blend(img, Image.new("RGB", img.size, BG), min(1.0, 0.25 + u))
            d = ImageDraw.Draw(img, "RGBA")
            d.text((W // 2, H // 2 - 30), "조립 끝", font=font(54, True), fill=INK, anchor="mm")
            d.text((W // 2, H // 2 + 40), "나사 %d개 · 너트 %d개 · 출력 부품과 사는 부품의 치수는 조립 안내에" % (pl.n_screw, pl.n_nut), font=font(26), fill=SUB, anchor="mm")
            return img
        d.rectangle([0, 0, W, 150], fill=(244, 242, 238, 215))
        d.text((48, 30), "DeteCam 수신기 조립 시뮬레이션", font=font(46, True), fill=INK)
        d.text((50, 96), "출력 부품과 사는 부품을 조립 안내의 순서대로 · 나사 %d개를 하나씩 · 너트 %d개" % (pl.n_screw, pl.n_nut), font=font(25), fill=SUB)
        return img
    if k == "card":
        u = (t - cur["t0"]) / (cur["t1"] - cur["t0"])
        a = int(255 * (0.86 if 0.12 < u < 0.88 else 0.86 * min(u / 0.12, (1 - u) / 0.12)))
        d.rectangle([0, H // 2 - 95, W, H // 2 + 95], fill=(244, 242, 238, a))
        d.text((W // 2, H // 2 - 28), cur["title"], font=font(58, True), fill=(34, 38, 43, a), anchor="mm")
        d.text((W // 2, H // 2 + 46), cur["text"], font=font(27), fill=(91, 100, 112, a), anchor="mm")
        return img
    d.rectangle([0, 0, W, 64], fill=(244, 242, 238, 225))
    d.text((28, 12), cur["step"], font=font(32, True), fill=INK)
    ns = sum(1 for a in pl.acts if a["kind"] == "screw" and not a["reverse"] and a["t1"] <= t)
    nn = sum(1 for a in pl.acts if a.get("nut_count") and a["t1"] <= t)
    d.text((W - 28, 18), "나사 %d / %d   너트 %d / %d" % (ns, pl.n_screw, nn, pl.n_nut), font=font(24), fill=SUB, anchor="ra")
    d.rectangle([0, H - 70, W, H], fill=(244, 242, 238, 230))
    d.text((28, H - 54), cur["text"], font=font(28), fill=INK)
    total = pl.acts[-1]["t1"]
    d.rectangle([0, H - 6, int(W * t / total), H], fill=HOT)
    return img


# ------------------------------------------------------------------ 실행
def ffmpeg_writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True); os.makedirs(paths.VID, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "join":
        n = int(sys.argv[2])
        lst = os.path.join(TMP, "parts.txt")
        open(lst, "w").write("".join("file '%s'\n" % os.path.join(TMP, "part%02d.mp4" % k) for k in range(n)))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUTV], check=True)
        print("saved", OUTV, "%.1f MB" % (os.path.getsize(OUTV) / 1e6))
        sys.exit(0)
    pl = make_plan()
    total = pl.acts[-1]["t1"]
    if cmd == "gif":                                                               # 문서에 넣는 짧은 움직이는 그림: 판을 넣고 누름쇠를 조이는 장면
        t0 = [a for a in pl.acts if a["kind"] == "plate"][0]["t0"] - 0.3
        t1 = [a for a in pl.acts if a["kind"] == "screw" and pl.screws[a["idx"]]["name"].startswith("누름쇠")][-1]["t1"]
        out = os.path.join(paths.IMG, "35_나사_조이기.gif")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.2f" % t0, "-t", "%.2f" % (t1 - t0), "-i", OUTV, "-vf",
                        "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", out], check=True)
        print("saved", out, "%.1f MB, %.1f초" % (os.path.getsize(out) / 1e6, t1 - t0))
        sys.exit(0)
    if cmd == "plan":
        for a in pl.acts:
            if a["kind"] in ("card", "intro", "end"):
                print("%6.1f  [%s] %s" % (a["t0"], a["kind"], a.get("title", a["text"])))
        print("길이 %.1f초 (%d 프레임), 나사 %d, 너트 %d, 동작 %d" % (total, int(total * FPS), pl.n_screw, pl.n_nut, len(pl.acts)))
        json.dump(dict(length_s=round(total, 1), fps=FPS, size=[W, H], screws=pl.n_screw, nuts=pl.n_nut, chapters=pl.chapters, file="조립_시뮬레이션.mp4"),
                  open(os.path.join(paths.RES, "anim_assembly.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    R = Renderer(pl)
    if cmd == "still":
        for ts in sys.argv[2:]:
            R.frame(float(ts)).save(os.path.join(TMP, "still_%07.2f.png" % float(ts))); print("still", ts)
    elif cmd == "render":
        k, n = int(sys.argv[2]), int(sys.argv[3])
        nf = int(total * FPS); a, b = nf * k // n, nf * (k + 1) // n
        wtr = ffmpeg_writer(os.path.join(TMP, "part%02d.mp4" % k))
        for i in range(a, b):
            wtr.stdin.write(np.asarray(R.frame(i / FPS).convert("RGB"), dtype=np.uint8).tobytes())
            if (i - a) % 240 == 0:
                print("frame %d / %d" % (i, b), flush=True)
        wtr.stdin.close(); wtr.wait()
        print("done part", k)
