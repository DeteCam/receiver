"""DeteCam 패널 수신기 (v4, 두 대역 판) 의 치수와 부품. 단위 mm. 모터가 없다 (숙소용 회전 받침은 따로).

패널 좌표: 원점 = 안테나 받침판 앞면의 한가운데. x = 앞에서 볼 때 오른쪽, y = 위, z = 방 쪽(앞).
허브 좌표: 원점 = 통 바닥 밑면의 한가운데. x = 긴 쪽 (파이의 USB 쪽이 +x), y = 폭, z = 위.
회전 받침 좌표: 원점 = 도는 축이 받침대 윗판(덱)의 윗면과 만나는 점. z = 위, y = 패널이 보는 쪽.

패널 하나 = 통(1) + 앞덮개(2) + 자석 걸이판(3) 또는 받침 발(4) + ESP32 누름쇠(5) + 안테나 받침판(9)
허브     = 허브 통(6) + 허브 선반(7) + 허브 뚜껑(8)
회전 받침 = 받침대(10) + 돌림판(11) + 축 이음쇠(12). 숙소에서 받침 발을 단 패널을 얹고 28BYJ-48 로 45도씩 돌린다 (위로 20도 기울여)

[실측] 표시는 부품을 받은 뒤 재서 고칠 값이다. [관문 4] 는 관문 4 에서 재 보고 정할 값이다.
2026-11-19: 사 오는 2.4 GHz 판(CoreHW CHW1010-ANT2) 대신 두 대역 필름 안테나 16개를 붙이는 받침판과 반사판으로 바꿨다 (스펙 008).
"""
import numpy as np
from cadlib import *

# ---- 사 오는 부품 ----
FILM = (34.9, 9.0, 0.2)        # 두 대역 필름 안테나 Molex 146153: 길이, 폭, 두께. 선(U.FL)은 한쪽 짧은 변 끝에서 나온다 (데이터시트)
PITCH = 33.0                   # 안테나 간격 (4 x 4). 두 대역의 가운데 (다른 방법 검토 5.5절)
FILM_ANG = 45.0                # 안테나를 눕힌 각도 (도). 이웃과 겹치지 않고 두 편파를 고루 받는다
AP, AP_T = 140.0, 2.0          # 안테나 받침판(뽑는다) 한 변, 두께
RG = 17.5                      # [관문 4] 안테나 면에서 반사판 앞면까지 (15~20 사이에서 정한다)
REF_T = 1.0                    # 반사판(알루미늄 판 140 x 140) 두께
LEG = [(-40.0, -23.0), (40.0, -23.0), (-40.0, 23.0), (40.0, 23.0)]           # 받침판 다리 = 반사판 기둥 자리 (모듈 줄 사이, SMA 단자 사이)
XIAO = (21.0, 17.8, 1.2)       # Seeed XIAO ESP32-C5: 길이, 폭, 기판 두께. USB-C 는 짧은 변 가운데, 반대쪽 끝에 U.FL
USB_C = (9.6, 3.8)             # USB-C 플러그가 지나는 창
CELL = (77.0, 21.0, 19.5)      # 18650 한 알 홀더 (스위치 달린 것): 길이, 폭, 높이
SP4T = (45.0, 30.0, 11.0)      # RF 스위치 모듈 HMC7992 (SMA 6개): 기판 45 x 30, SMA 까지의 높이
COMB = (32.0, 26.0, 11.0)      # [실측] 두 대역 합침기 모듈
MAG_D, MAG_T = 20.0, 3.0       # 접시 구멍 네오디뮴 자석 지름 20 x 3T
NUT14 = (11.11, 5.56, 6.8)     # 1/4-20 너트: 맞변, 두께, 나사 구멍
BOLT14 = (11.11, 4.4)          # 1/4-20 육각 볼트 머리: 맞변, 두께
M3N = (5.5, 2.4)               # M3 너트: 맞변, 두께
PI = (85.0, 56.0, 1.6)         # 라즈베리파이 3B+
PI_HOLES = [(3.5, 3.5), (3.5, 52.5), (61.5, 3.5), (61.5, 52.5)]
BATT = (152.0, 76.0, 22.0)     # 보조 배터리 한계 (큰 것 기준)
DONGLE = (34.0, 16.0, 8.0, 12.0)   # 무선 동글: 길이, 폭, 두께, 꽂히는 길이
MOTOR = dict(body_d=28.0, body_h=19.0, off=8.0, ear=35.0, ear_w=7.0, boss_d=9.0, boss_h=1.5, shaft=9.0, cover=(17.0, 15.0, 16.5))   # 28BYJ-48: 몸통, 축이 몸통 가운데에서 비킨 거리, 귀 구멍 사이, 축 길이(보스 위), 배선 덮개(폭, 튀어나온 길이, 높이)
BEAR = (75.0, 8.0)             # [실측] 회전 받침 베어링(정사각 턴테이블): 한 변, 두께
BEAR_LO = [(-31.0, -31.0), (31.0, -31.0), (-31.0, 31.0), (31.0, 31.0)]       # [실측] 아래 판 나사 자리 (받침대에)
BEAR_HI = [(-31.0, -31.0), (31.0, -31.0), (-31.0, 31.0), (31.0, 31.0)]       # [실측] 위 판 나사 자리 (돌림판에, 아래 판의 큰 구멍으로 조인다)

# ---- 패널 통 ----
MX = 10.0                      # 받침판 옆 여유 = 안테나 선이 반사판 뒤로 내려가는 길 (안쪽 반폭 = 70 + 10)
TOP, BOT = 24.0, 46.0          # 받침판 위 띠(기준 안테나), 받침판 아래 칸(ESP32, 배터리)
WALL, FLOOR = 2.4, 2.0
CAV = 24.0                     # 반사판 뒷면에서 통 바닥까지 (RF 모듈 여덟과 선이 들어간다)
GAPB = 0.4                     # 받침판 둘레 틈
LEDGE = 2.5                    # 받침판 가장자리를 받치는 턱의 너비
DIV = 2.0                      # 칸막이 두께
RC = 6.0                       # 통 모서리 둥글기
IX = AP / 2 + MX               # 안쪽 반폭 80
OX = IX + WALL                 # 바깥 반폭 82.4
Y1 = AP / 2 + TOP; Y0 = -AP / 2 - BOT          # 안쪽 위끝 94, 아래끝 -116
OY1, OY0 = Y1 + WALL, Y0 - WALL                # 바깥 96.4, -118.4
ZB = -AP_T                     # 받침판 뒷면 = 턱 윗면
ZR = -RG - REF_T               # 반사판 뒷면 = 기둥 윗면
ZF = ZR - CAV                  # 통 바닥 안쪽 면 -42.5
ZO = ZF - FLOOR                # 통 뒷면 -44.5
BOSS = [(-IX + 3.6, Y1 - 3.6), (IX - 3.6, Y1 - 3.6), (-IX + 3.6, Y0 + 3.6), (IX - 3.6, Y0 + 3.6), (-(AP / 2 + GAPB + 3.3), 0.0), (AP / 2 + GAPB + 3.3, 0.0)]   # 앞덮개 나사 6개 (M3 x 10)
BOSS_R = 3.3
MOUNT = [(-40.0, -40.0), (40.0, -40.0), (-40.0, 40.0), (40.0, 40.0)]          # 뒷면 걸이 자리: 80 x 80 의 M3 구멍 4개
XIAO_C = (IX - 0.3 - XIAO[0] / 2, Y0 + 24.0)   # XIAO 가운데 (눕혀 놓는다: USB-C 가 오른쪽 벽으로 나온다)
XIAO_Z = ZF + 1.6              # XIAO 기판 밑면
USB_H = 3.2                    # USB-C 단자의 높이 (기판 윗면에서)
CLIP_X = IX - 0.3 - 3.8        # 누름쇠의 x
CLIP_Z = XIAO_Z + XIAO[2] + USB_H + 1.2      # 누름쇠 밑면 = 나사 기둥의 윗면
CELL_C = (-33.0, Y0 + CELL[1] / 2 + 1.5)       # 배터리 홀더 가운데
MODS = [(-47.0, 46.0, "s"), (0.0, 46.0, "s"), (47.0, 46.0, "s"), (-47.0, 0.0, "s"), (0.0, 0.0, "s"), (47.0, 0.0, "c"), (-47.0, -46.0, "s"), (0.0, -46.0, "s")]   # 반사판 뒤의 RF 모듈: 16갈래 다섯(첫 단 넷 + 둘째 단), 합침기, 위상 스위치 둘

# ---- 앞덮개 ----
AIR = 3.0                      # 안테나 앞의 빈틈 (덮개가 안테나에 닿지 않게)
SKIN = 1.2                     # 덮개 살 두께 (전파가 지나는 곳)
ZC = AIR + SKIN                # 덮개 겉면 4.2
MARK = (108.0, 44.0)           # AR 마커 띠 (marker/AR_마커.pdf: 34mm 마커 둘). 아래 칸 앞에 붙인다
MARK_SIZE, MARK_DX = 34.0, 27.0             # 마커 한 변, 마커 가운데의 x (좌우 ±27)
MARK_C = (0.0, Y0 + BOT / 2 - 1.0)

# ---- 자석 걸이판 ----
MT_W, MT_T = 112.0, 6.0        # 한 변, 두께
MAGS = [(-28.0, 0.0), (28.0, 0.0), (0.0, -28.0), (0.0, 28.0)]
TAB = (26.0, 16.0, 4.5)        # 위 고리: 폭, 높이, 구멍 반지름

# ---- 받침 발 (창턱·교탁·사물함 위에 세우거나 삼각대·회전 받침에 올린다) ----
FT_W, FT_T = 80.0, 4.0         # 폭, 판 두께
FT_D = 70.0                    # 발이 세로 판 뒤로 나가는 길이
FT_FRONT = 24.0                # 발 앞끝의 z (앞덮개보다 20mm 앞: 앞으로 넘어지지 않게)
FT_TOP = -28.0                 # 세로 판 위끝의 y (아래쪽 걸이 구멍 둘만 쓴다)
FT_GAP = 0.3                   # 통 밑면과 발 윗면 사이 (통이 발 위에 얹힌다)
FT_BLOCK = 7.5                 # 너트가 들어가는 두꺼운 곳의 높이

# ---- 회전 받침 (숙소) ----
TILT = 20.0                    # 패널을 위로 기울이는 각 (도). 계산: 20도가 수평 · 35도보다 낫다 (추정 구간 계산 5절)
TB_R, TB_H, DECK_T = 55.0, 36.0, 4.0           # 받침대 반지름, 높이, 윗판 두께
TT_R, TT_T = 52.0, 8.0         # 돌림판 반지름, 두께 (베어링 위에 얹힌다)
TT_AXIS = -70.0                # 도는 축이 지나는 패널의 z: 20도 기울이면 무게 가운데가 뒤로 약 40mm 옮겨 가므로 그 아래에 둔다 (어림)
# ---- 허브 ----
HL, HW_, HH = 160.0, 82.0, 54.0     # 안쪽 길이, 폭, 높이
HWALL, HFLOOR = 2.4, 2.4
DECK_Z = HFLOOR + BATT[2] + 2.0     # 선반 밑면 (배터리 위)
DECK_T = 2.0
PI_X0 = -HL / 2 + 6.0               # 파이의 SD 카드 쪽 끝
PI_STAND = 4.0


def hex_cs(af):
    r = af / 2 / np.cos(np.pi / 6); a = np.radians(np.arange(6) * 60 + 30)
    return poly(np.c_[r * np.cos(a), r * np.sin(a)])



def _outline(grow=0.0):
    return rrect(-OX - grow, OX + grow, OY0 - grow, OY1 + grow, RC + grow)


def tray():
    """1. 패널 통. 반사판은 기둥 넷에 얹히고, 그 위의 안테나 받침판 다리가 반사판을 누르며, 앞덮개가 받침판 가장자리를 누른다.
    안테나 선은 받침판과 반사판 사이를 지나 좌우 10mm 틈으로 반사판 뒤의 RF 모듈에 간다. 출력: 뒷면을 바닥에, 받침 없이."""
    body = extrude_z(_outline(), ZO, 0.0) - extrude_z(rrect(-IX, IX, Y0, Y1, RC - WALL), ZF, 1.0)
    hb = AP / 2 + GAPB
    add = [box(-IX, IX, hb, hb + DIV, ZF - 0.01, 0.0), box(-IX, IX, -hb - DIV, -hb, ZF - 0.01, 0.0),                                 # 위·아래 칸막이 (받침판과 반사판의 위아래 자리)
           prism_xy_to_yz(poly([(hb + 0.01, ZB), (hb - LEDGE, ZB), (hb - LEDGE, ZB - 0.8), (hb + 0.01, ZB - 0.8 - LEDGE - 0.01)]), -IX, IX),            # 받침판 위아래 가장자리를 받치는 턱 (밑은 45도)
           prism_xy_to_yz(poly([(-hb - 0.01, ZB), (-hb + LEDGE, ZB), (-hb + LEDGE, ZB - 0.8), (-hb - 0.01, ZB - 0.8 - LEDGE - 0.01)]), -IX, IX)]
    for sx in (-1, 1):
        add += [box(min(sx * hb, sx * IX), max(sx * hb, sx * IX), -7.0, 7.0, ZF - 0.01, 0.0)]                                        # 옆 막이 (받침판의 좌우 자리, 옆 나사 자리. 바닥까지)
        for y in (-48.0, 48.0):                                                                                                     # 반사판의 좌우 자리 (선이 지나는 틈은 남긴다)
            add.append(box(min(sx * hb, sx * (hb + 1.6)), max(sx * hb, sx * (hb + 1.6)), y - 6.0, y + 6.0, ZF - 0.01, -RG + 0.5))
    add += [cz(3.0, ZF - 0.01, ZR, x, y) for x, y in LEG]                                                                          # 반사판 기둥
    add += [cz(BOSS_R, ZF - 0.01, 0.0, x, y) for x, y in BOSS[:4]]
    add += [cz(4.2, ZF - 0.01, ZF + 1.2, x, y) for x, y in MOUNT]                                                                   # 걸이 나사 자리 둘레
    xc, yc = XIAO_C; hw = XIAO[1] / 2; xa = xc - XIAO[0] / 2 - 2.0
    add += [box(xa, IX, yc - hw - 2.0, yc - hw + 1.2, ZF - 0.01, XIAO_Z), box(xa, IX, yc + hw - 1.2, yc + hw + 2.0, ZF - 0.01, XIAO_Z),          # XIAO 받침 두 줄
            box(xa, IX, yc - hw - 2.0, yc - hw - 0.25, XIAO_Z - 0.01, XIAO_Z + 2.4), box(xa, IX, yc + hw + 0.25, yc + hw + 2.0, XIAO_Z - 0.01, XIAO_Z + 2.4),
            box(xa, xa + 1.4, yc - hw - 2.0, yc + hw + 2.0, ZF - 0.01, XIAO_Z + 2.4),
            cz(2.6, ZF - 0.01, CLIP_Z, CLIP_X, yc - hw - 4.4), cz(2.6, ZF - 0.01, CLIP_Z, CLIP_X, yc + hw + 4.4)]                    # 누름쇠 나사 기둥 (M2)
    cx_, cy_ = CELL_C
    add += [box(cx_ - CELL[0] / 2 - 1.6, cx_ - CELL[0] / 2 - 0.3, cy_ - 8, cy_ + 8, ZF - 0.01, ZF + 6.0), box(cx_ + CELL[0] / 2 + 0.3, cx_ + CELL[0] / 2 + 1.6, cy_ - 8, cy_ + 8, ZF - 0.01, ZF + 6.0),
            box(cx_ - 20, cx_ + 20, cy_ + CELL[1] / 2 + 0.3, cy_ + CELL[1] / 2 + 1.6, ZF - 0.01, ZF + 6.0)]                         # 배터리 홀더 울타리
    for x in (-23.5, -7.0, 7.0, 23.5):                                                                                              # 묶음끈 고리 (모듈 줄 사이, SMA 단자 사이)
        for y in (-23.0, 23.0):
            add.append(box(x - 3.0, x + 3.0, y - 6.0, y + 6.0, ZF - 0.01, ZF + 4.6) - box(x - 4, x + 4, y - 3.6, y + 3.6, ZF, ZF + 3.0))
    m = union([body] + add)
    cut = [cz(1.3, -11.0, 1.0, x, y) for x, y in BOSS[:4]]                                                                          # 앞덮개 나사 안내 구멍 (지름 2.6)
    cut += [cz(1.3, ZB - 11.0, 1.0, x, y) for x, y in BOSS[4:]]
    cut += [cz(1.7, ZO - 1, ZF + 2, x, y) for x, y in MOUNT]                                                                        # 걸이 나사 구멍 (지름 3.4)
    cut += [cz(0.85, CLIP_Z - 7.0, CLIP_Z + 1, CLIP_X, yc - hw - 4.4), cz(0.85, CLIP_Z - 7.0, CLIP_Z + 1, CLIP_X, yc + hw + 4.4)]    # M2 안내 구멍
    zu = XIAO_Z + XIAO[2] + 1.6
    cut.append(prism_xy_to_yz(rrect(yc - USB_C[0] / 2, yc + USB_C[0] / 2, zu - USB_C[1] / 2, zu + USB_C[1] / 2, 1.2), IX - 1.0, OX + 1.0))        # USB-C 창 (오른쪽 벽)
    cut += [box(-30.0, 30.0, -hb - DIV - 0.1, -hb + 0.1, ZF, ZR - 4.0), box(-6.0, 6.0, hb - 0.1, hb + DIV + 0.1, ZB - 6.0, ZB - 3.0 - 0.01)]   # 선이 지나는 굴 (아래 칸 ↔ 모듈 칸, 위 띠 ↔ 받침판 뒤)
    cut.append(box(-3.0, 3.0, OY1 - 1.0, OY1 + 1, -0.6, 0.1))                                                                       # 위쪽 표시 홈
    return m - union(cut)


def bezel():
    """2. 앞덮개. 전파가 지나는 곳은 1.2mm 살이고, 테두리 턱이 받침판 가장자리를 눌러 안테나 앞에 3mm 빈틈을 만든다. 출력: 겉면을 바닥에, 받침 없이."""
    skin = extrude_z(_outline(), AIR, ZC)
    hb = AP / 2 + GAPB; win = AP / 2 - LEDGE
    frame = extrude_z(_outline(), 0.0, AIR + 0.01) - union([box(-win, win, -win, win, -1, AIR + 1),
                                                             extrude_z(rrect(-IX + 0.4, IX - 0.4, hb + DIV + 0.4, Y1 - 0.4, 3.0), -1, AIR + 1),      # 위 띠 (기준 안테나를 덮개 안쪽에 붙인다)
                                                             extrude_z(rrect(-IX + 0.4, IX - 0.4, Y0 + 0.4, -hb - DIV - 0.4, 3.0), -1, AIR + 1)])    # 아래 칸
    post = [cz(BOSS_R + 0.4, 0.0, AIR + 0.01, x, y) for x, y in BOSS]
    m = union([skin, frame] + post)
    cut = [cz(1.7, -1, ZC + 1, x, y) for x, y in BOSS] + [cz(3.1, ZC - 1.8, ZC + 1, x, y) for x, y in BOSS]
    mx, my = MARK_C; hx, hy = MARK[0] / 2, MARK[1] / 2
    for (x0, x1, y0, y1) in ((-hx, hx, hy, hy + 0.6), (-hx, hx, -hy - 0.6, -hy), (-hx - 0.6, -hx, -hy - 0.6, hy + 0.6), (hx, hx + 0.6, -hy - 0.6, hy + 0.6)):
        cut.append(box(mx + x0, mx + x1, my + y0, my + y1, ZC - 0.3, ZC + 1))
    cut.append(box(-3.0, 3.0, OY1 - 1.0, OY1 + 1, -0.1, 1.0))
    return m - union(cut)


def film_centers():
    c = (np.arange(4) - 1.5) * PITCH
    return [(x, y) for y in c[::-1] for x in c]                     # 앞에서 볼 때 왼쪽 위부터 줄마다 (firmware 의 안테나 번호 차례)


def _film_cs(grow=0.0):
    L, W = FILM[0] / 2 + grow, FILM[1] / 2 + grow; a = np.radians(FILM_ANG); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-np.sin(a), np.cos(a)])
    return poly([L * u + W * v, -L * u + W * v, -L * u - W * v, L * u - W * v])


def film_feed(x, y, d=None):
    """안테나 선이 나오는 끝 (안테나 긴 축의 +쪽 끝에서 조금 바깥). 받침판의 선 구멍 자리"""
    d = FILM[0] / 2 + 2.0 if d is None else d; a = np.radians(FILM_ANG)
    return x + d * np.cos(a), y + d * np.sin(a)


def ant_plate():
    """9. 안테나 받침판. 앞면의 표시선 16개 안에 필름 안테나를 붙이고(양면 접착), 선은 끝의 구멍으로 뒤로 넘긴다.
    뒷면의 다리 넷이 반사판을 눌러 안테나 면과 반사판의 거리(RG)를 잡는다. 출력: 앞면을 바닥에 (다리가 위로), 받침 없이."""
    h = AP / 2
    m = union([box(-h, h, -h, h, ZB, 0.0)] + [cz(3.0, -RG, ZB + 0.01, x, y) for x, y in LEG])
    cut = [extrude_z((_film_cs(0.6) - _film_cs(0.0)).translate([x, y]), -0.4, 1.0) for x, y in film_centers()]                     # 안테나 자리 표시선 (폭 0.6, 깊이 0.4)
    cut += [cz(2.6, ZB - 1, 1.0, *film_feed(x, y)) for x, y in film_centers()]                                                     # 선 구멍 (U.FL 플러그가 지난다)
    cut += [box(-3.0, 3.0, h - 1.2, h + 1, -0.4, 1.0)]                                                                              # 위쪽 표시 홈
    return m - union(cut)


def mount_plate():
    """3. 자석 걸이판. 통 뒷면에 M3 x 8 나사 4개(통 안에서 조이고, 너트는 이 판에). 자석 4개로 칠판·철문·사물함에 붙고, 위 고리로 걸이에 건다.
    좌표: 통에 닿는 면 z = ZO, 벽에 닿는 면 z = ZO - MT_T. 출력: 통에 닿는 면을 바닥에 (자석 자리가 위로 열린다)."""
    h = MT_W / 2; z1, z0 = ZO, ZO - MT_T
    plate = extrude_z(rrect(-h, h, -h, h, 8.0), z0, z1)
    tab = extrude_z(rrect(-TAB[0] / 2, TAB[0] / 2, h - 6.0, h + TAB[1], 6.0), z0, z1)
    m = union([plate, tab])
    cut = [cz(1.7, z0 - 1, z1 + 1, x, y) for x, y in MOUNT]
    cut += [extrude_z(hex_cs(M3N[0] + 0.3), z0 - 1, z0 + M3N[1] + 0.4).translate([x, y, 0]) for x, y in MOUNT]                       # 너트 자리 (벽 쪽에서 넣는다)
    cut += [cz(MAG_D / 2 + 0.2, z0 - 1, z0 + MAG_T + 0.2, x, y) for x, y in MAGS]                                                    # 자석 자리
    cut += [cz(1.3, z0 - 1, z1 + 1, x, y) for x, y in MAGS]                                                                          # 자석 나사(M3 접시머리) 안내 구멍
    cut.append(cz(TAB[2], z0 - 1, z1 + 1, 0.0, h + TAB[1] - 8.0))                                                                    # 고리 구멍
    return m - union(cut)


def foot():
    """4. 받침 발. 통을 발 위에 얹고 뒷면 아래쪽 구멍 2개에 조인다 (M3 x 8, 너트는 이 판에). 창턱·교탁·사물함 위에 세우고, 발 밑의 1/4인치 너트로 삼각대에도 올린다.
    좌표: 패널 좌표. ㄴ자 한 단면을 x 로 민 꼴이라 옆으로 눕혀 받침 없이 뽑힌다."""
    hw = FT_W / 2; z1, z0 = ZO, ZO - FT_T; yb = OY0 - FT_GAP                                                                         # 발 윗면 (통 밑면보다 조금 아래)
    prof = [(yb - FT_T, FT_FRONT), (yb, FT_FRONT), (yb, z1), (FT_TOP, z1), (FT_TOP, z0), (yb + FT_BLOCK + 12.0, z0), (yb + FT_BLOCK, z0 - 12.0),
            (yb + FT_BLOCK, z0 - 24.0), (yb, z0 - 24.0), (yb, z0 - FT_D), (yb - FT_T, z0 - FT_D)]                                     # (y, z)
    m = prism_xy_to_yz(poly(prof), -hw, hw)
    cut = [cz(1.7, z0 - 1, z1 + 1, x, y) for x, y in MOUNT[:2]]
    cut += [extrude_z(hex_cs(M3N[0] + 0.3), z0 - 1, z0 + M3N[1] + 0.4).translate([x, y, 0]) for x, y in MOUNT[:2]]
    zc = z0 - 12.0
    cut.append(cy(NUT14[2] / 2, yb - FT_T - 1, yb + FT_BLOCK + 1, 0.0, zc))                                                          # 삼각대 나사 구멍
    cut.append(prism_xz(hex_cs(NUT14[0] + 0.3), yb - FT_T + 1.6, yb + FT_BLOCK + 1).translate([0, 0, zc]))                           # 1/4인치 너트 자리 (위에서 넣는다)
    return m - union(cut)


def xiao_clip():
    """5. ESP32 누름쇠. XIAO 의 USB-C 단자 위를 가로질러 눌러 M2 x 8 나사 2개로 (핀 헤더를 위로 납땜해도 걸리지 않는다). 출력: 납작한 면을 바닥에."""
    xc, yc = XIAO_C; hw = XIAO[1] / 2; z0 = CLIP_Z
    m = extrude_z(rrect(CLIP_X - 3.0, CLIP_X + 3.0, yc - hw - 7.0, yc + hw + 7.0, 2.0), z0, z0 + 2.0)
    m += box(CLIP_X - 3.0, CLIP_X + 3.0, yc - 4.2, yc + 4.2, XIAO_Z + XIAO[2] + USB_H + 0.15, z0 + 0.01)                              # USB-C 단자를 누르는 턱
    return m - union([cz(1.15, z0 - 3, z0 + 3, CLIP_X, yc - hw - 4.4), cz(1.15, z0 - 3, z0 + 3, CLIP_X, yc + hw + 4.4)])


# ---- 허브 ----
def hub_box():
    """6. 허브 통: 아래에 보조 배터리, 그 위 선반에 파이와 동글 둘. 출력: 그대로."""
    ox, oy = HL / 2 + HWALL, HW_ / 2 + HWALL; zt = HFLOOR + HH
    m = extrude_z(rrect(-ox, ox, -oy, oy, 6.0), 0.0, zt) - extrude_z(rrect(-HL / 2, HL / 2, -HW_ / 2, HW_ / 2, 3.6), HFLOOR, zt + 1)
    add = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            xa, xb = sorted((sx * (HL / 2 - 6.8), sx * HL / 2)); ya, yb = sorted((sy * (HW_ / 2 - 6.8), sy * HW_ / 2))
            add.append(box(xa, xb, ya, yb, HFLOOR - 0.01, zt))                                                                        # 뚜껑 나사 기둥 (네 귀)
    for sx in (-1, 1):                                                                                                               # 선반 받침 턱 (양 끝 벽)
        add.append(box(sx * HL / 2 - (3.0 if sx > 0 else 0), sx * HL / 2 + (0 if sx > 0 else 3.0), -HW_ / 2 + 8, HW_ / 2 - 8, HFLOOR - 0.01, DECK_Z))
    m = union([m] + add)
    cut = [cz(1.3, zt - 12, zt + 1, sx * (HL / 2 - 3.4), sy * (HW_ / 2 - 3.4)) for sx in (-1, 1) for sy in (-1, 1)]
    cut.append(box(ox - HWALL - 1, ox + 1, -9.0, 9.0, DECK_Z + DECK_T + 6.0, zt + 1))                                                # 선이 나가는 홈 (USB 쪽 끝 벽)
    cut.append(box(-ox - 1, -ox + HWALL + 1, -9.0, 9.0, zt - 10.0, zt + 1))                                                          # 반대쪽 끝: 배터리 선
    for k in range(7):                                                                                                               # 옆 숨구멍
        x = -45 + 15 * k
        cut += [box(x - 1.5, x + 1.5, oy - HWALL - 1, oy + 1, DECK_Z + 8, zt - 8), box(x - 1.5, x + 1.5, -oy - 1, -oy + HWALL + 1, DECK_Z + 8, zt - 8)]
    return m - union(cut)


def hub_deck():
    """7. 허브 선반: 배터리 위에 얹고 파이를 M2.5 나사 4개로. 출력: 그대로 (기둥이 위로)."""
    z0 = DECK_Z; m = box(-HL / 2 + 0.4, HL / 2 - 0.4, -HW_ / 2 + 0.4, HW_ / 2 - 0.4, z0, z0 + DECK_T)
    post = [cz(3.0, z0 + DECK_T - 0.01, z0 + DECK_T + PI_STAND, PI_X0 + a, -PI[1] / 2 + b) for a, b in PI_HOLES]
    m = union([m] + post)
    cut = [cz(1.05, z0 - 1, z0 + DECK_T + PI_STAND + 1, PI_X0 + a, -PI[1] / 2 + b) for a, b in PI_HOLES]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(box(sx * HL / 2 - 8 if sx > 0 else -HL / 2 - 1, HL / 2 + 1 if sx > 0 else -HL / 2 + 8, sy * HW_ / 2 - 8 if sy > 0 else -HW_ / 2 - 1, HW_ / 2 + 1 if sy > 0 else -HW_ / 2 + 8, z0 - 1, z0 + 5))   # 네 귀 (나사 기둥을 피한다)
    cut.append(box(-HL / 2 - 1, -HL / 2 + 14, -11, 11, z0 - 1, z0 + 5))                                                              # 배터리 선이 올라오는 자리
    cut += [box(-30 + 20 * k - 6, -30 + 20 * k + 6, -18, 18, z0 - 1, z0 + 5) for k in range(4)]                                       # 바람 구멍
    return m - union(cut)


def hub_lid():
    """8. 허브 뚜껑: M3 x 10 나사 4개. 출력: 겉면을 바닥에."""
    ox, oy = HL / 2 + HWALL, HW_ / 2 + HWALL; zt = HFLOOR + HH
    m = extrude_z(rrect(-ox, ox, -oy, oy, 6.0), zt, zt + 2.4)
    lip = extrude_z(rrect(-HL / 2 + 0.3, HL / 2 - 0.3, -HW_ / 2 + 0.3, HW_ / 2 - 0.3, 3.4), zt - 3.0, zt + 0.01) - extrude_z(rrect(-HL / 2 + 2.0, HL / 2 - 2.0, -HW_ / 2 + 2.0, HW_ / 2 - 2.0, 2.0), zt - 4, zt + 1)
    lipcut = [box(sx * HL / 2 - 8 if sx > 0 else -HL / 2 - 1, HL / 2 + 1 if sx > 0 else -HL / 2 + 8, sy * HW_ / 2 - 8 if sy > 0 else -HW_ / 2 - 1, HW_ / 2 + 1 if sy > 0 else -HW_ / 2 + 8, zt - 4, zt + 0.02) for sx in (-1, 1) for sy in (-1, 1)]
    lipcut += [box(HL / 2 - 3, HL / 2 + 1, -10, 10, zt - 4, zt + 0.02), box(-HL / 2 - 1, -HL / 2 + 3, -10, 10, zt - 4, zt + 0.02)]
    m = union([m, lip - union(lipcut)])
    cut = [cz(1.7, zt - 1, zt + 4, sx * (HL / 2 - 3.4), sy * (HW_ / 2 - 3.4)) for sx in (-1, 1) for sy in (-1, 1)]
    cut += [cz(3.1, zt + 0.8, zt + 4, sx * (HL / 2 - 3.4), sy * (HW_ / 2 - 3.4)) for sx in (-1, 1) for sy in (-1, 1)]
    cut += [box(-48 + 16 * k - 1.5, -48 + 16 * k + 1.5, -24, 24, zt - 1, zt + 4) for k in range(7)]                                   # 숨구멍
    return m - union(cut)



# ---- 회전 받침 (숙소) ----
MOTOR_Z = -0.2                 # 모터 윗면(귀)의 높이: 덱 윗면의 귀 홈(깊이 1)에 얹어 베어링 밑으로 숨긴다
HUB_AF, HUB_Z0, HUB_IN = 14.0, 1.8, 5.0          # 축 이음쇠(12): 육각 맞변, 밑면 높이(모터 보스 위), 돌림판 육각 홈에 들어가는 깊이


def _motor_cut():
    o = MOTOR["off"]; r = MOTOR["body_d"] / 2 + 0.8; z1 = MOTOR_Z; z0 = z1 - MOTOR["body_h"] - 2.0
    cw, cl, ch = MOTOR["cover"]
    return union([cz(r, z0, z1 + 1.0, 0.0, -o), box(-cw / 2 - 0.6, cw / 2 + 0.6, -o - r - cl + 4.0, -o, z1 - ch - 1.0, z1 + 1.0)])


def turn_base():
    """10. 회전 받침 받침대. 윗면에 베어링 아래 판을 나사 넷으로, 가운데 받침 둘 위에 28BYJ-48 을 귀 나사 둘로 얹는다 (몸통은 칸 속, 축이 가운데로 올라온다).
    밑면 가운데의 1/4인치 너트로 삼각대에 올린다. 선은 옆 홈으로. 출력: 밑면을 바닥에 (모터 칸이 위로 열린다), 받침 없이."""
    o = MOTOR["off"]
    m = cz(TB_R, -TB_H, 0.0)
    cut = [_motor_cut()]                                                                                                            # 모터 칸 (위에서 넣는다)
    ea = MOTOR["ear"] / 2 + 4.1
    cut.append(box(-ea, ea, -o - MOTOR["ear_w"] / 2 - 0.6, -o + MOTOR["ear_w"] / 2 + 0.6, MOTOR_Z - 0.8, 1.0))                    # 귀 홈: 모터 윗면이 베어링 밑(z 0)보다 낮게
    cut += [cz(1.3, -6.0, 1.0, x, y) for x, y in BEAR_LO]                                                                           # 베어링 나사 안내 구멍 (M3 x 8 또는 나무나사)
    cut += [cz(1.3, MOTOR_Z - 8.0, MOTOR_Z + 1.0, s * MOTOR["ear"] / 2, -o) for s in (-1, 1)]                                      # 모터 귀 나사 (위에서 M3 x 8)
    zm = MOTOR_Z - MOTOR["body_h"] - 2.0                                                                                            # 모터 칸 밑
    cut.append(box(-6.0, 6.0, -TB_R - 1, -o - 10.0, zm, zm + 8.0))                                                                  # 선이 나가는 홈 (드라이버 ULN2003 은 밖에)
    zn = -TB_H
    cut.append(extrude_z(hex_cs(NUT14[0] + 0.3), zn - 1, zn + NUT14[1] + 0.4))                                                      # 1/4인치 너트 자리 (밑에서 넣는다)
    cut.append(cz(NUT14[2] / 2, zn - 1, zm + 0.01))
    return m - union(cut)


def turn_hub():
    """12. 축 이음쇠. 모터 축(D)을 받아 베어링 가운데 구멍을 지나 돌림판의 육각 홈에 끼워진다. 모터가 베어링 밑에 숨으므로 필요하다.
    출력: 세워서, 받침 없이."""
    z1 = BEAR[1] + HUB_IN - 0.3; zs = MOTOR_Z + MOTOR["boss_h"] + MOTOR["shaft"]
    return extrude_z(hex_cs(HUB_AF), HUB_Z0, z1) - extrude_z(d_bore_cs(), HUB_Z0 - 1, zs + 0.7)


def _wedge_frame():
    """받침 발의 밑면이 닿는 기울어진 면: (y 시작, y 끝, 높이 시작). 발 밑면의 앞뒤를 회전 받침 y 로 옮긴다"""
    yb = (ZO - FT_T - FT_D) - TT_AXIS                                # 발 뒤끝
    ye = min(FT_FRONT - TT_AXIS, yb + 105.0)                          # 발 앞쪽 일부만 받친다 (가볍게)
    return yb, ye, BEAR[1] + TT_T + 3.0


def turn_top():
    """11. 회전 받침 돌림판. 베어링 위 판에 나사 넷, 밑면 가운데의 D 구멍이 모터 축에 끼워진다. 윗면은 20도 기운 받침이고,
    받침 발을 얹어 1/4인치 볼트(길이 50mm, 밑에서, 머리는 돌림판 밑에 숨는다)로 발의 너트에 조인다. 출력: 밑면을 바닥에."""
    z0 = BEAR[1]; z1 = z0 + TT_T; t = np.tan(np.radians(TILT)); c, sn = np.cos(np.radians(TILT)), np.sin(np.radians(TILT))
    disk = cz(TT_R, z0, z1)
    yb, ye, hb = _wedge_frame()
    wedge = prism_xy_to_yz(poly([(yb, z0), (ye, z0), (ye, hb + t * (ye - yb)), (yb, hb)]), -FT_W / 2, FT_W / 2)                         # 돌림판 밖으로 나간 곳도 바닥까지 (받침 없이 뽑힌다)
    lip = prism_xy_to_yz(poly([(yb - 4.0, z0), (yb + 0.01, z0), (yb + 0.01, hb + 4.0), (yb - 4.0, hb + 4.0)]), -FT_W / 2, FT_W / 2)   # 발 뒤끝을 막는 턱
    m = union([disk, wedge, lip])
    cut = [extrude_z(hex_cs(HUB_AF + 0.3), z0 - 1.0, z0 + HUB_IN)]                                                                  # 축 이음쇠(12)의 육각 홈 (밑에서 끼운다)
    cut += [cz(1.3, z0 - 1, z0 + 6.0, x, y) for x, y in BEAR_HI]                                                                    # 베어링 위 판 나사 안내 구멍
    yn = (ZO - FT_T - 12.0) - TT_AXIS                                 # 발의 너트 자리 (발 밑판 밑면에서 12 위의 패널 z)
    zs = hb + t * (yn - yb)                                           # 그 자리의 기운 면 높이
    ax = np.array([0.0, -sn, c])                                      # 기운 면의 수직 방향 (z 를 x 축으로 TILT 돌린 것)
    L = (zs - z0) / c; pb = np.array([0.0, yn, zs]) - ax * L          # 볼트 축이 돌림판 밑면(z0)을 지나는 점
    bolt = cz(NUT14[2] / 2 + 0.2, -5.0, L + 5.0).rotate([TILT, 0, 0]).translate(list(pb))
    head = extrude_z(hex_cs(BOLT14[0] + 0.4), -10.0, BOLT14[1] + 3.6).rotate([TILT, 0, 0]).translate(list(pb))                      # 머리 자리 (밑에서 3mm 더 깊게: 기운 머리가 돌림판 밑으로 나오지 않게)
    cut += [bolt, head]
    return m - union(cut)


# ---- 사 오는 부품의 모형 (그림·영상·검사용) ----
def reflector():
    h = AP / 2
    return box(-h, h, -h, h, -RG - REF_T, -RG)


def films():
    return union([extrude_z(_film_cs().translate([x, y]), -0.2, FILM[2] - 0.2) for x, y in film_centers()])


def xiao():
    xc, yc = XIAO_C; hl, hw = XIAO[0] / 2, XIAO[1] / 2; zt = XIAO_Z + XIAO[2]
    m = box(xc - hl, xc + hl, yc - hw, yc + hw, XIAO_Z, zt)
    m += box(xc + hl - 7.3, xc + hl + 1.0, yc - 4.5, yc + 4.5, zt, zt + USB_H)                                                        # USB-C (오른쪽 벽 쪽)
    m += box(xc - hl + 1.0, xc + hl - 8.5, yc - 5.5, yc + 5.5, zt, zt + 2.2)                                                         # 가림쇠
    return m


def cell():
    cx_, cy_ = CELL_C
    return box(cx_ - CELL[0] / 2, cx_ + CELL[0] / 2, cy_ - CELL[1] / 2, cy_ + CELL[1] / 2, ZF, ZF + 3.0) + cx(9.2, cx_ - 32.5, cx_ + 32.5, cy_, ZF + 10.4)


def rf_modules():
    """RF 모듈 여덟: 16갈래 스위치 다섯, 위상 스위치 둘, 합침기. 반사판 뒤 칸에 묶음끈으로"""
    out = []
    for (x, y, k) in MODS:
        s = SP4T if k == "s" else COMB
        out.append(box(x - s[0] / 2, x + s[0] / 2, y - s[1] / 2, y + s[1] / 2, ZF + 3.2, ZF + 4.8))
        out += [cy(3.1, y + s[1] / 2 - 0.01, y + s[1] / 2 + 6.0, x + dx, ZF + 4.0 + 3.1) for dx in ((-14, 0, 14) if k == "s" else (-9, 9))]
        out += [cy(3.1, y - s[1] / 2 - 6.0, y - s[1] / 2 + 0.01, x + dx, ZF + 4.0 + 3.1) for dx in ((-14, 0, 14) if k == "s" else (0,))]
    return union(out)


def ref_antenna():
    """기준 안테나: 같은 두 대역 필름 안테나를 앞덮개 위 띠 안쪽에 붙인다"""
    return box(-FILM[0] / 2, FILM[0] / 2, AP / 2 + 8.0, AP / 2 + 8.0 + FILM[1], AIR - 0.25, AIR - 0.05)


def magnets():
    z0 = ZO - MT_T
    return union([cz(MAG_D / 2, z0, z0 + MAG_T, x, y) - cz(1.8, z0 - 1, z0 + MAG_T + 1, x, y) for x, y in MAGS])


def pi_board():
    z = DECK_Z + DECK_T + PI_STAND; y0 = -PI[1] / 2
    m = box(PI_X0, PI_X0 + PI[0], y0, y0 + PI[1], z, z + PI[2])
    m += box(PI_X0 + PI[0] - 17.5, PI_X0 + PI[0] + 2.0, y0 + 2.0, y0 + 17.0, z + PI[2], z + PI[2] + 16.0)                              # USB 2단
    m += box(PI_X0 + PI[0] - 17.5, PI_X0 + PI[0] + 2.0, y0 + 20.5, y0 + 35.5, z + PI[2], z + PI[2] + 16.0)
    m += box(PI_X0 + PI[0] - 21.0, PI_X0 + PI[0] + 2.0, y0 + 38.5, y0 + 54.5, z + PI[2], z + PI[2] + 13.5)                             # 랜
    m += box(PI_X0 + 7.0, PI_X0 + 58.0, y0 + 49.5, y0 + 54.5, z + PI[2], z + PI[2] + 8.5)                                             # GPIO 핀
    return m


def dongles():
    z = DECK_Z + DECK_T + PI_STAND + PI[2]; y0 = -PI[1] / 2; x0 = PI_X0 + PI[0] + 2.0
    return union([box(x0, x0 + DONGLE[0] - DONGLE[3], y0 + yc - DONGLE[1] / 2, y0 + yc + DONGLE[1] / 2, z + zz - DONGLE[2] / 2, z + zz + DONGLE[2] / 2) for yc, zz in ((9.5, 11.5), (28.0, 11.5))])


def battery():
    return extrude_z(rrect(-BATT[0] / 2 + 3, BATT[0] / 2 - 3, -BATT[1] / 2 + 2, BATT[1] / 2 - 2, 6.0), HFLOOR, HFLOOR + BATT[2] - 1.0)


PRINTED = "#27348B"; LIGHT = "#8A9BC4"; PCB = "#1F6B47"; GOLD = "#C9A227"; STEEL = "#C3C8CF"; DARK = "#3A3F47"; HOT = "#EB6834"



def panel_assembly(mount="plate"):
    """이름 -> (형상, 묶음, 색, 뽑는 부품인가)"""
    A = {"tray": (tray(), "panel", PRINTED, True), "bezel": (bezel(), "panel", LIGHT, True), "ant_plate": (ant_plate(), "panel", "#4A5BA8", True),
         "reflector": (reflector(), "hw", STEEL, False), "films": (films(), "hw", GOLD, False),
         "xiao": (xiao(), "hw", DARK, False), "xiao_clip": (xiao_clip(), "panel", HOT, True), "cell": (cell(), "hw", STEEL, False), "rf": (rf_modules(), "hw", STEEL, False),
         "ref": (ref_antenna(), "hw", GOLD, False)}
    if mount == "plate":
        A["mount"] = (mount_plate(), "panel", HOT, True); A["magnets"] = (magnets(), "hw", STEEL, False)
    elif mount == "foot":
        A["foot"] = (foot(), "panel", HOT, True)
    return A


def motor():
    o = MOTOR["off"]; z1 = MOTOR_Z
    m = cz(MOTOR["body_d"] / 2, z1 - MOTOR["body_h"], z1, 0.0, -o)
    m += box(-MOTOR["ear"] / 2 - 3.5, MOTOR["ear"] / 2 + 3.5, -o - MOTOR["ear_w"] / 2, -o + MOTOR["ear_w"] / 2, z1 - 0.8, z1)
    m += cz(MOTOR["boss_d"] / 2, z1, z1 + MOTOR["boss_h"]) + cz(2.5, z1 + MOTOR["boss_h"], z1 + MOTOR["boss_h"] + MOTOR["shaft"])
    cw, cl, ch = MOTOR["cover"]
    return m + box(-cw / 2, cw / 2, -o - MOTOR["body_d"] / 2 - cl + 4.0, -o, z1 - ch, z1 - 0.5)


def bearing():
    h = BEAR[0] / 2
    return box(-h, h, -h, h, 0.0, BEAR[1]) - cz(22.0, -1, BEAR[1] + 1)


def panel_on_turntable():
    """패널(받침 발)을 돌림판의 기운 면에 얹은 자리로 옮긴다: 패널 좌표 -> 회전 받침 좌표"""
    yb, ye, hb = _wedge_frame()
    def place(m):
        m = m.rotate([90, 0, 0]).rotate([0, 0, 180])                  # 패널 y(위) -> z, 패널 z(앞) -> y
        m = m.translate([0.0, -TT_AXIS, -(OY0 - FT_GAP - FT_T)])        # 발 밑면을 z = 0 으로, 축을 y = 0 으로
        m = m.translate([0.0, -yb, 0.0]).rotate([TILT, 0, 0]).translate([0.0, yb, hb])
        return m
    return place


def turntable_assembly():
    place = panel_on_turntable()
    A = {k: (place(v[0]), v[1], v[2], v[3]) for k, v in panel_assembly("foot").items()}
    A.update({"turn_base": (turn_base(), "turn", PRINTED, True), "turn_top": (turn_top(), "turn", HOT, True), "turn_hub": (turn_hub(), "turn", LIGHT, True), "motor": (motor(), "hw", DARK, False), "bearing": (bearing(), "hw", STEEL, False)})
    return A


def hub_assembly():
    return {"hub_box": (hub_box(), "hub", PRINTED, True), "hub_deck": (hub_deck(), "hub", HOT, True), "hub_lid": (hub_lid(), "hub", LIGHT, True),
            "battery": (battery(), "hw", DARK, False), "pi": (pi_board(), "hw", PCB, False), "dongles": (dongles(), "hw", STEEL, False)}


def marker_corners():
    """AR 마커 둘의 귀 (패널 좌표, mm). 차례: 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 (앞에서 볼 때). 번호 0 = 왼쪽, 1 = 오른쪽"""
    mx, my = MARK_C; h = MARK_SIZE / 2; z = ZC + 0.1
    return {i: [[cx_ - h, my + h, z], [cx_ + h, my + h, z], [cx_ + h, my - h, z], [cx_ - h, my - h, z]] for i, cx_ in ((0, mx - MARK_DX), (1, mx + MARK_DX))}
