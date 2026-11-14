"""DeteCam 2축 수신기 v1 - 부품 모델.
모든 부품은 조립 좌표로 만든다.
  고정 좌표 F : 원점 = 안테나, X 앞, Y 왼쪽, Z 위
  좌우 회전 P : F를 Z축 둘레로 pan 만큼 돌린 좌표 (베일, 모터2)
  위아래 회전 T: P를 Y축 둘레로 tilt 만큼 돌린 좌표 (포크, 기어 휠, 판)
"""
import numpy as np
from cadlib import *
from params import *

# ----------------------------------------------------------------- 공통
def pillar_sec(off=0.0):
    c = P_CORNER
    s = poly([(PX0 + c, PY0), (PX1 - c, PY0), (PX1, PY0 + c), (PX1, PY1 - c),
              (PX1 - c, PY1), (PX0 + c, PY1), (PX0, PY1 - c), (PX0, PY0 + c)]) ^ CS.circle(R_KEEP)
    return s.offset(off, JoinType.Miter) if off else s


def pi_hole_xz():
    """파이 고정 구멍의 (X, Z)"""
    return [(BX0 + b, BZ1 - a) for a, b in PI_HOLES]


EAR_SCREWS_SHELL = [(-62.0, -6.0), (-62.0, 2.5), (24.0, -6.0), (24.0, 2.5)]
EAR_SCREWS_LID = [(-60.0, -21.0), (34.0, -21.0)]      # 앞쪽 나사는 머리(반지름 36) 밖으로 빼서 조립 뒤에도 드라이버가 들어간다
SEAT_SCREW = (-50.2, -3.0)
CABLE_HOLE = (-27.0, -15.0, -20.0, -10.0)          # x0, x1, y0, y1 (기둥 윗면)
TRAY_SCREW_ANGLES = (45.0, 135.0, 225.0, 315.0)
SNAP_PTS = [(PX1 - WALL - 0.2, -5.0), (PX1 - WALL - 0.2, -70.0),
            (PX0 + WALL + 0.2, -5.0), (PX0 + WALL + 0.2, -70.0)]     # (X, Z) 딸깍 돌기


# ----------------------------------------------------------------- 고정부
def _pillar_body():
    outer = extrude_z(pillar_sec(), PZ0, PZ1)
    inner = extrude_z(pillar_sec(-WALL), PZ0 - 1, PZ1 - WALL)
    return outer - inner


def _ears(y0, y1, front_x1=30.0):
    back = box(-67.0, -44.0, y0, y1, PZ0, PZ0 + 3.0)     # 깎인 모서리에도 붙도록 기둥 안쪽까지 겹친 뒤 속을 잘라 낸다
    front = box(8.0, front_x1, y0, y1, PZ0, PZ0 + 3.0)
    return ((back + front) - extrude_z(pillar_sec(-0.5), PZ0 - 1, PZ0 + 4)) ^ cz(EAR_CLIP_R, PZ0 - 1, PZ0 + 4)


def pillar_shell():
    """기둥 몸통: 파이를 세워 고정하는 쪽 (+Y). 출력: +Y 면을 바닥에."""
    body = _pillar_body() ^ box(-200, 200, Y_PART, 200, -300, 300)
    add = []
    for x, z in pi_hole_xz():                            # 파이 받침 기둥 + 꽂는 못
        add.append(cy(3.0, PI_T / 2, PY1 - WALL + 0.2, x, z))
        add.append(cy(1.25, -3.0, PI_T / 2 + 0.1, x, z))
    sx, sy = SEAT_SCREW                                  # 모터 자리 고정 나사 기둥 (벽까지 이어서 출력 가능하게)
    add.append(cz(3.4, 1.5, PZ1 - WALL + 0.2, sx, sy))            # 아래 ㄱ자 플러그와 3mm 띄운다
    add.append(box(sx - 3.4, sx + 3.4, sy, PY1 - WALL + 0.2, 1.5, PZ1 - WALL + 0.2))
    # 뚜껑 맞춤 턱: 벽에 붙은 부분 + 뚜껑 안으로 들어가는 부분
    zr = (PZ0 + 6.0, PZ1 - WALL - 0.3)
    lip_a = extrude_z(pillar_sec(-WALL + 0.2), *zr) - extrude_z(pillar_sec(-WALL - 1.4), zr[0] - 1, zr[1] + 1)
    lip_b = extrude_z(pillar_sec(-WALL - 0.2), *zr) - extrude_z(pillar_sec(-WALL - 1.4), zr[0] - 1, zr[1] + 1)
    add.append(lip_a ^ box(-200, 200, Y_PART, Y_PART + 1.6, -300, 300))
    for k in range(7):                                   # 턱 아래를 층 높이(0.2)마다 0.2씩 줄여 45도로 (받침 없이 출력)
        step = (extrude_z(pillar_sec(-WALL + 0.2), *zr)
                - extrude_z(pillar_sec(-WALL - 1.4 + 0.2 * (k + 1)), zr[0] - 1, zr[1] + 1))
        add.append(step ^ box(-200, 200, Y_PART + 1.6 + 0.2 * k - 0.01, Y_PART + 1.6 + 0.2 * (k + 1), -300, 300))
    add.append(lip_b ^ box(-200, 200, Y_PART - 2.5, Y_PART + 0.1, -300, 300))
    for x, z in SNAP_PTS:                                # 딸깍 돌기
        sgn = 1 if x > 0 else -1
        add.append(M.sphere(1.0, 16).translate([x - sgn * 0.55, Y_PART - 1.4, z]))
    m = union([body] + add) ^ extrude_z(pillar_sec(), PZ0, PZ1)
    # 앞쪽 받침턱: 모터 자리를 아래에서 나사 하나로 당겨 잡는다 (머리가 앞으로 들리지 않게)
    fx, fy = FRONT_SCREW
    brk = prism_xz(poly([(12.0, PZ1), (BRK_X1, PZ1), (BRK_X1, PZ1 - 3.0), (BRK_X1 - 1.5, PZ1 - 3.0),
                         (PX1 - 0.5, PZ1 - 3.0 - (BRK_X1 - 1.5 - PX1 + 0.5)), (12.0, PZ1 - 3.0 - (BRK_X1 - 1.5 - PX1 + 0.5))]),
                   -8.0, PY1) - extrude_z(pillar_sec(-0.6), -300, 300)
    m = union([m, _ears(Y_PART, PY1), brk])
    cut = [cz(1.3, 1.4, PZ1 + 0.1, sx, sy)]
    cut.append(extrude_z(teardrop_cs(1.7, -90.0), PZ1 - 4.0, PZ1 + 0.1).translate([fx, fy, 0]))    # 앞쪽 나사 구멍
    cut.append(extrude_z(teardrop_cs(3.2, -90.0), PZ1 - 20.0, PZ1 - 3.0).translate([fx, fy, 0]))   # 나사 머리 자리 (물방울: 받침 없이 출력)
    sdx = BX0 + 28.0                                     # 마이크로SD가 보드 끝에서 2.5mm 튀어나오는 자리
    cut.append(box(sdx - 8.0, sdx + 8.0, Y_PART - 0.1, 3.4, PZ1 - WALL - 0.5, PZ1 + 0.1))
    for x, y in EAR_SCREWS_SHELL:                        # 물방울 구멍 (꼭지가 출력할 때 위쪽 = -Y)
        cut.append(extrude_z(teardrop_cs(1.7, -90.0), PZ0 - 1, PZ0 + 4).translate([x, y, 0]))
    return m - union(cut)


def pillar_lid():
    """기둥 뚜껑: 파이 부품면 쪽 (-Y)"""
    body = _pillar_body() ^ box(-200, 200, -200, Y_PART, -300, 300)
    add = []
    holes = pi_hole_xz()
    for x, z in (holes[2], holes[3]):                    # 파이 누름 기둥
        add.append(cy(3.0, PY0 + 0.1, -PI_T / 2 - 0.3, x, z))
    m = union([body] + add) ^ extrude_z(pillar_sec(), PZ0, PZ1)
    m = m + _ears(PY0, Y_PART - 0.2, 38.5)
    cut = []
    for x, z in (holes[2], holes[3]):
        cut.append(cy(1.6, -4.0, 0, x, z))
    x0, x1, y0, y1 = CABLE_HOLE
    cut.append(box(x0, x1, y0, Y_PART + 0.1, PZ1 - WALL - 0.5, PZ1 + 0.5))      # 전선 구멍
    for x, y in EAR_SCREWS_LID:                          # 물방울 구멍 (꼭지가 출력할 때 위쪽 = +Y)
        cut.append(extrude_z(teardrop_cs(1.7, 90.0), PZ0 - 1, PZ0 + 4).translate([x, y, 0]))
    for i in range(7):                                   # 통풍 틈 (파이 열이 위로 빠지게)
        xv = -34.0 + i * 6.5
        cut.append(box(xv - 1.3, xv + 1.3, PY0 - 1, PY0 + WALL + 1, -44.0, -8.0))
    for x, z in SNAP_PTS:                                # 딸깍 홈
        sx = 1 if x > 0 else -1
        cut.append(M.sphere(1.15, 16).translate([x - sx * 0.55, Y_PART - 1.4, z]))
    return m - union(cut)


def _foot_profile(R, zt, zb, w, c=5.0):
    """윗모서리를 45도로 깎은 속 빈 원반의 (r, z) 단면"""
    return [(0, zt), (R - c, zt), (R, zt - c), (R, zb), (R - w, zb), (R - w, zt - c - 0.4 * w),
            (R - c - 0.4 * w, zt - w), (0, zt - w)]


def _tray_xy(R=FOOT_R):
    r = R - FOOT_WALL - 4.6
    return [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for a in TRAY_SCREW_ANGLES]


def foot_window(R=FOOT_R):
    """배터리 창의 (폭의 절반, 깊이). 작은 받침은 벽이 더 휘어 있어 깊이 판다."""
    return (38.0, 16.0) if R < 80 else (42.0, 15.0)


def foot(R=FOOT_R):
    """받침: 속이 빈 원반. 아래는 바닥판(tray)으로 닫는다. R=70 작은 것(S), R=95 큰 것(L)"""
    zt, zb, w = FOOT_ZT, FOOT_ZT - FOOT_H, FOOT_WALL
    shell = revolve_profile(_foot_profile(R, zt, zb, w, FOOT_CH))
    bosses = [cz(4.5, zt - 12.0, zt - w + 0.1, x, y) for x, y in EAR_SCREWS_SHELL + EAR_SCREWS_LID]
    cols = [cz(4.5, zb + TRAY_T, zt - w + 0.1, x, y) for x, y in _tray_xy(R)]
    m = union([shell] + bosses + cols) ^ cz(R, zb, zt)
    cut = [extrude_z(rrect(-50.0, -16, -22, 0, 4), zt - w - 1, zt + 1)]        # 전선이 내려가는 구멍
    for x, y in EAR_SCREWS_SHELL + EAR_SCREWS_LID:
        cut.append(cz(1.3, zt - 11.0, zt + 1, x, y))
    for x, y in _tray_xy(R):
        cut.append(cz(1.3, zb + TRAY_T - 1, zb + TRAY_T + 12, x, y))
    cut.append(box(-R - 1, -R + 12, -16, 16, zb - 1, zb + 16))                 # 뒤: 랜선·충전선·스위치
    hw, dp = foot_window(R)
    cut.append(box(-hw, hw, -R - 1, -R + dp, zb - 1, zb + 26))                 # 오른쪽: 배터리를 밀어 넣는 창
    return m - union(cut)


def tray(R=FOOT_R):
    """받침 바닥판: 배터리를 올려놓는다."""
    zb = FOOT_ZT - FOOT_H
    m = cz(R - FOOT_WALL - 0.4, zb, zb + TRAY_T)
    cut = []
    for x, y in _tray_xy(R):
        cut.append(cz(1.7, zb - 1, zb + TRAY_T + 1, x, y))
        cut.append(M.cylinder(1.6, 3.3, 1.7).translate([x, y, zb - 0.1]))
    return m - union(cut)


def motor_seat():
    """모터1 자리: 기둥 위에 얹는 원형 받침 + 둘레 벽. 출력: 그대로."""
    base = cz(SEAT_R, SEAT_Z0, M1_Z0)
    tongue = box(-54, -30, -9, 3, SEAT_Z0, M1_Z0) ^ cz(R_KEEP - 1, SEAT_Z0, M1_Z0)
    skirt = cz(SEAT_R, M1_Z0, M1_ZT) - cz(SEAT_R - 1.4, M1_Z0 - 1, M1_ZT + 1)
    posts = [cz(3.0, M1_Z0, M1_ZT - 0.8, -M_OFFSET, s * M_EAR) for s in (1, -1)]
    posts += [box(-M_OFFSET - 1, -M_OFFSET + 1, 17.5 if s > 0 else -34.0, 34.0 if s > 0 else -17.5,
                  M1_Z0, M1_ZT - 4) for s in (1, -1)]
    fx, fy = FRONT_SCREW
    posts.append(cz(3.6, M1_Z0, M1_Z0 + 6.0, fx, fy))                  # 앞쪽 나사 받이
    m = union([base, tongue, skirt] + posts)
    cut = [cz(1.3, M1_Z0 + 5, M1_ZT, -M_OFFSET, s * M_EAR) for s in (1, -1)]
    cut.append(cz(1.3, SEAT_Z0 - 0.1, M1_Z0 + 5.2, fx, fy))
    x0, x1, y0, y1 = CABLE_HOLE
    cut.append(box(x0, x1, y0, y1, SEAT_Z0 - 1, M1_Z0 + 1))            # 전선 구멍
    cut.append(cz(1.7, SEAT_Z0 - 1, M1_Z0 + 1, *SEAT_SCREW))           # 고정 나사
    z0, z1 = M1_Z0 + 5, M1_ZT - 3
    slot = prism_xy_to_yz(poly([(-2.2, z0), (2.2, z0), (2.2, z1 - 2.2), (0, z1), (-2.2, z1 - 2.2)]), SEAT_R - 3, SEAT_R + 1)
    for k in range(8):                                                  # 통풍 틈 (윗변이 뾰족해서 받침 없이 출력)
        cut.append(rot_z(slot, 22.5 + 45 * k))
    return m - union(cut)


def spring_cup():
    """전선통: 모터2 전선(리본)이 감겼다 풀렸다 하는 통. 출력: 그대로."""
    floor = cz(SEAT_R, CUP_Z0, CUP_Z0 + 2)
    wall = cz(SEAT_R, CUP_Z0 + 2, CUP_Z1) - cz(SEAT_R - 2.6, CUP_Z0 + 1, CUP_Z1 + 1)
    m = floor + wall
    cut = [cz(4.9, CUP_Z0 - 1, CUP_Z0 + 3)]
    cut.append(cz(SEAT_R - 1.4, CUP_LEDGE_Z, CUP_Z1 + 1))                           # 뚜껑이 얹히는 턱
    cut.append(box(SEAT_R - 3, SEAT_R + 1, -2.6, 2.6, CUP_LEDGE_Z, CUP_Z1 + 1))     # 뚜껑 돌기 자리
    for s in (1, -1):
        cut.append(cz(1.7, CUP_Z0 - 1, CUP_Z0 + 3, -M_OFFSET, s * M_EAR))
        cut.append(M.cylinder(1.8, 1.7, 3.3).translate([-M_OFFSET, s * M_EAR, CUP_Z0 + 0.3]))
    cut.append(extrude_z(sector(30.2, 33.2, 218, 238), CUP_Z0 - 1, CUP_Z0 + 3))     # 리본이 내려가는 틈 (3 x 11)
    return m - union(cut)


def cup_lid():
    """전선통 뚜껑 + 좌우 회전 멈춤턱 + 멈춤 고리가 도는 가운데 테. 출력: 그대로(턱이 위)."""
    disc = cz(SEAT_R - 1.6, CUP_LEDGE_Z, LID_ZT)
    tab = box(SEAT_R - 3, SEAT_R - 0.1, -2.4, 2.4, CUP_LEDGE_Z, LID_ZT)
    block = extrude_z(sector(17.0, 21.0, 75, 105), LID_ZT - 0.1, LID_ZT + 2.6)
    collar = cz(9.7, CUP_LEDGE_Z, LID_ZT + 2.4)
    return union([disc, tab, block, collar]) - cz(8.6, CUP_LEDGE_Z - 1, LID_ZT + 3)


def stop_ring(angle=270.0):
    """좌우 ±270도에서 멈추게 하는 고리 (뚜껑 가운데 테를 축으로 자유롭게 돈다)"""
    ring = cz(12.0, LID_ZT + 0.2, LID_ZT + 2.6) - cz(10.0, LID_ZT, LID_ZT + 3)
    arm = extrude_z(sector(11.5, 21.0, angle - 30, angle + 30), LID_ZT + 0.2, LID_ZT + 2.6)
    return ring + arm


# ----------------------------------------------------------------- 좌우 회전부 (P)
def _arm_cs(left):
    top = poly([(-23, 14), (23, 14), (23, 40), (12, BAR_Z1), (-12, BAR_Z1), (-23, 40)])
    return CS.batch_hull([CS.circle(22.0 if left else 9.5), top])


def bail(d=D_BORE_D, flat=D_BORE_FLAT):
    """회전 틀: 모터1 축에 얹혀 좌우로 도는 틀. 모터2와 드라이버 2번을 싣는다. 출력: 뒤집어서."""
    bar = box(-BAR_HW, BAR_HW, -BAIL_OUT, BAIL_OUT, BAR_Z0, BAR_Z1)
    rt = CAP_R - (BAR_Z1 - BAR_Z0)                       # 45도 원뿔 (뒤집어서 받침 없이 출력)
    cap = revolve_profile([(0, BAR_Z1), (rt, BAR_Z1), (CAP_R, BAR_Z0), (CAP_R - 2.1, BAR_Z0),
                           (rt - 1.3, BAR_Z1 - 1.6), (0, BAR_Z1 - 1.6)])
    thrust = cz(33.0, LID_ZT + THRUST_GAP, BAR_Z0 + 0.6) - cz(30.5, LID_ZT, BAR_Z0 + 1)             # 처짐 받침 테
    right = prism_xz(_arm_cs(False), BAIL_IN, BAIL_OUT)
    left = prism_xz(_arm_cs(True), -BAIL_OUT, -BAIL_IN)
    hub = cz(HUB_R, HUB_Z0, BAR_Z0 + 0.1)
    peg = extrude_z(sector(12.5, 15.5, 75, 105), LID_ZT + 0.8, BAR_Z0 + 0.1)
    # 위아래 회전 멈춤 못 2개 (왼쪽 팔 바깥면)
    ang = np.degrees(np.arcsin(3.0 / 18.0))
    stops = []
    for a in (TILT_STOP_HI - 90 + 10 + ang, TILT_STOP_LO - 90 - 10 - ang):
        ar = np.radians(a)
        stops.append(prism_xz(teardrop_cs(3.0, 90.0), -BAIL_OUT - 4.0, -BAIL_OUT + 0.1)
                     .translate([18.0 * np.cos(ar), 0, 18.0 * np.sin(ar)]))          # 뒤집어 출력할 때 아래가 뾰족하게
    keeper = cz(16.9, LID_ZT + 3.1, BAR_Z1 - 1.5) - cz(15.6, LID_ZT, BAR_Z1)                          # 멈춤 고리가 들뜨지 않게 누르는 테
    m = union([bar, cap, thrust, keeper, right, left, hub, peg] + stops)
    cut = []
    rw = RIBBON_HW
    cut.append(cz(d / 2 + 0.1, HUB_Z0 - 0.1, HUB_Z0 + 2.6))                                          # 축의 둥근 부분
    cut.append(extrude_z(d_bore_cs(d, flat), HUB_Z0 + 2.5, HUB_Z0 + 9.0))                            # 축의 납작한 부분
    cut.append(box(-rw, rw, -HUB_R - 0.1, -HUB_R + 1.7, HUB_Z0 - 0.1, BAR_Z0 + 1.7))                 # 리본 홈
    cut.append(box(-rw, rw, -BAIL_IN, -HUB_R + 1.7, BAR_Z0 - 0.1, BAR_Z0 + 1.7))                    # 리본 길
    cut.append(box(-rw, rw, -34.5, -29.5, LID_ZT, BAR_Z0 + 1.7))                                     # 받침 테를 지나는 틈
    cut.append(box(-rw, rw, -17.5, -15.0, LID_ZT, BAR_Z0 + 1.7))                                     # 누름 테를 지나는 틈
    # 모터2 전선 5가닥이 가로대 윗면 홈으로 건너간다 (모터2 쪽 -> 드라이버 2 쪽)
    gz, gt = BAR_Z1 - 3.4, BAR_Z1 + 0.1                # 홈 바닥은 V자 (뒤집어 출력하면 지붕이 뾰족해서 받침이 필요 없다)
    vee = lambda hw: poly([(-hw, gt), (hw, gt), (hw, gz + hw), (0, gz), (-hw, gz + hw)])
    cut.append(prism_xy_to_yz(vee(1.3), -1.0, 1.0).rotate([0, 0, 90]).scale([1, 55.4, 1]))           # 길이 방향 홈 (y -55.4..55.4)
    for y0 in (51.0, -54.4):                                                                         # 긴 홈의 끝(±55.4)과 면이 겹치지 않게 1mm 안쪽에 둔다
        cut.append(prism_xy_to_yz(vee(1.7), 0.0, BAR_HW + 0.1).translate([0, y0 + 1.7, 0]))          # 옆으로 빠지는 길
    hole = teardrop_cs(PIN_R + 0.15, -90.0)                                                          # 물방울 구멍: 뒤집어 출력할 때 지붕이 처지지 않는다
    cut.append(prism_xz(hole, BAIL_IN - 0.5, BAIL_OUT + 0.5))                                        # 회전축 구멍
    cut.append(prism_xz(hole, -BAIL_OUT - 0.5, -BAIL_IN + 0.5))
    cut.append(prism_xz(teardrop_cs(4.9, -90.0), BAIL_IN - 0.5, BAIL_OUT + 0.5).translate([0, 0, CDIST]))   # 모터2 축 보스
    for s in (1, -1):
        cut.append(cy(1.3, BAIL_IN - 0.5, BAIL_OUT + 0.5, s * M_EAR, M2_BODY_Z))                    # 모터2 귀 나사
    return m - union(cut)


def pinion(phase=0.0, d=D_BORE_D, flat=D_BORE_FLAT):
    cs, _ = gear_cs(MOD, ZP, shift=P_SHIFT)
    cs = cs.rotate(-90 + phase) - d_bore_cs(d, flat).rotate(phase)
    return prism_xz(cs, PINION_Y0, PINION_Y0 + PINION_T).translate([0, 0, CDIST])


# ----------------------------------------------------------------- 위아래 회전부 (T)
def fork():
    """포크: 판을 잡고 위아래로 도는 ㄷ자 틀"""
    outer = rrect(-40, BAR_R1, -ARM_OUT, ARM_OUT, 12.0)
    inner = rrect(-60, BAR_R0, -ARM_IN, ARM_IN, 6.0)
    cs = (outer - inner) ^ CS.square([BAR_R1 + 9.0, 2 * ARM_OUT + 2]).translate([-9.0, -ARM_OUT - 1])
    m = extrude_z(cs, -FORK_HZ, FORK_HZ)
    keys = [box(-KEY_HX, KEY_HX, KEY_Y0, ARM_IN + 0.1, -KEY_HZ, KEY_HZ),
            box(-KEY_HX, KEY_HX, -ARM_IN - 0.1, -KEY_Y0, -KEY_HZ, KEY_HZ)]
    tenon = box(BAR_R1 - 0.1, BAR_R1 + TENON_L, -TENON / 2, TENON / 2, -FORK_HZ, FORK_HZ)
    # 간격 원판 쪽 돌기 위의 작은 턱: 원판을 거꾸로 끼우거나 큰 기어와 자리를 바꿔 끼우지 못하게 한다
    ridge = box(-2.0, 2.0, -ARM_IN - 0.1, -KEY_Y0, KEY_HZ - 0.1, KEY_HZ + 1.2)
    m = union([m, tenon, ridge] + keys)
    hole = teardrop_cs(PIN_R, 90.0)
    cut = [prism_xz(hole, KEY_Y0 - 1, ARM_OUT + 1), prism_xz(hole, -ARM_OUT - 1, -KEY_Y0 + 1)]
    cut.append(cz(1.1, -10, 10, BAR_R1 + 2.2, 0))            # 판 고정 핀 구멍
    return m - union(cut)


def wheel():
    """기어 휠 48T: 포크 오른쪽 팔에 끼워 같이 돈다"""
    cs, _ = gear_cs(MOD, ZW)
    cs = cs.rotate(90 + 360.0 / ZW / 2)
    holes = [CS.circle(5.6).translate([19.0 * np.cos(a), 19.0 * np.sin(a)])
             for a in np.radians(np.arange(30, 360, 60))]
    cs = cs - CS.batch_boolean(holes, m3.OpType.Add) - CS.circle(PIN_R + 0.1)
    m = prism_xz(cs, WHEEL_Y0, WHEEL_Y0 + WHEEL_T)
    pocket = box(-KEY_HX - 0.15, KEY_HX + 0.15, KEY_Y0, WHEEL_Y0 + WHEEL_T + 1, -KEY_HZ - 0.15, KEY_HZ + 0.15)
    return m - pocket


def spacer():
    """왼쪽 간격 원판 + 위아래 회전 멈춤 돌기"""
    disc = cy(13.0, -ARM_IN, -WHEEL_Y0)
    tab = prism_xz(sector(12.5, 21.0, -100, -80), -WHEEL_Y0 - 4.0, -WHEEL_Y0)
    pocket = box(-KEY_HX - 0.15, KEY_HX + 0.15, -ARM_IN - 1, -KEY_Y0, -KEY_HZ - 0.15, KEY_HZ + 0.15)
    notch = box(-2.2, 2.2, -ARM_IN - 1, -KEY_Y0, KEY_HZ, KEY_HZ + 1.45)            # 판 틀의 작은 턱이 들어가는 홈 (위쪽에만)
    return (disc + tab) - pocket - notch - cy(PIN_R + 0.1, -ARM_IN - 1, -WHEEL_Y0 + 1)


def pin(side=1, r=PIN_R - 0.05):
    """회전축 핀 (좌우 1개씩). 출력: 머리를 바닥에 세워서."""
    if side > 0:
        return cy(r, BAIL_IN - 0.5, ARM_OUT) + cy(5.0, ARM_OUT, ARM_OUT + 1.6)
    return cy(r, -ARM_OUT, -BAIL_IN + 0.5) + cy(5.0, -ARM_OUT - 1.6, -ARM_OUT)


def plate_panel(length=PLATE_L, landscape=False):
    """판 받침: 바깥면에 알루미늄 테이프를 붙인다. 가운데 사각 구멍으로 포크에 꽂는다."""
    hw, hl = PLATE_W / 2, length / 2
    x1 = R_FOIL
    x0 = x1 - PANEL_T
    rib = 3.0
    panel = box(x0, x1, -hw, hw, -hl, hl)
    frame = (box(x0 - rib, x0 + 0.1, -hw, hw, -hl, hl)
             - box(x0 - rib - 1, x0 + 1, -hw + 2.4, hw - 2.4, -hl + 2.4, hl - 2.4))
    col_x0 = x0 - 4.0
    collar = box(col_x0, x0 + 0.1, -9, 9, -9, 9)
    ribs = []
    for sy in (1, -1):
        for sz in (1, -1):
            p0 = np.array([sy * 8.0, sz * 8.0]); p1 = np.array([sy * (hw - 1.5), sz * (hl - 1.5)])
            d = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-d[1], d[0]]) * 1.0
            ribs.append(prism_xy_to_yz(poly([p0 + n, p1 + n, p1 - n, p0 - n]), x0 - rib, x0 + 0.1))
    m = union([panel, frame, collar] + ribs)
    sa, sb = TENON / 2 + 0.15, FORK_HZ + 0.15            # 십자 구멍: 세로·가로 두 방향으로 꽂을 수 있다
    cut = [box(col_x0 - 1, x0, -sa, sa, -sb, sb), box(col_x0 - 1, x0, -sb, sb, -sa, sa)]
    cut.append(cy(1.1, -10, 10, x0 - 2.2, 0))
    cut.append(cz(1.1, -10, 10, x0 - 2.2, 0))
    m = m - union(cut)
    return m.rotate([90, 0, 0]) if landscape else m


def foil(length=PLATE_L, landscape=False):
    hw, hl = PLATE_W / 2, length / 2
    m = box(R_FOIL, R_FOIL + 0.15, -hw, hw, -hl, hl)
    return m.rotate([90, 0, 0]) if landscape else m


def test_coupon():
    """맞춤 시험 조각. 앞줄: 모터 축 구멍 3종(옆의 점 개수 = 맞춤 번호)과 세운 벽의 핀 구멍 2종(끼움 3.0 / 회전 3.15).
    뒷줄: 판 꽂는 구멍, M3 나사 구멍 2종, 판 고정 핀 구멍."""
    m = box(0, 60, 0, 34, 0, 6)
    cut = []
    for i in (1, 2, 3):                                        # 모터 축
        d, f, _ = FITS[i]
        xc = 9.0 + (i - 1) * 12.0
        cut.append(extrude_z(d_bore_cs(d, f), -1, 7).translate([xc, 10.5, 0]))
        for k in range(i):                                     # 점 i개
            cut.append(cz(0.9, 5.2, 7, xc + (k - (i - 1) / 2) * 2.6, 3.4))
    # 핀 구멍은 실제 부품과 같이 옆으로 누운 물방울 구멍으로 시험한다 (세운 벽에 2개)
    m = m + box(38.0, 60.0, 0.0, 6.0, 5.9, 20.0)
    cut.append(prism_xz(teardrop_cs(PIN_R, 90.0), -1, 7).translate([44.0, 0, 12.5]))             # 판 틀과 같은 구멍 (핀이 빡빡하게)
    cut.append(prism_xz(teardrop_cs(PIN_R + 0.15, 90.0), -1, 7).translate([54.0, 0, 12.5]))      # 회전 틀과 같은 구멍 (핀이 헐겁게 돈다)
    cut.append(box(52.0, 56.0, 2.6, 3.4, 19.2, 21))            # 헐거운 구멍 표시 (벽 윗면의 짧은 줄)
    c = 0.15
    cut.append(box(-TENON / 2 - c, TENON / 2 + c, -FORK_HZ - c, FORK_HZ + c, -1, 7).translate([11, 24.5, 0]))   # 판 꽂는 구멍
    cut.append(cz(1.3, -1, 7, 27.0, 24.5))                     # M3 나사가 파고드는 구멍
    cut.append(cz(1.7, -1, 7, 36.0, 24.5))                     # M3 나사가 그냥 지나가는 구멍
    cut.append(cz(1.1, -1, 7, 44.0, 24.5))                     # 판 고정 핀(지름 2) 구멍
    return m - union(cut)


def tenon_gauge():
    """시험용 돌기 (포크의 판 꽂는 돌기와 같은 치수 10 x 12)"""
    return box(-9, 9, -FORK_HZ, FORK_HZ, 0, 3) + box(-TENON / 2, TENON / 2, -FORK_HZ, FORK_HZ, 0, 12)


# ----------------------------------------------------------------- 구매 부품 (모양만)
def motor_body_dummy():
    """28BYJ-48: 축이 +Z, 축 위치가 원점, 앞면(귀 높이)이 z=0, 몸통 중심은 -X 쪽"""
    body = cz(M_BODY_R, -M_BODY_H, 0, -M_OFFSET, 0)
    ears = CS.batch_hull([CS.circle(3.5).translate([-M_OFFSET, M_EAR]),
                          CS.circle(3.5).translate([-M_OFFSET, -M_EAR])])
    ears = extrude_z(ears - CS.circle(2.1).translate([-M_OFFSET, M_EAR])
                     - CS.circle(2.1).translate([-M_OFFSET, -M_EAR]), -0.8, 0)
    boss = cz(4.5, 0, 1.5)
    cap = box(-M_OFFSET - 17.0, -M_OFFSET - 12.0, -7.3, 7.3, -M_BODY_H + 1, -2) - cz(M_BODY_R + 0.05, -M_BODY_H - 1, 1, -M_OFFSET, 0)
    return union([body, ears, boss]), cap


def motor_shaft_dummy(flat_across="y"):
    sq = [6, 3.0] if flat_across == "y" else [3.0, 6]
    return extrude_z(CS.circle(2.5) ^ CS.square(sq, True), 4.0, M_SHAFT_L) + cz(2.5, 1.55, 4.1)


def _m1(m):
    return m.translate([0, 0, M1_ZT])


def _m2(m):
    # 축을 +Y로, 몸통 중심을 -Z 쪽으로
    T = np.array([[0, 1, 0, 0], [0, 0, 1, BAIL_IN], [1, 0, 0, CDIST]], dtype=np.float64)
    return m.transform(T)


def pi_dummy():
    t = PI_T / 2
    board = box(BX0, BX1, -t, t, BZ0, BZ1)
    board = board - union([cy(1.375, -2, 2, x, z) for x, z in pi_hole_xz()])
    parts = []
    for xc in (BX0 + 29.0, BX0 + 47.0):                                   # USB 2단
        parts.append(box(xc - 6.6, xc + 6.6, -t - 15.6, -t, BZ0 - 2, BZ0 + 15.4))
    xc = BX0 + 10.25
    parts.append(box(xc - 8, xc + 8, -t - 13.5, -t, BZ0 - 2, BZ0 + 19))    # 랜
    parts.append(box(BX0 + 50, BX0 + 55.1, -t - 8.5, -t, BZ1 - 57.9, BZ1 - 7.1))   # GPIO
    parts.append(box(BX0 - 1, BX0 + 10.5, -t - 6.5, -t, BZ1 - 39.5, BZ1 - 24.5))   # HDMI
    parts.append(box(BX0 - 1, BX0 + 5, -t - 2.9, -t, BZ1 - 14.6, BZ1 - 6.6))       # 전원 단자
    parts.append(box(BX0 + 18, BX0 + 32, -t - 2.4, -t, BZ1 - 38, BZ1 - 24))        # SoC
    parts.append(box(BX0 + 33.5, BX0 + 46, -t - 2.0, -t, BZ1 - 20, BZ1 - 8))       # 무선 모듈 덮개
    return board, union(parts)


def antenna_marker():
    return M.sphere(2.2, 24)


def usb_plug_dummy():
    """ㄱ자 마이크로 5핀 플러그와 케이블"""
    t = PI_T / 2
    plug = box(BX0 - 10.3, BX0 - 1.2, -t - 6.0, -t - 0.2, BZ1 - 15, BZ1 - 6.5)
    cable = cz(1.9, PZ0 - 6, BZ1 - 14, BX0 - 5.6, -t - 3.4)
    return plug + cable


def driver_dummy():
    """ULN2003 보드: 35 x 32, 부품은 +Z 쪽. 원점 = 보드 중심"""
    board = box(-17.5, 17.5, -16, 16, 0, 1.6)
    board = board - union([cz(1.5, -1, 3, sx * 15.0, sy * 13.25) for sx in (-1, 1) for sy in (-1, 1)])     # 구멍 4개 (흔한 보드: 간격 30 x 26.5, 지름 3)
    comps = union([box(-9, 1, -5, 5, 1.6, 9.6), box(4, 16.5, -6.5, 6.5, 1.6, 8.6),
                   box(-16.5, -11, -5, 5, 1.6, 10)])
    return board, comps


def _drv2(m):
    # 왼쪽 팔 안쪽 면에 붙임: 보드 법선이 +Y(안쪽)
    T = np.array([[1, 0, 0, 0], [0, 0, 1, -BAIL_IN + 1.2], [0, -1, 0, 29.0]], dtype=np.float64)
    return m.transform(T)


def _drv1(m):
    # 받침 천장 아래, 전선 구멍 바로 옆(모터1 전선 24cm가 닿는 곳): 부품이 아래를 향함
    T = np.array([[0, 1, 0, -34.0], [1, 0, 0, 22.0], [0, 0, -1, FOOT_ZT - FOOT_WALL - 1.2]], dtype=np.float64)
    return m.transform(T)


BATTERY = {"S": (65.0, 100.0, 12.0), "L": (70.0, 144.0, 16.0)}     # 그림용 기본 크기 (가로, 세로, 두께)


def battery_dummy(R=FOOT_R):
    lx, ly, t = BATTERY["S" if R < 80 else "L"]
    zb = FOOT_ZT - FOOT_H + TRAY_T + 0.5
    y0 = -(R - 13.0)                                           # 단자 쪽 끝을 창 안쪽 13mm에 둔다
    return box(-lx / 2, lx / 2, y0, y0 + ly, zb, zb + t)


# ----------------------------------------------------------------- 조립
COL = dict(body="#ECE8E1", moving="#30363D", gear="#D4622B", plate="#C2C8CE",
           panel="#4A525B", pcb="#2F7D4F", metal="#A9AEB3", cap="#3F78C9",
           dark="#22262B", ant="#E0533A", batt="#3B4046", ring="#D4622B")


def pose_P(m, pan):
    return m.rotate([0, 0, pan])


def pose_T(m, pan, tilt):
    return m.rotate([0, -tilt, pan])


def assembly(pan=0.0, tilt=0.0, landscape=False, length=PLATE_L, dummies=True, R=FOOT_R):
    """이름 -> (형상, 그룹, 색, 출력부품 여부). R = 받침 반지름(70 또는 95)"""
    A = {}
    A["foot"] = (foot(R), "F", COL["body"], True)
    A["tray"] = (tray(R), "F", COL["body"], True)
    A["pillar_shell"] = (pillar_shell(), "F", COL["body"], True)
    A["pillar_lid"] = (pillar_lid(), "F", COL["body"], True)
    A["motor_seat"] = (motor_seat(), "F", COL["body"], True)
    A["spring_cup"] = (spring_cup(), "F", COL["body"], True)
    A["cup_lid"] = (cup_lid(), "F", COL["body"], True)
    A["stop_ring"] = (stop_ring(270.0 + pan / 2.0), "S", COL["ring"], True)
    A["bail"] = (pose_P(bail(), pan), "P", COL["moving"], True)
    A["pinion"] = (pose_P(pinion(-4.0 * tilt), pan), "P", COL["gear"], True)
    A["fork"] = (pose_T(fork(), pan, tilt), "T", COL["moving"], True)
    A["wheel"] = (pose_T(wheel(), pan, tilt), "T", COL["gear"], True)
    A["spacer"] = (pose_T(spacer(), pan, tilt), "T", COL["gear"], True)
    A["pin_r"] = (pose_T(pin(1), pan, tilt), "T", COL["body"], True)
    A["pin_l"] = (pose_T(pin(-1), pan, tilt), "T", COL["body"], True)
    A["plate_panel"] = (pose_T(plate_panel(length, landscape), pan, tilt), "T", COL["panel"], True)
    if dummies:
        A["foil"] = (pose_T(foil(length, landscape), pan, tilt), "T", COL["plate"], False)
        mb, mc = motor_body_dummy()
        A["motor1"] = (_m1(mb), "F", COL["metal"], False)
        A["motor1_cap"] = (_m1(mc), "F", COL["cap"], False)
        A["motor1_shaft"] = (pose_P(_m1(motor_shaft_dummy("y")), pan), "P", COL["metal"], False)
        A["motor2"] = (pose_P(_m2(mb), pan), "P", COL["metal"], False)
        A["motor2_cap"] = (pose_P(_m2(mc), pan), "P", COL["cap"], False)
        A["motor2_shaft"] = (pose_P(_m2(motor_shaft_dummy("x").rotate([0, 0, -4.0 * tilt])), pan), "P", COL["metal"], False)
        b, c = pi_dummy()
        A["pi_board"] = (b, "F", COL["pcb"], False)
        A["pi_parts"] = (c, "F", COL["metal"], False)
        A["usb_plug"] = (usb_plug_dummy(), "F", COL["dark"], False)
        A["antenna"] = (antenna_marker(), "F", COL["ant"], False)
        db, dc = driver_dummy()
        A["driver2_board"] = (pose_P(_drv2(db), pan), "P", COL["pcb"], False)
        A["driver2_parts"] = (pose_P(_drv2(dc), pan), "P", COL["dark"], False)
        A["driver1_board"] = (_drv1(db), "F", COL["pcb"], False)
        A["driver1_parts"] = (_drv1(dc), "F", COL["dark"], False)
        A["battery"] = (battery_dummy(R), "F", COL["batt"], False)
    return A
