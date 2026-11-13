"""DeteCam 수신기 v2 (관절 3개) 부품 모델. 좌표와 치수는 p2.py 참고.

무리:  B 받침(고정)  ->  Y 요크(J1 받침 회전)  ->  H 머리(J2 끄덕임)  ->  R 회전부(J3 판 회전)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from cadlib import *
import parts as v1
from p2 import *


# ----------------------------------------------------------------- 도우미
def chbox(x0, x1, y0, y1, z0, z1, c):
    """모서리를 모두 45도로 깎은 상자"""
    return M.batch_hull([box(x0, x1, y0 + c, y1 - c, z0 + c, z1 - c), box(x0 + c, x1 - c, y0, y1, z0 + c, z1 - c),
                         box(x0 + c, x1 - c, y0 + c, y1 - c, z0, z1)])


def gear_big(mod, z, alpha_deg=20.0, backlash=0.2, n=8):
    """잇수가 많은 기어의 인벌류트 단면"""
    al = np.radians(alpha_deg); r = mod * z / 2.0; rb = r * np.cos(al); ra = r + mod; rf = r - 1.25 * mod
    s = mod * np.pi / 2 - backlash
    inv = lambda a: np.tan(a) - a
    def half(rho):
        return s / (2 * r) + inv(al) - inv(np.arccos(min(1.0, rb / rho)))
    rs = np.linspace(max(rb, rf), ra, n)
    flank = [(rho, half(rho)) for rho in rs]
    pts = []; pitch = 2 * np.pi / z
    for i in range(z):
        c = i * pitch; h0 = flank[0][1]
        if rf < rb:
            pts.append((rf * np.cos(c - h0), rf * np.sin(c - h0)))
        for rho, h in flank:
            pts.append((rho * np.cos(c - h), rho * np.sin(c - h)))
        for rho, h in flank[::-1]:
            pts.append((rho * np.cos(c + h), rho * np.sin(c + h)))
        if rf < rb:
            pts.append((rf * np.cos(c + h0), rf * np.sin(c + h0)))
        a0, a1 = c + h0, c + pitch - h0
        for a in np.linspace(a0, a1, 3)[1:-1]:
            pts.append((rf * np.cos(a), rf * np.sin(a)))
    return poly(pts)


def pinion_cs(mod, z, shift, ra, alpha_deg=20.0, backlash=0.15, n=10):
    """잇수가 적은 작은 기어의 단면. 이끝이 뾰족해지지 않게 바깥 반지름 ra 에서 자른다."""
    al = np.radians(alpha_deg); r = mod * z / 2.0; rb = r * np.cos(al); rf = r - mod * (1.25 - shift)
    s = mod * (np.pi / 2 + 2 * shift * np.tan(al)) - backlash
    inv = lambda a: np.tan(a) - a
    def half(rho):
        return s / (2 * r) + inv(al) - inv(np.arccos(min(1.0, rb / rho)))
    flank = [(rho, half(rho)) for rho in np.linspace(max(rb, rf), ra, n)]
    assert flank[-1][1] * ra * 2 > 0.4, "이끝이 너무 얇음"
    pts = []; pitch = 2 * np.pi / z
    for i in range(z):
        c = i * pitch; h0 = flank[0][1]
        if rf < rb:
            pts.append((rf * np.cos(c - h0), rf * np.sin(c - h0)))
        for rho, h in flank:
            pts.append((rho * np.cos(c - h), rho * np.sin(c - h)))
        for rho, h in flank[::-1]:
            pts.append((rho * np.cos(c + h), rho * np.sin(c + h)))
        if rf < rb:
            pts.append((rf * np.cos(c + h0), rf * np.sin(c + h0)))
        a0, a1 = c + h0, c + pitch - h0
        for a in np.linspace(a0, a1, 4)[1:-1]:
            pts.append((rf * np.cos(a), rf * np.sin(a)))
    return poly(pts)


def pol(r, deg):
    a = np.radians(deg)
    return r * np.cos(a), r * np.sin(a)


def pose_head(m, yaw=0.0, pitch=0.0, spin=None):
    """머리 좌표의 형상을 세계 좌표로. spin 을 주면 먼저 판 회전축 둘레로 돌린다."""
    if spin is not None:
        m = m.rotate([0, 0, spin])
    m = m.translate([0, -Y_T, -S_T]).rotate([0, 0, 90])
    return m.rotate([0, pitch, 0]).translate([0, 0, H_T]).rotate([0, 0, yaw])


def pose_yoke(m, yaw=0.0):
    return m.rotate([0, 0, yaw])


def yoke_to_head(m):
    """끄덕임 0도일 때의 요크 좌표로 만든 형상을 머리 좌표로 옮긴다 (pose_head 의 거꾸로)"""
    return m.translate([0, 0, -H_T]).rotate([0, 0, -90]).translate([0, Y_T, S_T])


def pin2_xz():
    x, z = pol(CD2, PIN_DIR)
    return x, H_T + z


def m1_shaft():
    return pol(CD1, M1_DIR)


def lock_xz():
    x, z = pol(LOCK_R, LOCK_DIR)
    return x, H_T + z


def stop_screws_xz():
    """끄덕임 멈춤 나사 2개의 (x, z). 머리 옆의 돌기가 -5도(원점 잡는 쪽, 평평한 면)와 90도(비스듬한 면)에서 여기에 닿는다."""
    scr = np.degrees(1.5 / LUG_R) + 0.3
    lo = pol(LUG_R, 270.0 - PITCH_MIN + np.degrees(np.arctan(3.0 / LUG_R)) + scr)      # 돌기의 평평한 면 (머리 Y = Y_T - 3)
    hi = pol(LUG_R, 270.0 - PITCH_MAX - np.degrees(np.arctan(2.0 / LUG_R)) - scr)      # 돌기의 비스듬한 면 (나사 끝이 닿는 곳은 Y_T + 2)
    return [(lo[0], H_T + lo[1]), (hi[0], H_T + hi[1])]


# ----------------------------------------------------------------- 받침 (세계 좌표)
def tub(size="S"):
    """받침 통: 배터리가 들어간다. 출력: 그대로."""
    b = BASE[size]; R, bw, bl = b["R"], b["bw"], b["bl"]
    shell = revolve_profile([(0, 0), (R - 1.5, 0), (R, 1.5), (R, TUB_H - 2), (R - 2, TUB_H), (R - 3, TUB_H), (R - 3, 2), (0, 2)])
    posts = [cz(4.5, 1.9, TUB_H, x, y) for x, y in POSTS]
    inner = cz(R - 3.2, 0, TUB_H)
    y0 = -R + BATT_IN                                                                  # 배터리를 1cm 더 안으로: ㄱ자 USB 플러그가 받침 안에 들어온다
    rails = [box(s * (bw / 2 + 0.5) if s > 0 else -(bw / 2 + 2.5), (bw / 2 + 2.5) if s > 0 else -(bw / 2 + 0.5), y0, y0 + bl, 1.9, 5.0) for s in (1, -1)]
    rails.append(box(-28, 28, y0 + bl, y0 + bl + 2.0, 1.9, 6.0))
    m = union([shell] + posts + [union(rails) ^ inner])
    wall = cz(R + 1, 2.0, TUB_H + 1) - cz(R - 3.1, 1, TUB_H + 2)
    cut = [wall ^ box(-(bw / 2 + 2.0), bw / 2 + 2.0, -R - 2, 0, 2.0, TUB_H + 2)]        # 배터리 창 (선은 배터리 위로 지나 가운데 통으로 간다)
    hw, z0 = LAN_NOTCH
    cut.append(wall ^ box(-R - 2, 0, -hw, hw, z0, TUB_H + 2))                         # 뒤: 랜선이 나가는 홈 (위가 트여 있어 눌러 넣는다)
    cut.append(cy(5.0, 0, R + 2, 0, 2.3 + BATT_T / 2))                                # 왼쪽: 배터리를 밀어 빼는 구멍 (지름 10)
    for x, y in POSTS:
        cut.append(cz(1.3, 14.0, TUB_H + 1, x, y))
    for a in (-20, -10, 0, 10, 20):                                                   # 앞쪽 눈금. 가운데 줄은 바닥까지: 수신기의 앞을 맞출 때 쓴다
        cut.append(box(R - 0.8, R + 1, -0.9, 0.9, -1 if a == 0 else 9, 21).rotate([0, 0, a]))
    return m - union(cut)


def lan_cap():
    """랜선 홈 마개: 랜선을 안 쓸 때(혼자 동작) 받침 뒤의 홈을 막는다. 덮개를 덮기 전에 위에서 벽에 끼워 내린다. 출력: 그대로 (세워서)."""
    R = BASE["S"]["R"]; hw, z0 = LAN_NOTCH; c = LANCAP
    zt = TUB_H - 0.2; yb = hw - c["side"]; yf = hw + c["lap"]; zf = z0 - c["lap"]
    body = revolve_profile([(R - 3.3, z0 + 0.1), (R + 0.3, z0 + 0.1), (R + 0.3, TUB_H - 2.2), (R - 1.8, zt), (R - 3.3, zt)]) ^ box(-R - 3, 0, -yb, yb, z0, TUB_H)
    inner = (cz(R - 3 - c["slot"], zf, zt) - cz(R - 3 - c["slot"] - c["t_in"], zf - 1, zt + 1)) ^ box(-R - 3, 0, -yf, yf, zf - 1, zt + 1)
    outer = (cz(R + c["slot"] + c["t_out"], zf, TUB_H - 2.2) - cz(R + c["slot"], zf - 1, TUB_H)) ^ box(-R - 3, 0, -yf, yf, zf - 1, TUB_H)
    return union([body, inner, outer])


def yaw_block_angle():
    return np.degrees(np.arctan2(YAW_STOP_XY[1], YAW_STOP_XY[0])) - 180.0


def cover(size="S"):
    """받침 덮개: 고정 기어(40T), 회전판이 얹히는 테, 가운데 전선 통. 출력: 그대로."""
    R = BASE[size]["R"]; Rc = R - 2.0
    flange = revolve_profile([(0, TUB_H), (Rc, TUB_H), (Rc, COV_Z1 - 1.5), (Rc - 1.5, COV_Z1), (0, COV_Z1)])
    g, _ = gear_cs(MOD, ZW, backlash=0.15)
    gear = extrude_z(g, COV_Z1 - 0.1, GEAR_Z1)
    tube = cz(TUBE_RO, TUB_H, TUBE_Z1)
    ring = cz(RING_R1, COV_Z1 - 0.1, RING_Z1) - cz(RING_R0, COV_Z1 - 1, RING_Z1 + 1)
    rs = np.hypot(*YAW_STOP_XY); a = yaw_block_angle()
    block = extrude_z(sector(rs - 2.0, rs + 2.0, a - 3.5, a + 3.5, 8), TUB_H, 39.0)
    m = union([flange, gear, tube, ring, block])
    cut = [cz(TUBE_RI, TUB_H - 1, TUBE_Z1 + 1)]
    cut.append(revolve_profile([(TUBE_RO - 1.5, CLIP_Z0), (TUBE_RO + 0.6, CLIP_Z0), (TUBE_RO + 0.6, CLIP_Z0 + CLIP_T + 0.2 + 2.1),
                                (TUBE_RO - 1.5, CLIP_Z0 + CLIP_T + 0.2)]))                                   # 고리 홈 (윗벽은 45도)
    cut.append(revolve_profile([(BB_R - BB_W / 2, BB_Z0), (BB_R + BB_W / 2, BB_Z0), (BB_R + BB_W / 2, RING_Z1 + 1), (BB_R - BB_W / 2, RING_Z1 + 1)]))
    for x, y in POSTS:
        cut.append(cz(1.7, TUB_H - 1, COV_Z1 + 1, x, y))
        cut.append(M.cylinder(1.9, 1.7, 3.2).translate([x, y, COV_Z1 - 1.8]))                               # 접시머리 자리
    cut.append(extrude_z(poly([(Rc - 9, 0), (Rc - 3, -3), (Rc - 3, 3)]), COV_Z1 - 0.6, COV_Z1 + 1))          # 앞 표시(받침 회전 0도)
    return m - union(cut)


def clip():
    """회전판이 들리지 않게 가운데 통에 끼우는 고리. 출력: 그대로."""
    ro = TUBE_RO - 1.5 + 0.15
    m = cz(ro + 4.0, CLIP_Z0, CLIP_Z0 + CLIP_T) - cz(ro, CLIP_Z0 - 1, CLIP_Z0 + CLIP_T + 1)        # 바깥 반지름 16.15. 손잡이는 없앴다: 깊이 숙였을 때 판 모서리가 지나는 높이라서
    m = m - box(0, 40, -(ro - 0.9), ro - 0.9, CLIP_Z0 - 1, CLIP_Z0 + CLIP_T + 1)
    return m - cz(1.2, CLIP_Z0 - 1, CLIP_Z0 + CLIP_T + 1, -ro - 2.0, 0)                           # 뺄 때 드라이버 끝을 거는 구멍


# ----------------------------------------------------------------- 요크 (요크 좌표)
def arm_profile():
    body = poly([(WING_X1, DECK_Z0), (WING_X1, H_T), (-10.0, 129.0), (WING_X0, 86.0), (WING_X0, 64.0), (-68.0, DECK_Z0)])
    sx, sz = stop_screws_xz()[1]
    lobe = CS.batch_hull([CS.circle(6.5).translate([sx, sz]), CS.circle(1.0).translate([sx - 20.0, sz - 20.0]), CS.circle(1.0).translate([sx + 12.0, sz - 2.0])])   # 위쪽 멈춤 나사 자리. 뒤로 비스듬히 이어서 받침 없이 출력
    return body + CS.circle(12.0).translate([0, H_T]) + lobe      # 축 둘레 원은 앞 모서리(x = 12)와 딱 맞닿는다


def arm_window():
    w = poly([(-2.0, 52.0), (-2.0, 91.0), (-9.0, 102.0), (-28.0, 74.0), (-28.0, 52.0)])
    return w.offset(-3.0, JoinType.Round).offset(3.0, JoinType.Round)


def motor1_T():
    """받침 회전 모터: 회전판 위에 축을 아래로 두고 얹는다."""
    sx, sy = m1_shaft(); ux, uy = M1_BODY_DIR
    return np.array([[-ux, -uy, 0, sx], [-uy, ux, 0, sy], [0, 0, -1, DECK_Z1 + M1_PAD]], dtype=np.float64)


def motor2_T():
    """끄덕임 모터: 오른쪽(+Y) 팔 바깥면에 붙이고 축은 안쪽(-Y)으로. 몸통 중심은 축에서 M2_BODY_DIR 쪽(뒤 아래)이라
    전선이 나오는 파란 뚜껑이 팔 뒤 아래의 트인 자리에 온다: 전선이 거기서 바로 팔 안쪽으로 들어간다."""
    px, pz = pin2_xz(); a = np.radians(M2_BODY_DIR); c, s_ = np.cos(a), np.sin(a)
    return np.array([[-c, s_, 0, px], [0, 0, -1, ARM_Y1], [-s_, -c, 0, pz]], dtype=np.float64)


def m2_ears():
    """모터 2 귀 나사의 (x, z)"""
    T = motor2_T()
    return [tuple((T @ np.array([-M_OFFSET, s * M_EAR, 0, 1.0]))[[0, 2]]) for s in (1, -1)]


def m2_cap_xz():
    """모터 2 전선이 나오는 곳(파란 뚜껑 끝)의 (x, z)"""
    p = motor2_T() @ np.array([-M_OFFSET - 17.0, 0, 0, 1.0])
    return float(p[0]), float(p[2])


def m1_ears():
    T = motor1_T()
    return [tuple((T @ np.array([-M_OFFSET, s * M_EAR, 0, 1.0]))[:2]) for s in (1, -1)]


def hood_z(x):
    """지붕 윗면의 높이 (45도)"""
    return HOOD_Z0 + (x - WING_X0)


def cheek_cs():
    """오른쪽 칸막이의 옆모습 (요크 XZ). 머리 아래는 낮고, 머리가 지나는 원(반지름 37.8) 밖에서 뒷가림 벽을 따라 올라갔다가 지붕 선을 따라 뒷벽으로 내려온다."""
    t = BOX_T
    return poly([(WING_X1, DECK_Z1 - 0.1), (WING_X1, CHEEK_Z), (-40.0, CHEEK_Z), (HOOD_X1, hood_z(HOOD_X1)), (HOOD_X1, SHROUD_Z1), (HOOD_X1 - t, SHROUD_Z1),
                 (HOOD_X1 - t, hood_z(HOOD_X1 - t)), (WING_X0, HOOD_Z0), (WING_X0, DECK_Z1 - 0.1)])


def cable_box():
    """전선 상자: 앞벽, 뒷벽 + 45도 지붕 + 뒷가림 벽, 오른쪽 칸막이. 왼쪽은 왼팔이 막는다. 요크와 한 몸으로 출력한다 (모두 받침 없이 선다)."""
    t = BOX_T; zb = DECK_Z1 - 0.1
    front = box(WING_X1 - t, WING_X1, -ARM_Y0 - 0.1, CHEEK_Y, zb, BOX_ZF)
    # 앞벽 아래 한가운데는 받침의 가운데 통(반지름 13.5)과 고리가 지나는 자리라 문처럼 파 놓는다. 통이 그 구멍을 안에서 막고 있어 속은 안 보인다
    # 문 위는 45도로 넓게 깎는다: 받침 없이 뽑을 때 허공을 건너는 길이가 28mm 에서 20mm 로 준다 (slicing/print_check.py). 고리와 통에서 2mm 넘게 떨어진다
    front = front - prism_xy_to_yz(poly([(-18.0, zb - 1), (-18.0, DOOR[0]), (-DOOR[1], DOOR[2]), (DOOR[1], DOOR[2]), (18.0, DOOR[0]), (18.0, zb - 1)]), WING_X1 - t - 1, WING_X1 + 1)
    rear = box(WING_X0, WING_X0 + t, -ARM_Y0 - 0.1, ARM_Y0 + 0.1, zb, HOOD_Z0)
    rear_r = box(WING_X0, WING_X0 + t, CHEEK_Y - 0.1, ARM_Y0 + 0.1, zb, 86.0)              # 기어 칸의 뒤끝은 팔 높이까지 막는다
    th = t * np.sqrt(2.0)                                                                  # 45도로 기운 판을 수평으로 잰 두께 (수직 두께 = t)
    zt = HOOD_Z0 + (HOOD_X1 - WING_X0)
    hood = prism_xz(poly([(WING_X0, HOOD_Z0 - 0.1), (WING_X0, HOOD_Z0), (HOOD_X1, zt), (HOOD_X1, zt - th), (WING_X0 + t, HOOD_Z0 - 0.1)]), -ARM_Y0 - 0.1, CHEEK_Y)   # 아래끝은 뒷벽 두께에 딱 맞춘다 (턱이 안 생기게)
    # 뒷가림 벽: 지붕 끝에서 똑바로 선다. 머리를 기울이면 머리 아래가 뒤로 들리는데, 그 틈으로 전선이 보이지 않게 막는다 (머리 뒤 아래 모서리가 도는 반지름 37.8 보다 12mm 밖)
    shroud = box(HOOD_X1 - t, HOOD_X1, -ARM_Y0 - 0.1, CHEEK_Y, zt - th, SHROUD_Z1)
    ribs = [prism_xz(poly([(HOOD_X1 - t + 0.1, hood_z(HOOD_X1 - t) - 0.5), (HOOD_X1 - t + 0.1, SHROUD_Z1 - 8.0), (-74.0, hood_z(-74.0) - 0.5)]), y - t / 2, y + t / 2) for y in SHROUD_RIBS]   # 벽을 뒤에서 받치는 살 (지붕 위에 선다)
    cheek = prism_xz(cheek_cs(), CHEEK_Y - t, CHEEK_Y)
    return union([front, rear, rear_r, hood, shroud, cheek] + ribs)


M2_SLOT = (-77.0, -70.0, 45.5, 61.5)     # 칸막이의 모터 2 전선 구멍 (x0, x1, z0, z1): 흰 5핀 플러그(14.9 x 5.8)가 지나간다


def yoke():
    """요크: 회전판 + 팔 2개 + 전선 상자. 받침 위에서 돌고, 머리를 양쪽에서 잡는다. 출력: 그대로."""
    px, pz = pin2_xz()
    deck_cs = CS.circle(DECK_R) + rrect(WING_X0, WING_X1, -ARM_Y1, ARM_Y1, 5.0)
    deck = extrude_z(deck_cs, DECK_Z0, DECK_Z1)
    arms = [prism_xz(arm_profile() - arm_window(), ARM_Y0, ARM_Y1), prism_xz(arm_profile(), -ARM_Y1, -ARM_Y0)]      # 창은 오른팔에만 (기어가 보인다). 왼팔은 막아서 상자 속이 안 보인다
    gus = []
    for x in (-68.0, -38.0, 4.0):                                              # 오른쪽 팔 안쪽의 낮은 받침살 (부채꼴 기어 아래). 왼쪽 팔은 앞벽·뒷벽이 붙잡아서 받침살이 필요 없다
        gus.append(prism_xy_to_yz(poly([(ARM_Y0 + 0.1, DECK_Z1 - 0.1), (ARM_Y0 + 0.1, DECK_Z1 + 3.5), (ARM_Y0 - 3.5, DECK_Z1 - 0.1)]), x - 1.5, x + 1.5))
    sx, sy = m1_shaft(); ux, uy = M1_BODY_DIR
    pad_cs = CS.circle(15.0).translate([sx + ux * M_OFFSET, sy + uy * M_OFFSET]) + CS.batch_hull([CS.circle(4.2).translate(list(e)) for e in m1_ears()])
    pad = extrude_z(pad_cs, DECK_Z1 - 0.1, DECK_Z1 + M1_PAD)                                          # 모터 1 자리 돋움 (귀 나사가 아래로 나오지 않게)
    m = union([deck, pad, cable_box()] + arms + gus)
    cut = [box(WING_X1, 100, -ARM_Y0 + 1, ARM_Y0 - 1, DECK_ZF, DECK_Z1 + 1)]                            # 앞쪽 윗면을 1mm 낮춘다
    cut.append(cz(TUBE_RO + 0.25, DECK_Z0 - 1, DECK_Z1 + 1))
    cut.append(cz(4.9, DECK_Z0 - 1, DECK_Z1 + M1_PAD + 1, sx, sy))                                     # 모터 1 축 보스
    for ex, ey in m1_ears():
        cut.append(cz(1.3, DECK_Z0 - 1, DECK_Z1 + M1_PAD + 1, ex, ey))                                 # 귀 나사 (M3x6: 5.2mm 들어가고 0.3mm 남는다)
    cut.append(cz(1.3, DECK_Z0 - 1, DECK_Z1 + 1, *YAW_STOP_XY))                                        # 받침 회전 멈춤 나사
    hole = prism_xz(teardrop_cs(PIN_R_LOOSE, 90.0), -ARM_Y1 - 1, ARM_Y1 + 1).translate([0, 0, H_T])     # 끄덕임 축 구멍 (핀이 헐겁게 돈다)
    cut.append(hole)
    cut.append(prism_xz(teardrop_cs(M2_HOLE_R, 90.0), ARM_Y0 - 1, ARM_Y1 + 1).translate([px, 0, pz]))   # 모터 2 구멍: 작은 기어를 축에 끼운 채로 바깥에서 넣는다 (물방울: 받침 없이 출력)
    bx, bz = BAND_ARM
    cut.append(prism_xz(poly([(bx + 1, bz + 5.2), (bx - 3.0, bz + 1.2), (bx - 3.0, bz - 1.2), (bx + 1, bz - 2.2)]), ARM_Y0 - 1, ARM_Y1 + 1))        # 고무줄 홈 (오른쪽 팔 앞 모서리, 윗면 45도)
    x0, x1, z0, z1 = M2_SLOT
    cut.append(box(x0, x1, CHEEK_Y - BOX_T - 1, CHEEK_Y + 1, z0, z1))                                  # 모터 2 전선이 칸막이를 지나 상자로 들어가는 구멍
    for ex, ez in m2_ears():
        cut.append(cy(1.3, ARM_Y0 + 0.6, ARM_Y1 + 1, ex, ez))                                          # 모터 2 귀 나사 (막힌 구멍: 긴 나사가 부채꼴 기어 쪽으로 나오지 않게)
    lx, lz = lock_xz()
    cut.append(cy(1.7, ARM_Y0 - 1, ARM_Y1 + 1, lx, lz))                                                # 끄덕임 고정 나사 구멍
    for x, z in stop_screws_xz():
        cut.append(cy(1.3, -ARM_Y1 - 1, -ARM_Y0 + 1, x, z))                                            # 끄덕임 멈춤 나사 2개
    for k, p in enumerate(LOCK_ANGLES + (85.0,)):                                                      # 오른쪽 팔 바깥면 각도 눈금 (0, 54.7, 85)
        a = np.radians(270.0 - p); r0, r1 = 15.5, 20.5
        tick = box(-0.6, 0.6, ARM_Y1 - 0.6, ARM_Y1 + 1, r0, r1).rotate([0, np.degrees(-(a - np.pi / 2)), 0]).translate([0, 0, H_T])
        cut.append(tick)
    return m - union(cut)


def pinion(phase=0.0, d=D_BORE_D, flat=D_BORE_FLAT):
    """작은 기어 10T. 모터 1과 모터 2에 같은 것을 하나씩 쓴다. 출력: 그대로."""
    g = pinion_cs(MOD, ZP, P_SHIFT, P_RA)
    return extrude_z(g.rotate(phase) - d_bore_cs(d, flat).rotate(phase), 0, 5.5)


def pinion1():
    sx, sy = m1_shaft()
    return pinion(8.0).translate([sx, sy, DECK_Z0 - 6.0])


def pinion2():
    px, pz = pin2_xz()
    return pinion(0.0).rotate([90, 0, 0]).translate([px, PIN2_Y1, pz])


def m2_ring():
    """모터 2 맞춤 링: 모터 2의 축 둘레 턱(지름 9)에 씌우고, 그 뒤에 작은 기어를 끼운다. 팔의 큰 구멍(지름 16.5)을 채워서 모터 자리를 잡아 준다.
    (작은 기어를 끼운 채로 넣으려고 구멍을 키운 뒤로는 귀 나사 2개만 자리를 잡았는데, 귀 구멍이 헐거워서 기어 사이가 ±0.6mm 까지 달라질 수 있었다.)
    출력: 턱 자리(넓은 구멍)를 위로."""
    px, pz = pin2_xz()
    r, L = M2_RING_R, M2_RING_L; rb, d = M2_RING_BOSS
    m = revolve_profile([(M2_RING_BORE, 0), (r - 0.4, 0), (r, 0.4), (r, L), (rb, L), (rb, L - d), (M2_RING_BORE, L - d)])     # z = 0 이 안쪽 끝(모서리를 깎았다), z = L 이 모터 앞면
    return m.rotate([-90, 0, 0]).translate([px, ARM_Y1 - L, pz])


def pin(side=1, r=PIN_FITS[2]):
    """끄덕임 축 핀 (좌우 1개씩). 출력: 머리를 바닥에 세워서."""
    y_in = HX - HW - BOSS_T              # 머리 옆벽 + 덧살 두께만큼 들어간다
    if side > 0:
        ptr = prism_xz(poly([(-2.2, -5.0), (2.2, -5.0), (0.0, -14.0)]), ARM_Y1, ARM_Y1 + 2.0).translate([0, 0, H_T])     # 각도 바늘 (0도에서 아래를 가리키게 꽂는다)
        return cy(r, y_in, ARM_Y1, 0, H_T) + cy(7.0, ARM_Y1, ARM_Y1 + 2.0, 0, H_T) + ptr
    return cy(r, -ARM_Y1, -y_in, 0, H_T) + cy(7.0, -ARM_Y1 - 2.0, -ARM_Y1, 0, H_T)


# ----------------------------------------------------------------- 머리 (머리 좌표)
def pi_hole_xz():
    return [(BX0 + b, BZ1 - a) for a, b in PI_HOLES]


def sector_mount_yz():
    """부채꼴 기어 고정 나사의 머리 좌표 (Y, Z)"""
    out = []
    for r, a in SEC_MOUNT:
        x, z = pol(r, a)
        out.append((Y_T - x, S_T + z))
    return out


def sector_gear():
    """부채꼴 기어 (112T 상당, 11.2:1). 머리 오른쪽 옆면에 나사 2개로 붙어 머리와 같이 돈다. 요크 좌표(끄덕임 0도)로 만든다."""
    g = gear_big(MOD, SEC_T).rotate(PIN_DIR + 360.0 / SEC_T / 2 * 0)
    wedge = poly([(0, 0)] + [pol(95.0, a) for a in np.linspace(SEC_LO, SEC_HI, 40)])
    rim = (g ^ wedge) - CS.circle(SEC_R - 9.0)
    band = sector(LOCK_R - 5.0, LOCK_R + 5.0, SEC_LO, SEC_HI, 40)
    spokes = CS()
    for a in (SEC_LO + 4.0, 250.0, 286.0, SEC_HI - 4.0):
        d = np.array(pol(1.0, a)); n = np.array([-d[1], d[0]]) * 3.5
        spokes = spokes + poly([d * 8 + n, d * (SEC_R - 7) + n, d * (SEC_R - 7) - n, d * 8 - n])
    cs = rim + band + spokes + CS.circle(23.0)
    cs = cs - CS.circle(PIN_R_TIGHT + 0.1)
    for r, a in SEC_MOUNT:
        cs = cs - CS.circle(1.7).translate(list(pol(r, a)))
    for p in LOCK_ANGLES:
        cs = cs - CS.circle(1.3).translate(list(pol(LOCK_R, LOCK_DIR + p)))
    cs = cs - CS.circle(1.6).translate(list(pol(*BAND_SEC)))                # 고무줄 구멍
    m = prism_xz(cs, SEC_Y0, SEC_Y1).translate([0, 0, H_T])
    for r, a in SEC_MOUNT:                                                  # 나사 머리 자리 (바깥면에서 파 들어간다)
        x, z = pol(r, a)
        m = m - cy(SEC_CB_R, SEC_Y1 - SEC_CB_D, SEC_Y1 + 1, x, H_T + z)
    return m


def sector_head():
    return yoke_to_head(sector_gear())


def _yz_prism(pts, x0, x1):
    return prism_xy_to_yz(poly(pts), x0, x1)


def button_seat():
    """전원 버튼 모듈(KY-004)이 앉는 자리: 뒷벽에서 올라온 살 두 줄. 기판 뒷면을 폼테이프로 살 위에 붙이고, 윗변은 턱에, 왼쪽 변은 옆벽에 댄다.
    버튼은 앞(뚜껑 쪽)을 보고, 누르는 힘은 살이 받는다. 전원선은 두 살 사이로 지나간다."""
    xi = HX - HW; yb = HY1 - HWB
    ys = BTN_PCB_Y + 1.6                                   # 기판 뒷면이 닿는 면
    zt = BTN_Z + BTN_TOP; zb = zt - BTN_L                  # 기판 윗변, 아랫변
    lip = 2.0                                              # 턱이 기판 뒷면보다 앞으로 나온 높이
    out = [box(-xi - 0.1, -xi + 2.1, ys, yb + 0.1, zb - 1.7, zt + 1.7)]                       # 벽 쪽 살 (기판 왼쪽 변 밑)
    out.append(box(-xi - 0.1, -xi + 2.1, ys - lip, ys + 0.1, zt + 0.2, zt + 1.7))             # 윗턱
    out.append(box(-xi - 0.1, -xi + 2.1, ys - lip, ys + 0.1, zb - 1.7, zb - 0.2))             # 아랫턱 (핀이 없는 왼쪽 모서리만)
    x0, x1 = BTN_X + 0.2, BTN_X + 2.0                                                         # 가운데 살: 버튼 바로 뒤. 버튼 다리의 납땜 자국(±3.25) 사이
    out.append(box(x0, x1, ys, yb + 0.1, zb + 5.5, zt + 1.7))
    out.append(box(x0, x1, ys - lip, ys + 0.1, zt + 0.2, zt + 1.7))
    out.append(box(-xi + 2.0, x0 + 0.1, ys + 2.5, ys + 3.7, zt - 6.0, zt + 1.7))              # 두 살을 잇는 다리 (납땜 자국보다 2.5mm 뒤)
    return out


def head_shell():
    """머리 몸통: 뒷벽과 옆벽. 파이와 드라이버 3장이 들어간다. 위는 모터 자리, 앞은 뚜껑이 닫는다. 출력: 뒷벽을 바닥에."""
    xi = HX - HW; yb = HY1 - HWB
    back = box(-HX, HX, yb, HY1, HZ0, HZ1)
    sides = box(-HX, -xi, HYL, HY1, HZ0, HZ1) + box(xi, HX, HYL, HY1, HZ0, HZ1)
    add = [back, sides]
    for x, z in pi_hole_xz():                                                  # 파이 받침 기둥 + 꽂는 못
        add.append(cy(3.0, PI_T / 2, yb + 0.1, x, z))
        add.append(cy(PEG_R, -3.0, PI_T / 2 + 0.1, x, z))
    for sx, sy in CROWN_SCREWS:                                                # 모터 자리 고정 나사 받이 (모서리 덩어리)
        s = 1 if sx > 0 else -1
        x0, x1 = (sx - 3.5, xi + 0.1) if s > 0 else (-xi - 0.1, sx + 3.5)
        if sy > -10:                                                           # 뒤쪽: 뒷벽에 붙는다
            add.append(box(x0, x1, sy - 3.5, yb + 0.1, 1.5, HZ1))
        else:                                                                  # 앞쪽: 옆벽에 붙고 아래(출력할 때 바닥 쪽)는 45도로 받친다
            add.append(box(x0, x1, sy - 3.5, sy + 3.5, 1.5, HZ1))
            xa, xb = (x0, x1) if s > 0 else (x1, x0)
            add.append(extrude_z(poly([(xa, sy + 3.3), (xb, sy + 3.3), (xb, sy + 3.3 + 1.5 * abs(x1 - x0))]), 1.5, HZ1))
    for sx, sz in LID_SCREWS:                                                  # 뚜껑 나사 받이
        s = 1 if sx > 0 else -1; w = LID_BOSS_W[s]
        x0, x1 = (sx - w, xi + 0.1) if s > 0 else (-xi - 0.1, sx + w)
        add.append(box(x0, x1, HYL, HYL + 7.0, sz - 3.5, sz + 3.5))
        xa, xb = (x0, x1) if s > 0 else (x1, x0)
        add.append(extrude_z(poly([(xa, HYL + 6.8), (xb, HYL + 6.8), (xb, HYL + 6.8 + 1.5 * abs(x1 - x0))]), sz - 3.5, sz + 3.5))
    for s in (1, -1):                                                          # 끄덕임 축 둘레 보강 (뒷벽에서 올라오는 살)
        x0, x1 = (xi - BOSS_T, xi + 0.1) if s > 0 else (-xi - 0.1, -xi + BOSS_T)
        add.append(box(x0, x1, HYL, yb + 0.1, S_T - 8.0, S_T + 8.0))
    for (y, z) in sector_mount_yz():                                           # 부채꼴 기어 나사 받이 (오른쪽)
        add.append(box(xi - BOSS_T, xi + 0.1, y - 3.5, yb + 0.1, z - 3.5, z + 3.5))
    # 왼쪽 바깥: 간격 보스(팔과 머리 사이를 채운다)와 멈춤 돌기. 아래쪽(출력할 때 바닥 쪽 = +Y)은 비스듬히 받친다
    boss = cx(8.0, -HX - 4.5, -HX + 0.1, Y_T, S_T)
    add.append(M.batch_hull([boss, box(-HX - 0.3, -HX + 0.1, Y_T + 8.0, Y_T + 17.5, S_T - 4.0, S_T + 4.0)]))
    lz = S_T - LUG_R
    # 멈춤 돌기: 앞쪽 면(-Y)은 평평하고, 출력할 때 바닥을 보는 면(+Y)은 45도로 깎아 받침 없이 뽑는다
    add.append(extrude_z(poly([(-HX + 0.1, Y_T - 3.0), (-HX - 4.5, Y_T - 3.0), (-HX - 4.5, Y_T - 1.5), (-HX + 0.1, Y_T + 3.1)]), lz - 2.5, lz + 2.5))
    add += button_seat()
    m = union(add)
    cut = []
    tight = teardrop_cs(PIN_R_TIGHT, 180.0)                                    # 꼭지가 -Y(출력할 때 위)
    cut.append(prism_xy_to_yz(tight, -HX - 5, -xi + 5).translate([0, Y_T, S_T]))
    cut.append(prism_xy_to_yz(tight, xi - 5, HX + 1).translate([0, Y_T, S_T]))
    for sx, sy in CROWN_SCREWS:
        cut.append(cz(1.3, 0.5, HZ1 + 1, sx, sy))
    for sx, sz in LID_SCREWS:
        cut.append(cy(1.3, HYL - 1, HYL + 6.4, sx, sz))
    for (y, z) in sector_mount_yz():
        cut.append(cx(1.3, xi - 5, HX + 1, y, z))
    wz0, wz1, wy0, wy1 = PLUG_WIN; c = 4.0                                      # 전원 플러그 창 (왼쪽 옆벽). 출력할 때 위가 되는 두 모서리(-Y)는 45도로 깎는다
    cut.append(prism_xy_to_yz(poly([(wy1, wz0), (wy1, wz1), (wy0 + c, wz1), (wy0, wz1 - c), (wy0, wz0 + c), (wy0 + c, wz0)]), -HX - 1, -xi + 0.05))
    for k in range(7):                                                         # 뒷벽 통풍 틈 (파이 뒤)
        x = -38.0 + k * 8.0
        cut.append(M.batch_hull([cy(1.3, yb - 1, HY1 + 1, x, -22.0), cy(1.3, yb - 1, HY1 + 1, x, -50.0)]))
    c = 1.5                                                                    # 뒤 모서리 깎기
    for s in (1, -1):
        cut.append(extrude_z(poly([(s * (HX - c), HY1 + 0.01), (s * (HX + 0.01), HY1 + 0.01), (s * (HX + 0.01), HY1 - c)]), HZ0 - 1, HZ1 + 1))
    return m - union(cut)


def head_lid():
    """머리 뚜껑 (앞면). 나사 4개. 출력: 바깥면을 바닥에."""
    m = box(-HX + 0.15, HX - 0.15, HY0, HYL, HZ0, HZ1)                         # 옆은 0.15 씩 좁게: 옆에 붙는 부채꼴 기어를 밀지 않게
    holes = pi_hole_xz()
    for k, (x, z) in enumerate(holes):                                          # 파이 누름 기둥 4개 (못 끝이 들어간다). 위쪽 둘은 전원 단자와 GPIO 핀을 피해 가늘게
        m = m + cy(2.6 if k < 2 else 3.0, HYL - 0.1, -PI_T / 2 - 0.3, x, z)
    cut = []
    for x, z in holes:
        cut.append(cy(1.6, -4.0, 0, x, z))
    for sx, sz in LID_SCREWS:
        cut.append(cy(1.7, HY0 - 1, HYL + 1, sx, sz))
    eye = M.batch_hull([cy(4.5, HY0 - 1, HY0 + 0.8, -30.0, -20.0), cy(4.5, HY0 - 1, HY0 + 0.8, 30.0, -20.0)])     # 앞면 '눈' 띠 (0.8 깊이)
    cut.append(eye)
    for k in range(8):                                                         # 통풍 틈 (파이 칩 앞)
        x = -38.0 + k * 6.5
        cut.append(M.batch_hull([cy(1.2, HY0 - 1, HYL + 1, x, -36.0), cy(1.2, HY0 - 1, HYL + 1, x - 6.0, -62.0)]))
    for z in (-74.0,):                                                         # 패널 선
        cut.append(box(-HX - 1, HX + 1, HY0 - 1, HY0 + 0.6, z - 0.5, z + 0.5))
    cut.append(cy(BTN_HOLE_R, HY0 - 1, HYL + 1, BTN_X, BTN_Z))                 # 전원 버튼 구멍: 안쪽의 버튼을 손끝으로 누른다
    cut.append(cy(BTN_HOLE_R + 1.5, HY0 - 1, HY0 + 0.5, BTN_X, BTN_Z))         # 구멍 둘레를 0.5 낮춘다 (손끝이 들어가는 자리)
    lx0, lx1, lz0, lz1 = LED_WIN; r = (lz1 - lz0) / 2
    cut.append(M.batch_hull([cy(r, HY0 - 1, HYL + 1, lx0 + r, lz0 + r), cy(r, HY0 - 1, HYL + 1, lx1 - r, lz0 + r)]))     # 상태등 창
    c = 1.5
    for s in (1, -1):
        cut.append(extrude_z(poly([(s * (HX - c), HY0 - 0.01), (s * (HX + 0.01), HY0 - 0.01), (s * (HX + 0.01), HY0 + c)]), HZ0 - 1, HZ1 + 1))
    return m - union(cut)


def crown():
    """모터 자리: 머리 위에 얹는 바닥판과 둥근 벽. 판 회전 모터(J3)가 들어간다. 출력: 그대로."""
    z0 = HZ1; zf = z0 + 2.0
    plate = extrude_z(rrect(-HX, HX, HY0, HY1, 1.5), z0, zf)
    floor = cz(SEAT_R, z0, zf)
    skirt = cz(SEAT_R, zf - 0.1, M1_ZT) - cz(SEAT_R - 1.4, zf, M1_ZT + 1)
    posts = [cz(3.0, zf - 0.1, M1_ZT - 0.8, -M_OFFSET, s * M_EAR) for s in (1, -1)]
    posts += [box(-M_OFFSET - 1, -M_OFFSET + 1, 17.5 if s > 0 else -34.5, 34.5 if s > 0 else -17.5, zf - 0.1, M1_ZT - 4) for s in (1, -1)]
    m = union([plate, floor, skirt] + posts)
    cut = [cz(1.3, zf + 5, M1_ZT + 1, -M_OFFSET, s * M_EAR) for s in (1, -1)]
    for sx, sy in CROWN_SCREWS:
        cut.append(cz(1.7, z0 - 1, zf + 1, sx, sy))
    x0, x1, y0, y1 = CABLE_HOLE
    cut.append(box(x0, x1, y0, y1, z0 - 1, zf + 1))
    za, zb = zf + 5.0, M1_ZT - 3.0
    slot = prism_xy_to_yz(poly([(-2.2, za), (2.2, za), (2.2, zb - 2.2), (0, zb), (-2.2, zb - 2.2)]), SEAT_R - 3, SEAT_R + 1)
    for k in range(8):
        cut.append(rot_z(slot, 22.5 + 45 * k))
    cut.append(box(SEAT_R - 2.2, SEAT_R + 1, -0.8, 0.8, M1_ZT - 3.0, M1_ZT + 1))                         # 판 0도 표시 (벽 윗가의 홈, +X)
    return m - union(cut)


# ----------------------------------------------------------------- 회전부 (머리 좌표, 판은 +X 쪽)
def blade(length=150.0):
    """판 받침. 바깥면(반지름 80)에 알루미늄 테이프를 붙인다. 출력: 바깥면을 바닥에."""
    hw = PLATE_W / 2; z0, z1 = BL_Z0, BL_Z0 + length
    x1 = R_FOIL; x0 = x1 - PANEL_T; rib = 3.0
    panel = box(x0, x1, -hw, hw, z0, z1)
    frame = box(x0 - rib, x0 + 0.1, -hw, hw, z0, z1) - box(x0 - rib - 1, x0 + 1, -hw + 2.4, hw - 2.4, z0 + 2.4, z1 - 2.4)
    zc = (BAR_Z0 + BAR_Z1) / 2; zt = zc + 6.0                                   # 막대 높이, 꽂는 자리의 윗면
    sx = COLLAR_X0 + 3.6; yw = 5.6                                              # 고정 나사 자리, 갈라 놓은 세로 살의 위치
    ribs = [box(x0 - rib, x0 + 0.1, -hw, hw, z - 1.0, z + 1.0) for z in np.arange(z0 + 46.0, z1 - 20.0, 46.0)]
    ribs.append(box(x0 - rib, x0 + 0.1, -1.0, 1.0, z0, zc))                     # 세로 살: 꽂는 자리 아래는 가운데 한 줄
    collar = box(COLLAR_X0, x0 + 0.1, -9.0, 9.0, zc - 6.0, zt)
    web = [box(x0 - 6.0, x0 + 0.1, -1.2, 1.2, zc - 26.0, zc), box(x0 - 6.0, x0 + 0.1, -26.0, 26.0, zc - 1.2, zc + 1.2)]
    for s in (1, -1):                                                           # 꽂는 자리 위는 두 줄로 가른다: 그 사이로 나사와 드라이버가 들어간다
        ribs.append(box(x0 - rib, x0 + 0.1, s * yw - 1.0, s * yw + 1.0, zc, z1))
        web.append(box(x0 - 6.0, x0 + 0.1, s * yw - 1.2, s * yw + 1.2, zc, zc + 26.0))
    m = union([panel, frame, collar] + ribs + web)
    cut = [box(COLLAR_X0 - 1, x0 - 0.6, -TENON_W / 2 - 0.15, TENON_W / 2 + 0.15, BAR_Z0 - 0.15, BAR_Z1 + 0.15)]      # 막대 끝이 들어가는 구멍
    cut.append(cz(1.7, BAR_Z1, zt + 1.0, sx, 0))                                                                   # 고정 나사 (위에서)
    cut.append(cz(3.7, zt, z1 + 1.0, sx, 0))                                                                       # 드라이버 길 (지름 7.4): 위쪽 테두리와 가로 살을 판 두께만 남기고 판다
    return m - union(cut)


def foil(length=150.0):
    return box(R_FOIL, R_FOIL + 0.15, -PLATE_W / 2, PLATE_W / 2, BL_Z0, BL_Z0 + length)


def rotor_bar(d=D_BORE_D, flat=D_BORE_FLAT):
    """회전 막대: 가운데가 모터 축, 한쪽 끝에 판, 반대쪽 끝은 균형추. 속을 꽉 채워(100%) 출력한다. 출력: 뒤집어서."""
    hub = cz(9.5, M1_ZT + 2.0, BAR_Z1)
    mid = chbox(-9.5, 9.5, -8.0, 8.0, BAR_Z0, BAR_Z1, 1.2)
    arm1 = M.batch_hull([mid, chbox(62.0, COLLAR_X0 - 0.3, -6.0, 6.0, BAR_Z0, BAR_Z1, 1.2)])
    tenon = box(COLLAR_X0 - 1.5, TENON_X1, -TENON_W / 2, TENON_W / 2, BAR_Z0, BAR_Z1)
    arm2 = M.batch_hull([mid, chbox(-POD_R0 - 4, -POD_R0 + 2, -8.0, 8.0, BAR_Z0, BAR_Z1, 1.2)])
    pod = chbox(-POD_R1, -POD_R0, -POD_W / 2, POD_W / 2, POD_Z0, BAR_Z1, 2.5)
    m = union([hub, mid, arm1, tenon, arm2, pod])
    cut = [cz(2.7, M1_ZT + 1.0, M1_ZT + 4.05)]                                                  # 축의 둥근 부분
    cut.append(extrude_z(d_bore_cs(d, flat), M1_ZT + 4.0, M1_ZT + 10.0))                        # 축의 납작한 부분 (위는 1mm 막힘)
    cut.append(cz(1.3, BAR_Z0 - 1, BAR_Z1 + 1, COLLAR_X0 + 3.6, 0))                             # 판 고정 나사
    zt = (POD_Z0 + BAR_Z0) / 2
    cut.append(prism_xy_to_yz(teardrop_cs(1.3, -90.0), -POD_R1 - 1, -POD_R1 + 12.0).translate([0, 0, zt]))      # 균형 조각 나사 (뒤집어 출력: 꼭지가 -Z)
    cut.append(box(8.2, 9.6, -0.8, 0.8, BAR_Z1 - 0.6, BAR_Z1 + 1))                              # 0도 표시
    return m - union(cut)


def trim_plate():
    """균형 조각: 균형추 바깥면에 나사로 덧붙여 무게를 맞춘다."""
    zt = (POD_Z0 + BAR_Z0) / 2
    m = chbox(-POD_R1 - 3.0, -POD_R1, -18.0, 18.0, zt - 8.0, zt + 8.0, 0.6)
    return m - cx(1.7, -POD_R1 - 4, -POD_R1 + 1, 0, zt)


def balance_200():
    """판 100 x 200 을 쓸 때 균형추 바깥면에 덧붙이는 덩어리 (속을 꽉 채워 출력)."""
    zt = (POD_Z0 + BAR_Z0) / 2
    m = chbox(-POD_R1 - 9.0, -POD_R1, -18.0, 18.0, zt - 8.5, zt + 8.5, 1.0)
    return m - cx(1.7, -POD_R1 - 10, -POD_R1 + 1, 0, zt)


def test_coupon():
    """맞춤 시험 조각: 모터 축 구멍 3종(점 개수 = 번호), 세운 벽의 핀 구멍 2종(빡빡/헐렁), 판 꽂는 구멍, M3 나사 구멍 2종,
    파이 기둥 못 2종(지름 2.5 = 머리 몸통과 같은 것, 2.3 = 옆에 점이 있는 것)."""
    m = box(0, 62, 0, 34, 0, 6)
    for k, (px_, r_) in enumerate(((46.0, PEG_R), (56.0, PEG_R - 0.1))):          # 파이 기둥과 못: 머리 몸통에서와 같은 모양, 같은 방향(세워서)으로 뽑힌다
        m = m + cz(3.0, 5.9, 7.6, px_, 21.0) + cz(r_, 7.5, 7.6 + 3.8, px_, 21.0)
    cut = []
    for i in (1, 2, 3):
        d, f, _ = FITS[i]
        xc = 9.0 + (i - 1) * 12.0
        cut.append(extrude_z(d_bore_cs(d, f), -1, 7).translate([xc, 10.5, 0]))
        for k in range(i):
            cut.append(cz(0.9, 5.2, 7, xc + (k - (i - 1) / 2) * 2.6, 3.4))
    m = m + box(38.0, 62.0, 0.0, 6.0, 5.9, 24.0)
    cut.append(prism_xz(teardrop_cs(PIN_R_TIGHT, 90.0), -1, 7).translate([44.5, 0, 14.5]))
    cut.append(prism_xz(teardrop_cs(PIN_R_LOOSE, 90.0), -1, 7).translate([55.5, 0, 14.5]))
    cut.append(box(53.5, 57.5, 2.6, 3.4, 23.2, 25))                               # 헐거운 구멍 표시 (벽 윗면의 짧은 줄)
    c = 0.15
    cut.append(box(-TENON_W / 2 - c, TENON_W / 2 + c, -3 - c, 3 + c, -1, 7).translate([10, 25, 0]))      # 판 꽂는 구멍 (10 x 6)
    cut.append(cz(1.3, -1, 7, 24.0, 25.0))
    cut.append(cz(1.7, -1, 7, 32.0, 25.0))
    cut.append(cz(0.9, 5.2, 7, 56.0, 28.5))                                       # 가는 못(2.3) 표시
    return m - union(cut)


def tenon_gauge():
    """시험용 돌기 (막대 끝과 같은 10 x 6)"""
    return box(-9, 9, -6, 6, 0, 3) + box(-TENON_W / 2, TENON_W / 2, -3, 3, 0, 10)


# ----------------------------------------------------------------- 구매 부품 (모양만)
def driver_T(k):
    """드라이버 k (0 = 맨 위). 뒷벽에 폼테이프로 붙이고 부품은 앞(-Y)을 향한다."""
    return np.array([[1, 0, 0, DRV_X0 + 17.5], [0, 0, -1, HY1 - HWB - 1.5], [0, 1, 0, DRV_ZC[k]]], dtype=np.float64)


def driver_keepout(k):
    """드라이버 보드 + 꽂힌 점퍼선이 차지하는 부피 (보드 윗면에서 24mm)"""
    return box(-17.5, 17.5, -16, 16, 0, 1.6 + 24.0).transform(driver_T(k))


def gpio_keepout():
    """파이 GPIO 핀에 꽂힌 점퍼선이 차지하는 부피"""
    t = PI_T / 2
    return box(BX0 + 50, BX0 + 55.1, -t - 23.0, -t, BZ1 - 57.9, BZ1 - 7.1)


def usb_plug_dummy():
    t = PI_T / 2
    plug = box(BX0 - 10.3, BX0 - 1.2, -t - 6.0, -t - 0.2, BZ1 - 15, BZ1 - 6.5)
    cable = cz(1.9, HZ0 - 14, BZ1 - 14, BX0 - 6.6, -t - 3.4)             # 버튼 모듈 자리의 두 살 사이(폭 5.9)로 내려간다
    return plug + cable


def battery_dummy(size="S"):
    b = BASE[size]; R = b["R"]
    lx, ly = b["bw"] - 1.0, b["bl"] - 1.0
    y0 = -R + BATT_IN + 0.5
    return box(-lx / 2, lx / 2, y0, y0 + ly, 2.3, 2.3 + BATT_T)


def usb_a_dummy(size="S"):
    """배터리 쪽 ㄱ자 USB 플러그: 몸통이 배터리 끝면에서 9mm 나오고 옆으로 눕는다"""
    y1 = -BASE[size]["R"] + BATT_IN + 0.5
    return box(-14.0, 14.0, y1 - 9.0, y1, 5.0, 15.0)


def lan_plug_dummy():
    """파이 랜 단자에 꽂힌 플러그(덮개 포함)와 선 조금. 아래(-Z)로 나온다"""
    t = PI_T / 2; xc = BX0 + 10.25
    plug = box(xc - 6.5, xc + 6.5, -t - 13.0, -t - 1.5, BZ0 - 27.0, BZ0 - 2.0)
    return plug + cz(3.0, BZ0 - 35.0, BZ0 - 26.0, xc, -t - 7.0)


def dongle_dummy(d=None):
    """무선 동글. 파이의 USB 에 꽂혀 아래(-Z)로 나온다. 머리 치마 안에 거의 다 들어간다.
    d = DONGLE: 폰과 통신하는 동글 (GPIO 쪽 USB).  d = DONGLE2: 탐지용 모니터 모드 동글 (가운데 USB)"""
    d = d or DONGLE; t = PI_T / 2; xc = BX0 + d["x"]; yc = -t - d["up"]; z1 = BZ0 - 2.0
    return box(xc - d["W"] / 2, xc + d["W"] / 2, yc - d["T"] / 2, yc + d["T"] / 2, z1 - (d["L"] - d["plug"]), z1)


def button_dummy():
    """전원 버튼 모듈 KY-004: 기판, 버튼, 꽂힌 점퍼선 2가닥"""
    zt = BTN_Z + BTN_TOP; zb = zt - BTN_L; y = BTN_PCB_Y
    pcb = box(BTN_X - BTN_W / 2, BTN_X + BTN_W / 2, y, y + 1.6, zb, zt)
    sw = box(BTN_X - 3.0, BTN_X + 3.0, y - 3.5, y + 0.1, BTN_Z - 3.0, BTN_Z + 3.0) + cy(1.75, y - 5.0, y - 3.4, BTN_X, BTN_Z)
    wires = box(BTN_X - 3.9, BTN_X + 3.9, y - 2.8, y - 0.2, zb - 14.5, zb + 2.5)
    return pcb, sw + wires


def lug_point(pitch):
    """끄덕임 멈춤 돌기의 중심 (요크 XZ)"""
    x, z = pol(LUG_R, 270.0 - pitch)
    return x, H_T + z


# ----------------------------------------------------------------- 조립
COL = dict(body=MIDNIGHT, accent="#2B3F7A", foil="#C9CED4", metal="#A9AEB3", cap="#3F78C9", pcb="#2F7D4F", dark="#22262B",
           ant="#E0533A", batt="#50565E", screw="#C9CCD1")


def assembly(yaw=0.0, pitch=0.0, spin=0.0, size="S", length=150.0, dummies=True, col=None, lan=False):
    """이름 -> (형상, 무리, 색, 출력부품 여부). lan=False: 혼자 동작하는 모습(랜선 없이 홈 마개 + 무선 동글). lan=True: 개발할 때(랜선을 꽂은 모습)"""
    c = dict(COL); c.update(col or {})
    H = lambda m: pose_head(m, yaw, pitch)
    R = lambda m: pose_head(m, yaw, pitch, spin)
    Y = lambda m: pose_yoke(m, yaw)
    A = {}
    A["tub"] = (tub(size), "B", c["body"], True)
    A["cover"] = (cover(size), "B", c["accent"], True)
    A["clip"] = (clip(), "B", c["accent"], True)
    if not lan and size == "S":
        A["lan_cap"] = (lan_cap(), "B", c["accent"], True)
    A["yoke"] = (Y(yoke()), "Y", c["body"], True)
    A["pinion1"] = (Y(pinion1()), "Y", c["accent"], True)
    A["pinion2"] = (Y(pinion2()), "Y", c["accent"], True)
    A["m2_ring"] = (Y(m2_ring()), "Y", c["accent"], True)
    A["pin_r"] = (Y(pin(1)), "Y", c["accent"], True)
    A["pin_l"] = (Y(pin(-1)), "Y", c["accent"], True)
    A["head_shell"] = (H(head_shell()), "H", c["body"], True)
    A["head_lid"] = (H(head_lid()), "H", c["body"], True)
    A["crown"] = (H(crown()), "H", c["accent"], True)
    A["sector"] = (H(sector_head()), "H", c["accent"], True)
    A["rotor_bar"] = (R(rotor_bar()), "R", c["accent"], True)
    A["blade"] = (R(blade(length)), "R", c["body"], True)
    if dummies:
        A["foil"] = (R(foil(length)), "R", c["foil"], False)
        mb, mc = v1.motor_body_dummy()
        A["motor3"] = (H(v1._m1(mb)), "H", c["metal"], False)
        A["motor3_cap"] = (H(v1._m1(mc)), "H", c["cap"], False)
        A["motor2"] = (Y(mb.transform(motor2_T())), "Y", c["metal"], False)
        A["motor2_cap"] = (Y(mc.transform(motor2_T())), "Y", c["cap"], False)
        A["motor1"] = (Y(mb.transform(motor1_T())), "Y", c["metal"], False)
        A["motor1_cap"] = (Y(mc.transform(motor1_T())), "Y", c["cap"], False)
        b, cc = v1.pi_dummy()
        A["pi_board"] = (H(b), "H", c["pcb"], False)
        A["pi_parts"] = (H(cc), "H", c["metal"], False)
        A["usb_plug"] = (H(usb_plug_dummy()), "H", c["dark"], False)
        if lan:
            A["lan_plug"] = (H(lan_plug_dummy()), "H", c["dark"], False)
        A["dongle"] = (H(dongle_dummy()), "H", c["dark"], False)
        A["dongle2"] = (H(dongle_dummy(DONGLE2)), "H", c["dark"], False)
        bp, bs = button_dummy()
        A["button_board"] = (H(bp), "H", c["pcb"], False)
        A["button_parts"] = (H(bs), "H", c["dark"], False)
        A["antenna"] = (H(v1.antenna_marker()), "H", c["ant"], False)
        db, dc = v1.driver_dummy()
        for k in range(3):
            A["driver%d_board" % (3 - k)] = (H(db.transform(driver_T(k))), "H", c["pcb"], False)
            A["driver%d_parts" % (3 - k)] = (H(dc.transform(driver_T(k))), "H", c["dark"], False)
        A["battery"] = (battery_dummy(size), "B", c["batt"], False)
        A["usb_a"] = (usb_a_dummy(size), "B", c["dark"], False)
    return A


if __name__ == "__main__":
    A = assembly()
    tot = 0
    for k, (m, g, col, pr) in A.items():
        bb = m.bounding_box()
        print("%-14s %s %s vol %6.1f cm3  x %6.1f..%6.1f y %6.1f..%6.1f z %6.1f..%6.1f  덩어리 %d" % (
            k, g, "P" if pr else "-", m.volume() / 1000, bb[0], bb[3], bb[1], bb[4], bb[2], bb[5], len(m.decompose())))
        tot += m.volume() / 1000 if pr else 0
    print("출력 부품 부피 합계 %.0f cm3" % tot)
