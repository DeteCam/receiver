"""나사 머리까지 넣은 추가 겹침 검사: 판 150 (볼트·너트·마개·나사 머리 포함), 모터 자리·뚜껑·마커 틀·핀 막이 나사 머리,
그리고 크기가 조금 다른 부품(버튼 모듈, 드라이버 보드 구멍 간격)이 들어가는지.
    python3 check_extra.py              전부 (15분쯤)
    python3 check_extra.py --variants   크기가 다른 부품만 다시 보고 결과 파일의 그 칸만 바꾼다"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *

def variants():
    """크기가 조금 다른 부품도 들어가는가 (만든 곳마다 다른 것: 버튼 모듈, 드라이버 보드 구멍 간격)"""
    var = {}
    zt = BTN_Z + BTN_TOP
    pcb_max = box(BTN_X - BTN_W / 2, BTN_X - BTN_W / 2 + BTN_W_MAX, BTN_PCB_Y, BTN_PCB_Y + 1.6, zt - BTN_L_MAX, zt)       # 가장 큰 버튼 기판: 왼쪽 위 귀를 같은 자리에 댄다
    var["버튼 기판 %.1f x %.1f 와 머리 몸통·뚜껑의 겹침 (mm3)" % (BTN_W_MAX, BTN_L_MAX)] = round((pcb_max ^ union([P.head_shell(), P.head_lid()])).volume(), 3)
    var["버튼이 뚜껑 구멍 가운데에서 비켜도 되는 거리 (mm)"] = round(BTN_HOLE_R - 1.75, 2)
    shank = {(a_, c_): cz(1.0 - 0.01, -1, 3, a_ * DRV_HOLE[0] / 2, c_ * DRV_HOLE[1] / 2) for a_ in (-1, 1) for c_ in (-1, 1)}       # M2 나사 (기둥 자리)
    for sx_, sz_ in ((29.0, 26.0), (30.0, 27.0), (30.0, 26.5), (28.5, 26.5), (29.5, 27.5), (28.5, 25.5), (30.5, 27.5), (28.5, 27.5), (31.5, 24.5)):     # 0.5mm, 한 방향 1mm, 두 방향 1mm, 2mm
        def board(dx=0.0, dy=0.0):
            """35 x 32 보드, 구멍 지름 3. 보드 가운데를 (dx, dy) 만큼 옮겨 놓는다"""
            return box(-17.5 + dx, 17.5 + dx, -16 + dy, 16 + dy, 0, 1.6) - union([cz(1.5, -1, 3, a_ * sx_ / 2 + dx, c_ * sz_ / 2 + dy) for a_ in (-1, 1) for c_ in (-1, 1)])
        four = (board() ^ union(list(shank.values()))).volume()
        dy2 = -(DRV_HOLE[1] - sz_) / 2                   # 아래 줄 두 구멍을 기둥에 맞춘다 (보드를 그만큼 옮긴다)
        two_x = (board(0.0, dy2) ^ union([shank[(-1, -1)], shank[(1, -1)]])).volume()
        dx2 = -(DRV_HOLE[0] - sx_) / 2                   # 왼쪽 줄 두 구멍을 기둥에 맞춘다
        two_z = (board(dx2, 0.0) ^ union([shank[(-1, -1)], shank[(-1, 1)]])).volume()
        var["드라이버 보드 구멍 간격 %.1f x %.1f: M2 나사가 보드에 걸리는 부피 (mm3), 네 개 / 아래 두 개 / 왼쪽 두 개" % (sx_, sz_)] = [round(four, 3), round(two_x, 3), round(two_z, 3)]
    for k, v in var.items():
        print("== %s: %s" % (k, v))
    return var


if "--variants" in sys.argv:                            # 크기가 다른 부품만: 결과 파일의 그 칸만 바꾼다
    out = os.path.join(paths.RES, "check_extra.json")
    res = json.load(open(out, encoding="utf-8")); res["크기가 다른 부품"] = variants()
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1); sys.exit(0)


mb, mc = v1.motor_body_dummy(); b, cc = v1.pi_dummy(); db, dc = v1.driver_dummy()
HEAD_SCREWS = union([cz(3.0, HZ1 + 2.0, HZ1 + 4.6, sx, sy) for sx, sy in CROWN_SCREWS]                       # 모터 자리 나사 머리 (둥근머리 2.6mm)
                    + [cy(3.0, HY0 - 2.6, HY0, sx, sz) for sx, sz in LID_SCREWS]                            # 뚜껑 나사 머리
                    + [cz(3.0, M1_ZT, M1_ZT + 2.6, -M_OFFSET, s * M_EAR) for s in (1, -1)]                  # 모터 3 귀 나사 머리
                    + [P.head_hw()])                                                                        # 마커 틀 나사 머리
HEADP = union([P.head_shell(), P.head_lid(), P.crown(), P.marker_frame(), v1._m1(mb), v1._m1(mc), HEAD_SCREWS, P.sector_head()])
px, pz = P.pin2_xz()
YOKE_SCREWS = union([cy(3.0, ARM_Y1 + 0.8, ARM_Y1 + 3.4, ex, ez) for ex, ez in P.m2_ears()]                 # 모터 2 귀 나사 머리
                    + [cz(3.0, DECK_Z1 + M1_PAD, DECK_Z1 + M1_PAD + 2.6, ex, ey) for ex, ey in P.m1_ears()]  # 모터 1 귀 나사 머리
                    + [cz(3.0, DECK_Z1, DECK_Z1 + 2.6, *YAW_STOP_XY), cz(1.5, DECK_Z1 - 10.0, DECK_Z1, *YAW_STOP_XY)]   # 받침 회전 멈춤 나사 (아래로 6mm 나온다)
                    + [cy(1.5, -ARM_Y1, -ARM_Y1 + 10.0, x, z) for x, z in P.stop_screws_xz()]              # 끄덕임 멈춤 나사 (안쪽으로 4mm 나온다)
                    + [cy(3.0, -ARM_Y1 - 2.6, -ARM_Y1, x, z) for x, z in P.stop_screws_xz()])
YOKE = union([P.yoke(), P.pin(1), P.pin(-1), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), P.pinion1(), P.pinion2(), P.m2_ring(), YOKE_SCREWS])
SETS = {
    "판 150 (0.5T, 동전 %d개)" % P.COINS[(150.0, 0.5)]: P.rotor_set(150.0),
    "판 150 (마개가 가장 내려온 경우)": P.rotor_set(150.0, low_cap=True),
    "판 150 (0.3T, 동전 %d개)" % P.COINS[(150.0, 0.3)]: P.rotor_set(150.0, t=0.3),
}
res = {}
for name, ROT in SETS.items():
    worst = {}; bad = []
    def note(tag, a, b_, what, tol=1.0):
        v = (a ^ b_).volume()
        g = a.min_gap(b_, 40.0) if v <= tol else 0.0
        if g < worst.get(tag, (1e9, ""))[0]:
            worst[tag] = (g, what)
        if v > tol:
            bad.append((tag, what, round(v, 1)))
    for spin in np.arange(0, 360, 5.0):
        note("회전부-머리", ROT.rotate([0, 0, spin]), HEADP, "spin %d" % spin)
    for size in ("S", "L"):
        BASE = union([P.tub(size), P.cover(size), P.battery_dummy(size), P.usb_a_dummy(size)] + [P.lan_cap(size)] + [P.clip().rotate([0, 0, a]) for a in (0.0, 90.0, 180.0, 270.0)])
        for pitch in list(np.arange(PITCH_MIN, PITCH_MAX, 5.0)) + [PITCH_MAX, TILT3]:
            for spin in np.arange(0, 360, 10.0):
                R = P.pose_head(ROT, 0, pitch, spin)
                if size == "S":
                    note("회전부-요크", R, YOKE, "pitch %.0f spin %d" % (pitch, spin))
                note("회전부-받침 %s" % size, R, BASE, "pitch %.0f spin %d" % (pitch, spin))
    print("== %s: 겹침 %d건 %s" % (name, len(bad), bad[:6]))
    for k, (g, what) in sorted(worst.items()):
        print("   %-12s 가장 좁은 틈 %5.1f mm (%s)" % (k, g, what))
    res[name] = dict(bad=bad, min_gap={k: [round(g, 2), w] for k, (g, w) in worst.items()})
# 나사 머리를 단 머리와 요크: 끄덕임 전 범위
worst = (1e9, ""); bad = []
HEAD_ONLY = union([P.head_shell(), P.head_lid(), P.crown(), P.marker_frame(), v1._m1(mb), v1._m1(mc), HEAD_SCREWS, b, cc, P.lan_plug_dummy(), P.usb_plug_dummy(), P.dongle_dummy(), P.dongle_dummy(DONGLE2)] + list(P.button_dummy()))
YOKE_NOPIN = union([P.yoke(), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), YOKE_SCREWS])
lug = P.extrude_z(P.poly([(-HX + 0.1, Y_T - 3.0), (-HX - 4.5, Y_T - 3.0), (-HX - 4.5, Y_T - 1.5), (-HX + 0.1, Y_T + 3.1)]), S_T - LUG_R - 2.5, S_T - LUG_R + 2.5)
for pitch in list(np.arange(PITCH_MIN + 1.0, PITCH_MAX - 0.5, 4.0)) + [TILT3]:
    H = P.pose_head(HEAD_ONLY - lug, 0, pitch)          # 멈춤 돌기는 멈춤 나사에 닿는 것이 일이라 뺀다
    v = (H ^ YOKE_NOPIN).volume(); g = H.min_gap(YOKE_NOPIN, 30.0) if v < 3.0 else 0.0
    if g < worst[0]:
        worst = (g, "pitch %.0f" % pitch)
    if v >= 3.0:
        bad.append(("pitch %.0f" % pitch, round(v, 1)))
print("== 나사 머리까지 넣은 머리-요크: 겹침 %d건 %s, 가장 좁은 틈 %.1f mm (%s)" % (len(bad), bad[:5], worst[0], worst[1]))
res["머리-요크 (나사 포함)"] = dict(bad=bad, min_gap=[round(worst[0], 2), worst[1]])
res["크기가 다른 부품"] = variants()
json.dump(res, open(os.path.join(paths.RES, "check_extra.json"), "w"), ensure_ascii=False, indent=1)
