"""나사 머리까지 넣은 추가 겹침 검사: 판 150·200 (볼트·너트·마개·나사 머리 포함), 모터 자리·뚜껑·마커 틀·핀 막이 나사 머리.
    python3 check_extra.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *

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
    "판 150 (1T, 동전 8개)": P.rotor_set(150.0),
    "판 150 (마개가 가장 내려온 경우)": P.rotor_set(150.0, low_cap=True),
    "판 200 (1T, 동전 11개)": P.rotor_set(200.0),
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
json.dump(res, open(os.path.join(paths.RES, "check_extra.json"), "w"), ensure_ascii=False, indent=1)
