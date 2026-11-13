"""겹침 검사: 가만히 있을 때의 부품 쌍, 받침 회전·끄덕임·판 회전 전 범위에서 무리끼리 겹치는지와 가장 좁은 틈."""
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
from cadlib import *
from p2 import *
import parts as v1

import paths
OUT = paths.RES + '/'
# 일부러 닿거나 맞물리는 쌍 (겹침 허용 부피 mm3)
ALLOW = {("cover", "pinion1"): 60, ("sector", "pinion2"): 60, ("yoke", "motor1"): 5, ("yoke", "motor2"): 5,
         ("pinion1", "motor1"): 200, ("pinion2", "motor2"): 200, ("rotor_bar", "motor3"): 200, ("crown", "motor3"): 5}


def allowed(a, b):
    return ALLOW.get((a, b), ALLOW.get((b, a), 0.3))


def static_pairs(A):
    bad = []
    names = [k for k in A if k != "antenna"]
    for a, b in itertools.combinations(names, 2):
        if not A[a][3] and not A[b][3]:
            continue                                   # 구매 부품끼리는 따지지 않는다
        v = (A[a][0] ^ A[b][0]).volume()
        if v > allowed(a, b):
            bad.append((a, b, round(v, 1)))
    return bad


def grp(A, g, exclude=()):
    return union([m for k, (m, gg, c, pr) in A.items() if gg == g and k not in exclude and k != "antenna"])


if __name__ == "__main__":
    res = {}
    for size, length in (("S", 150.0), ("L", 200.0)):
        A = P.assembly(0, 0, 0, size=size, length=length)
        bad = static_pairs(A)
        print("== 가만히 있을 때 겹침 (받침 %s, 판 %d): %s" % (size, length, "없음" if not bad else bad))
        res["static_%s" % size] = bad

    # 머리 좌표에서 한 번만 만든다
    mb, mc = v1.motor_body_dummy()
    b, cc = v1.pi_dummy(); db, dc = v1.driver_dummy()
    HEADP = union([P.head_shell(), P.head_lid(), P.crown(), v1._m1(mb), v1._m1(mc), b, cc, P.usb_plug_dummy(), P.lan_plug_dummy(), P.dongle_dummy()] + list(P.button_dummy())
                  + [db.transform(P.driver_T(k)) for k in range(3)] + [P.driver_keepout(k) for k in range(3)] + [P.gpio_keepout()])
    SEC = P.sector_head()
    YOKE = union([P.yoke(), P.pin(1), P.pin(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()),
                  mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), P.pinion1(), P.m2_ring()])
    P2 = P.pinion2()
    worst = {}; bad = []

    def note(name, a, b, tag, tol=1.0, search=40.0):
        v = (a ^ b).volume()
        gap = a.min_gap(b, search) if v <= tol else 0.0
        w = worst.get(name, (1e9, ""))
        if gap < w[0]:
            worst[name] = (gap, tag)
        if v > tol:
            bad.append((name, tag, round(v, 1)))

    for size, length in (("S", 150.0), ("L", 200.0)):
        BASE = union([P.tub(size), P.cover(size), P.battery_dummy(size), P.usb_a_dummy(size)] + ([P.lan_cap()] if size == "S" else []) + [P.clip().rotate([0, 0, a]) for a in (0.0, 90.0, 180.0, 270.0)])   # 고리는 네 방향 다
        ROT = union([P.rotor_bar(), P.blade(length), P.foil(length), P.trim_plate()])
        # 1) 판 회전부 vs 머리 (끄덕임과 상관없다)
        for spin in np.arange(0, 360, 5.0):
            note("회전부-머리 %s" % size, ROT.rotate([0, 0, spin]), HEADP + SEC, "spin %d" % spin)
        # 2) 끄덕임 전 범위
        for pitch in list(np.arange(PITCH_MIN, PITCH_MAX, 5.0)) + [PITCH_MAX, TILT3]:
            H = P.pose_head(HEADP, 0, pitch); S = P.pose_head(SEC, 0, pitch)
            if size == "S":
                note("머리-요크", H, YOKE, "pitch %.0f" % pitch, tol=3.0)      # 핀이 꽂힌 자리는 3mm3 까지 봐준다
                note("부채꼴-요크", S, YOKE, "pitch %.0f" % pitch)
                note("부채꼴-작은기어", S, P2, "pitch %.0f" % pitch, tol=80.0)
            note("머리-받침 %s" % size, H + S, BASE, "pitch %.0f" % pitch)
            for spin in np.arange(0, 360, 15.0):
                R = P.pose_head(ROT, 0, pitch, spin)
                note("회전부-요크 %s" % size, R, YOKE + P2, "pitch %.0f spin %d" % (pitch, spin))
                note("회전부-받침 %s" % size, R, BASE, "pitch %.0f spin %d" % (pitch, spin))
        # 3) 받침 회전 한 바퀴: 요크 vs 받침 (작은 기어 1은 맞물리는 자리라 뺀다)
        YK = union([P.yoke(), mb.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T())])
        for yaw in np.arange(0, 360, 10.0):
            note("요크-받침 %s" % size, YK.rotate([0, 0, yaw]), BASE, "yaw %d" % yaw)
    print("== 움직일 때 겹침 %d건" % len(bad)); [print("   ", x) for x in bad[:40]]
    print("== 가장 좁은 틈 (mm)")
    for k, (g, tag) in sorted(worst.items()):
        print("   %-18s %5.1f   (%s)" % (k, g, tag))
    res["moving_bad"] = bad; res["min_gap"] = {k: [round(g, 2), tag] for k, (g, tag) in worst.items()}
    json.dump(res, open(OUT + "check.json", "w"), ensure_ascii=False, indent=1)
