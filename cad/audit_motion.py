"""동작 검사: 기어를 실제 비율대로 같이 돌려 이가 파고드는지, 멈춤턱이 맞는 각에서 닿는지, 자세별로 판과 몸체 사이 틈이 얼마인지.
    python3 audit_motion.py        ->  결과 폴더의 audit_motion.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *

res = {}
px, pz = P.pin2_xz()

# ---------------------------------------------------------------- 1. 끄덕임 기어 (부채꼴 112T 상당 + 작은 기어 10T)
sec0 = P.sector_gear()
def sec_at(th):
    return sec0.translate([0, 0, -H_T]).rotate([0, th, 0]).translate([0, 0, H_T])
def pin2_at(ph):
    return P.pinion(ph).rotate([90, 0, 0]).translate([px, ARM_Y0 + 1.8, pz])
ratio2 = SEC_T / float(ZP)
best = None
for sign in (1, -1):
    for ph0 in np.arange(0, 36.0, 0.5):
        v = sum((sec_at(th) ^ pin2_at(ph0 + sign * ratio2 * th)).volume() for th in (0.0, 7.0, 31.0, 54.7, 88.0))
        if best is None or v < best[0]:
            best = (v, sign, ph0)
_, sign2, ph2 = best
free = [ph for ph in np.arange(ph2 - 6, ph2 + 6.01, 0.1) if (sec_at(0.0) ^ pin2_at(ph)).volume() < 0.01]
worst = max(((sec_at(th) ^ pin2_at(ph2 + sign2 * ratio2 * th)).volume(), th) for th in np.arange(PITCH_MIN, PITCH_MAX + 0.01, 1.25))
mid = (min(free) + max(free)) / 2 if free else ph2
worst_mid = max(((sec_at(th) ^ pin2_at(mid + sign2 * ratio2 * th)).volume(), th) for th in np.arange(PITCH_MIN, PITCH_MAX + 0.01, 1.25))
play2 = (max(free) - min(free)) if free else 0.0
res["nod_gear"] = dict(worst_overlap_mm3=round(worst_mid[0], 3), at_pitch=round(worst_mid[1], 1), pinion_free_deg=round(play2, 1), head_play_deg=round(play2 / ratio2, 2))
print("끄덕임 기어: 전 범위를 비율대로 돌렸을 때 이가 파고드는 부피 최대 %.3f mm3 (끄덕임 %.1f도). 작은 기어가 헛도는 각 %.1f도 = 머리 유격 %.2f도" % (
    worst_mid[0], worst_mid[1], play2, play2 / ratio2))

# ---------------------------------------------------------------- 2. 받침 회전 기어 (고정 40T + 작은 기어 10T, 회전판에 타고 돈다)
cov = P.cover("S"); sx, sy = P.m1_shaft()
def pin1_at(yaw, ph):
    return P.pinion(ph).translate([sx, sy, DECK_Z0 - 6.0]).rotate([0, 0, yaw])
ratio1 = ZW / float(ZP)
best = None
for sign in (1, -1):
    for ph0 in np.arange(0, 36.0, 0.5):
        v = sum((cov ^ pin1_at(y, ph0 + sign * ratio1 * y)).volume() for y in (0.0, 4.0, 13.0, 47.0, 101.0))
        if best is None or v < best[0]:
            best = (v, sign, ph0)
_, sign1, ph1 = best
free = [ph for ph in np.arange(ph1 - 6, ph1 + 6.01, 0.1) if (cov ^ pin1_at(0.0, ph)).volume() < 0.01]
mid = (min(free) + max(free)) / 2 if free else ph1
worst1 = max(((cov ^ pin1_at(y, mid + sign1 * ratio1 * y)).volume(), y) for y in np.arange(-174.0, 174.01, 1.5))
play1 = (max(free) - min(free)) if free else 0.0
res["yaw_gear"] = dict(worst_overlap_mm3=round(worst1[0], 3), at_yaw=round(worst1[1], 1), pinion_free_deg=round(play1, 1), yoke_play_deg=round(play1 / ratio1, 2))
print("받침 회전 기어: 이가 파고드는 부피 최대 %.3f mm3 (받침 %.1f도). 작은 기어가 헛도는 각 %.1f도 = 요크 유격 %.2f도" % (worst1[0], worst1[1], play1, play1 / ratio1))

# ---------------------------------------------------------------- 3. 멈춤턱
shell = P.head_shell()
lugbox = box(-HX - 5, -HX - 0.05, Y_T - 4, Y_T + 4, S_T - LUG_R - 3, S_T - LUG_R + 3)
lug = shell ^ lugbox
screws = union([cy(1.5, -ARM_Y1, -ARM_Y1 + 10.0, x, z) for x, z in P.stop_screws_xz()])
def touch(p):
    return (P.pose_head(lug, 0, p) ^ screws).volume()
lo = next((p for p in np.arange(10.0, -12.0, -0.1) if touch(p) > 0.02), None)
hi = next((p for p in np.arange(70.0, 100.0, 0.1) if touch(p) > 0.02), None)
res["nod_stops"] = dict(low=None if lo is None else round(lo, 1), high=None if hi is None else round(hi, 1))
print("끄덕임 멈춤: 돌기가 나사에 닿는 각 = 아래 %s도, 위 %s도 (설계 %.0f, %.0f)" % (res["nod_stops"]["low"], res["nod_stops"]["high"], PITCH_MIN, PITCH_MAX))
ys = cz(1.5, DECK_Z1 - 10.0, DECK_Z1, *YAW_STOP_XY)
def ytouch(y):
    return (ys.rotate([0, 0, y]) ^ cov).volume()
yp = next((y for y in np.arange(150.0, 190.0, 0.2) if ytouch(y) > 0.02), None)
yn = next((y for y in np.arange(-150.0, -190.0, -0.2) if ytouch(y) > 0.02), None)
free_mid = max(ytouch(y) for y in np.arange(-170.0, 170.1, 2.0))
res["yaw_stops"] = dict(pos=None if yp is None else round(yp, 1), neg=None if yn is None else round(yn, 1), rub_between=round(free_mid, 3))
print("받침 회전 멈춤: 나사가 턱에 닿는 각 = %s도, %s도 (설계 ±%.0f). 그 사이에서 덮개에 스치는 부피 %.3f" % (res["yaw_stops"]["neg"], res["yaw_stops"]["pos"], YAW_LIMIT, free_mid))

# ---------------------------------------------------------------- 4. 고정 나사 구멍이 0, 54.7, 90도에서 맞는가
lx, lz = P.lock_xz(); out = []
for p in LOCK_ANGLES:
    s = sec_at(p)
    probe = cy(1.25, SEC_Y0 - 0.5, SEC_Y1 + 0.5, lx, lz)
    out.append(round((probe ^ s).volume(), 3))
res["lock_holes_blocked_mm3"] = out
print("고정 나사: 0 / 54.7 / 90도에서 부채꼴 기어 구멍이 막힌 부피 =", out, "(0 이면 구멍이 딱 맞음)")

# ---------------------------------------------------------------- 5. 자세별 판과 몸체 사이 틈
mb, mc = v1.motor_body_dummy()
ROT = P.rotor_set(150.0, low_cap=True)
FIX = union([P.yoke(), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), P.pinion2(), P.m2_ring(), P.tub("S"), P.cover("S"), P.lan_cap()]
            + [P.clip().rotate([0, 0, a]) for a in (0.0, 90.0, 180.0, 270.0)])       # 고리는 어느 쪽을 보고 끼워질지 모른다: 네 방향을 다 넣는다
gaps = {}
for pitch in (0.0, 20.0, 40.0, TILT3, 60.0, 70.0, 80.0, 85.0, 87.0):
    g = min(P.pose_head(ROT, 0, pitch, spin).min_gap(FIX, 60.0) for spin in np.arange(0, 360, 6.0))
    gaps["%.1f" % pitch] = round(float(g), 1)
res["rotor_gap_by_pitch"] = gaps
print("판이 한 바퀴 돌 때 요크·받침과의 가장 좁은 틈 (끄덕임 각별, mm):", gaps)
json.dump(res, open(os.path.join(paths.RES, "audit_motion.json"), "w"), ensure_ascii=False, indent=1)
