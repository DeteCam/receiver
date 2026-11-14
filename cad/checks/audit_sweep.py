"""걸림 검사 (촘촘하게): 판이 도는 동안 몸체에 닿는 곳이 없는지를 check2.py 보다 잘게 나눠 다시 본다.
    python3 audit_sweep.py        ->  결과 폴더의 audit_sweep.json

check2.py 는 끄덕임 5도, 판 15도 간격으로 본다. 틈이 가장 좁은 곳은 그 사이에 있을 수 있어서 여기서는
  1) 받침 S·L 모두 (판 100 x 150, 판 틀(한 장)과 누름쇠, 나사 머리와 너트까지),
  2) 끄덕임을 멈춤턱(-5.3도, 86.9도)까지 1도 간격, 판을 3도 간격으로,
  3) 모터 축의 흔들림까지 넣어서 (판 모터 축이 1도 기울었다고 보고 네 방향으로) 본다.
머리 속 부품(동글 2개, 랜선 플러그 포함)도 머리에 넣는다.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *

STOP_LO, STOP_HI = -5.3, 86.9
mb, mc = v1.motor_body_dummy(); b, cc = v1.pi_dummy(); db, dc = v1.driver_dummy()
YOKE = union([P.yoke(), P.pin(1), P.pin(-1), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), P.pinion1(), P.pinion2(), P.m2_ring()])
ROTS = {150.0: P.rotor_set(150.0, low_cap=True)}      # 알루미늄 판, 판 틀, 누름쇠, 나사·너트, 마개(가장 내려온 경우)까지
HEADP = union([P.head_shell(), P.head_lid(), P.crown(), P.marker_frame(), P.head_hw(), v1._m1(mb), v1._m1(mc), P.sector_head()])
HUB = np.array([0.0, 0.0, M1_ZT + 5.0])                       # 축이 흔들릴 때의 중심 (모터 앞 베어링 근처)


def wobble(m, ax, deg):
    """판 모터 축이 deg 만큼 기울었을 때의 회전부 (머리 좌표, 판 회전 0도 기준)"""
    r = [deg, 0, 0] if ax == 0 else [0, deg, 0]
    return m.translate(list(-HUB)).rotate(r).translate(list(HUB))


def base(size):
    return union([P.tub(size), P.cover(size), P.battery_dummy(size), P.usb_a_dummy(size)] + [P.lan_cap(size)] + [P.clip().rotate([0, 0, a]) for a in (0.0, 90.0, 180.0, 270.0)])


res = {}; t0 = time.time()
pitches = [STOP_LO] + list(np.arange(-5.0, 86.01, 1.0)) + [TILT3, 86.5, STOP_HI]
spins = np.arange(0.0, 360.0, 3.0)
BASES = {s: base(s) for s in ("S", "L")}
for length, ROT in ROTS.items():
    # 판 - 머리 (끄덕임과 상관없다): 1도 간격
    g = min((ROT.rotate([0, 0, float(s)]).min_gap(HEADP, 30.0), float(s)) for s in np.arange(0.0, 360.0, 1.0))
    res["판 %d · 머리" % length] = dict(gap=round(g[0], 2), at="판 %d도" % g[1])
    print("판 %d - 머리: 가장 좁은 틈 %.2f mm (판 %d도)  [%.0f초]" % (length, g[0], g[1], time.time() - t0), flush=True)
    for size in ("S", "L"):
        FIX = YOKE + BASES[size]
        worst = (1e9, None); hit = []; by_pitch = {}
        for p in pitches:
            gp = 1e9
            for s in spins:
                R = P.pose_head(ROT, 0, float(p), float(s))
                v = (R ^ FIX).volume()
                gg = 0.0 if v > 0.5 else R.min_gap(FIX, 40.0)
                if v > 0.5:
                    hit.append((round(float(p), 1), float(s), round(v, 1)))
                gp = min(gp, gg)
                if gg < worst[0]:
                    worst = (gg, (float(p), float(s)))
            by_pitch[round(float(p), 1)] = round(float(gp), 2)
        key = "받침 %s · 판 %d" % (size, length)
        res[key] = dict(gap=round(worst[0], 2), at="끄덕임 %.1f도, 판 %d도" % worst[1], hits=hit[:10], n_hit=len(hit),
                        gap_soft=min(v for k, v in by_pitch.items() if k <= 85.0), gap_tilt=by_pitch[round(TILT3, 1)], by_pitch=by_pitch)
        print("%s: 겹침 %d건, 가장 좁은 틈 %.2f mm (%s). 85도까지 %.2f, 재는 자세 %.2f  [%.0f초]" % (key, len(hit), worst[0], res[key]["at"], res[key]["gap_soft"], by_pitch[round(TILT3, 1)], time.time() - t0), flush=True)
        # 축 흔들림 1도: 가장 좁았던 자세 둘레와 재는 자세에서
        wp, ws = worst[1]; wob = 1e9; wob_t = 1e9
        for ax in (0, 1):
            for d in (-1.0, 1.0):
                RW = wobble(ROT, ax, d)
                for p in (wp, 85.0):
                    for s in np.arange(ws - 12, ws + 12.1, 3.0):
                        R = P.pose_head(RW, 0, float(p), float(s))
                        g_ = 0.0 if (R ^ FIX).volume() > 0.5 else R.min_gap(FIX, 40.0)
                        if p > 85.5:
                            wob = min(wob, g_)
                        else:
                            wob_t = min(wob_t, g_)
                gh = min(RW.rotate([0, 0, float(s)]).min_gap(HEADP, 30.0) for s in np.arange(0.0, 360.0, 5.0))
                res.setdefault("판 %d · 머리 (축 1도 흔들림)" % length, dict(gap=1e9))
                res["판 %d · 머리 (축 1도 흔들림)" % length]["gap"] = round(min(res["판 %d · 머리 (축 1도 흔들림)" % length]["gap"], gh), 2)
        res[key]["gap_wobble_stop"] = round(float(min(wob, 1e8)), 2); res[key]["gap_wobble_85"] = round(float(wob_t), 2)
        print("    축이 1도 흔들릴 때: 멈춤턱 자세 %.2f mm, 85도 %.2f mm" % (wob, wob_t), flush=True)
for length in ROTS:
    k = "판 %d · 머리 (축 1도 흔들림)" % length
    print("%s: %.2f mm" % (k, res[k]["gap"]))
# 머리 - 요크 - 받침 (판 없이): 1도 간격, 머리 속 부품까지
HEADALL = union([HEADP, b, cc, P.usb_plug_dummy(), P.lan_plug_dummy(), P.dongle_dummy(), P.dongle_dummy(DONGLE2)] + list(P.button_dummy()))
lug = P.extrude_z(P.poly([(-HX + 0.1, Y_T - 3.0), (-HX - 4.5, Y_T - 3.0), (-HX - 4.5, Y_T - 1.5), (-HX + 0.1, Y_T + 3.1)]), S_T - LUG_R - 2.5, S_T - LUG_R + 2.5)
YK = union([P.yoke(), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T())])
worst = (1e9, None)
for p in np.arange(-5.0, 86.01, 1.0):
    H = P.pose_head(HEADALL - lug, 0, float(p))
    v = (H ^ YK).volume(); g_ = 0.0 if v > 3.0 else H.min_gap(YK, 30.0)
    if g_ < worst[0]:
        worst = (g_, float(p))
res["머리 · 요크"] = dict(gap=round(worst[0], 2), at="끄덕임 %.0f도" % worst[1])
print("머리(속 부품 포함) - 요크: 가장 좁은 틈 %.2f mm (끄덕임 %.0f도)  [%.0f초]" % (worst[0], worst[1], time.time() - t0))
json.dump(res, open(os.path.join(paths.RES, "audit_sweep.json"), "w"), ensure_ascii=False, indent=1)
