"""넘어짐 검사 (가만히 있을 때): 관절이 움직이는 전 범위에서 무게중심이 받침 안에 있는지, 얼마나 기울이거나 밀면 넘어가는지.
    python3 tipover.py        ->  결과 폴더의 tipover.json

보는 것
  - 받침 회전(-174~174도) x 끄덕임(-5~86도) x 판 회전(0~360도) 전부에서 무게중심이 받침 바닥 원 안쪽에 얼마나 들어와 있는가 (여유 mm)
  - 그 자세에서 바닥을 몇 도 기울이면 넘어가는가, 꼭대기를 얼마의 힘으로 밀면 넘어가는가
  - 받침 S·L, 배터리가 있을 때·없을 때, 알루미늄 판 1T (판 100 x 150, 판 틀과 누름쇠 포함)
  - 모터가 스스로 낼 수 있는 힘으로 자기를 넘어뜨릴 수 있는가 (판이나 머리가 벽에 걸렸을 때)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import tipmodel as TM
import paths
from p2 import *

T_SPIN, T_YAW, T_NOD = 34.3, 34.3 * (ZW / ZP) * 0.9, 34.3 * (SEC_T / ZP) * 0.9        # 모터가 낼 수 있는 가장 큰 힘 (mN·m): 판, 받침 회전, 끄덕임 (정격 34.3 = 350 gf·cm)
G = 9.81e-3
CASES = [("S", True, AL_T, 150.0), ("S", False, AL_T, 150.0), ("L", True, AL_T, 150.0), ("L", False, AL_T, 150.0)]


def scan(d):
    worst = None; best = None; rows = []
    for pitch in list(np.arange(PITCH_MIN, PITCH_MAX + 0.01, 3.5)) + [TILT3, 85.0]:
        for yaw in np.arange(-174.0, 174.01, 29.0):
            for spin in np.arange(0.0, 360.0, 30.0):
                s = d.stability(float(yaw), float(pitch), float(spin)); s["pose"] = (float(yaw), float(pitch), float(spin))
                if worst is None or s["margin"] < worst["margin"]:
                    worst = s
                if best is None or s["margin"] > best["margin"]:
                    best = s
        rows.append(None)
    return worst, best


def self_push(d, s):
    """모터가 스스로 낼 수 있는 힘이 장치를 넘어뜨리는 힘의 몇 %인가.
    판이 벽에 걸림: 판 모터가 판 틀 끝(반지름 96)을 미는 힘. 머리가 벽에 걸림: 끄덕임 모터가 판 틀 위끝을 미는 힘. 받침 회전 중 걸림: 판 끝(받침 축에서 가장 먼 곳)."""
    m = s["mass_g"] * G; restoring = m * s["margin"]                         # 버티는 모멘트 (N·mm)
    r_plate = FR_ENV
    f_spin = T_SPIN / r_plate                                                # N (mN·m / mm)
    lever_nod = (BL_Z0 + d.length + FR_END_TOP) - S_T                        # 끄덕임 축에서 판 틀 위끝까지
    f_nod = T_NOD / lever_nod
    f_yaw = T_YAW / (r_plate + 20.0)
    h = s["top_h"]
    return dict(spin=dict(force_N=round(f_spin, 2), pct=round(f_spin * h / restoring * 100, 0)), nod=dict(force_N=round(f_nod, 2), pct=round(f_nod * h / restoring * 100, 0)),
                yaw=dict(force_N=round(f_yaw, 2), pct=round(f_yaw * h / restoring * 100, 0)))


def reach(length):
    """판과 균형추가 닿는 가장 먼 곳 (받침 회전축에서의 수평 거리 mm)과 그때의 끄덕임 각. 받침이 도니까 모든 방향으로 이만큼 닿는다."""
    best = (0.0, 0.0)
    for pitch in np.arange(PITCH_MIN, PITCH_MAX + 0.01, 1.0):
        for spin in np.arange(0.0, 360.0, 3.0):
            pts = TM.frame_corners(length) + [[POD_XC - 14.0, y, 12.0] for y in (-19.0, 19.0)]
            r = max(float(np.hypot(*TM.head_to_world(q, 0.0, float(pitch), float(spin))[:2])) for q in pts)
            if r > best[0]:
                best = (r, float(pitch))
    tilt = max(float(np.hypot(*TM.head_to_world(q, 0.0, TILT3, float(sp))[:2])) for q in TM.frame_corners(length) for sp in np.arange(0.0, 360.0, 3.0))
    return dict(max_mm=round(best[0]), at_pitch=best[1], measure_mm=round(tilt), clear_cm=int(np.ceil((best[0] + 30.0) / 10.0)))


res = {"reach": {"%d" % L: reach(L) for L in (150.0,)}}
for L, r in res["reach"].items():
    print("판 %s: 움직이는 부분이 가운데에서 가장 멀리 %d mm 까지 닿는다 (끄덕임 %.0f도). 재는 자세에서는 %d mm. 둘레를 비울 거리: 가운데에서 %d cm" % (L, r["max_mm"], r["at_pitch"], r["measure_mm"], r["clear_cm"]))
for size, batt, t, length in CASES:
    d = TM.Device(size, batt, t, length); worst, best = scan(d)
    key = "받침 %s, 배터리 %s, 판 100×%d×%gT" % (size, "있음(%dg)" % TM.BATT_G[size] if batt else "없음", length, t)
    tilt = d.stability(0.0, TILT3, 0.0)
    res[key] = dict(size=size, battery=batt, plate_t=t, length=length, total_g=round(worst["mass_g"]), masses={k: round(v, 1) for k, v in d.mass().items()}, support_R=d.R,
                    worst=dict(pose=worst["pose"], off=round(worst["off"], 1), margin=round(worst["margin"], 1), com_h=round(float(worst["com"][2]), 1), tip_deg=round(worst["tip_deg"], 1),
                               push_top_N=round(worst["push_top_N"], 2), top_h=round(worst["top_h"])),
                    measure=dict(off=round(tilt["off"], 1), margin=round(tilt["margin"], 1), tip_deg=round(tilt["tip_deg"], 1), push_top_N=round(tilt["push_top_N"], 2), top_h=round(tilt["top_h"])),
                    best_margin=round(best["margin"], 1), self_push=self_push(d, worst))
    w = res[key]["worst"]; sp = res[key]["self_push"]
    print("%s: 전체 %d g. 가장 나쁜 자세(받침 %.0f, 끄덕임 %.0f, 판 %.0f도)에서 무게중심이 축에서 %.1f mm, 받침 끝까지 %.1f mm (받침 반지름 %.1f)" % (
        key, w["total_g"] if "total_g" in w else res[key]["total_g"], w["pose"][0], w["pose"][1], w["pose"][2], w["off"], w["margin"], d.R))
    print("    바닥을 %.1f도 기울이면 넘어감, 꼭대기(높이 %d mm)를 %.2f N (= %.0f g 의 힘)으로 밀면 넘어감. 재는 자세에서는 %.1f도, %.2f N" % (
        w["tip_deg"], w["top_h"], w["push_top_N"], w["push_top_N"] / 9.81 * 1000, res[key]["measure"]["tip_deg"], res[key]["measure"]["push_top_N"]))
    print("    모터가 스스로 미는 힘이 넘어뜨리는 힘의 몇 %%: 판이 벽에 걸림 %.0f%% (%.2f N), 받침 회전 중 걸림 %.0f%% (%.2f N), 머리를 숙이다 벽에 걸림 %.0f%% (%.2f N)" % (
        sp["spin"]["pct"], sp["spin"]["force_N"], sp["yaw"]["pct"], sp["yaw"]["force_N"], sp["nod"]["pct"], sp["nod"]["force_N"]))
json.dump(res, open(os.path.join(paths.RES, "tipover.json"), "w"), ensure_ascii=False, indent=1)
