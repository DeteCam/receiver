"""스탠드 받이(23)와 받침 통 밑면 구멍의 검사: 맞춤, 나사, 너트 자리, 스탠드 위에서의 넘어짐.   python3 stand_check.py   결과: results/stand_check.json

시험장에서는 수신기를 2.4m 높이에 올린다 (앉은 사람 머리 위로 보아야 한다: sim/rt). 받침 통 밑에 스탠드 받이를 M3 나사 4개로 붙이고,
받이 가운데의 1/4인치 너트를 스탠드(조명 스탠드, 삼각대, 압축봉 클램프)의 1/4인치 나사에 조인다.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]
import numpy as np
import parts2 as P
from cadlib import *
from p2 import *
import paths

G = 9.81
out = {}; bad = []


def vol(m):
    return float(m.volume())


# ---------------------------------------------------------------- 1. 맞춤 (겹침, 구멍)
mount = P.stand_mount(); nut = P.stand_nut()
for size in ("S", "L"):
    tub = P.tub(size)
    ov = vol(tub ^ mount)                                           # 받이 윗면 z=0 에 받침 통 바닥 z=0 이 닿는다
    out["겹침 mm3 (받침 통 %s ↔ 받이)" % size] = round(ov, 3)
    if ov > 0.5: bad.append("받침 통 %s 와 받이가 겹친다" % size)
    # 밑면 구멍이 기둥 안에 있고, 위쪽 구멍과 이어지지 않는가
    for x, y in POSTS:
        probe = cz(0.6, STAND_POST_HOLE[1] + 0.3, 13.7, x, y)       # 두 구멍 사이의 살
        if vol(tub ^ probe) < 0.95 * vol(probe): bad.append("받침 통 %s: 밑면 구멍과 위쪽 구멍 사이의 살이 모자란다" % size)
        wall = cz(STAND_POST_HOLE[0] + 2.0, 2.5, STAND_POST_HOLE[1], x, y) - cz(STAND_POST_HOLE[0] + 0.05, 2.0, STAND_POST_HOLE[1] + 1, x, y)
        if vol(tub ^ wall) < 0.97 * vol(wall): bad.append("받침 통 %s: 밑면 구멍 둘레 살(2mm)이 모자란다" % size)
ov_nut = vol(mount ^ nut)
out["겹침 mm3 (받이 ↔ 너트)"] = round(ov_nut, 3)
if ov_nut > 0.5: bad.append("너트가 받이에 겹친다")
out["너트 자리"] = dict(맞변=STAND_NUT_AF + STAND_NUT_FIT, 깊이=round(STAND_T - STAND_FLOOR, 2), 너트_두께=STAND_NUT_T, 너트_위_빈틈=round(STAND_T - STAND_FLOOR - STAND_NUT_T, 2), 바닥_두께=STAND_FLOOR)
out["스탠드 나사 길이 (받이 밑면에서)"] = dict(가장_짧게=round(STAND_FLOOR + 3.0, 1), 가장_길게=round(STAND_T, 1), 설명="너트에 3mm 넘게 물리고, 받침 통 바닥에 닿지 않는 범위 (mm)")

# ---------------------------------------------------------------- 2. 나사 4개
under_head = STAND_PLATE + 2.0                                      # 나사 머리 밑의 살 (윗판 3 + 덧살 2)
screws = {}
for L in (10.0, 12.0):
    eng = min(L - under_head, STAND_POST_HOLE[1])                   # 기둥에 물리는 길이
    shear_area = np.pi * 3.0 * eng * 0.75                           # M3 나사산이 무는 원통 면 (75% 가 문다고 본다)
    pull = shear_area * 20.0                                        # PLA 층 사이 전단 20 MPa (보수적)
    screws["M3 x %d" % L] = dict(물리는_길이_mm=round(eng, 1), 나사_하나의_뽑힘_힘_N=round(pull))
out["받이 고정 나사"] = screws
m_dev = 1.027; m_mount = vol(mount) * 1.24e-3 * 1e-3 * 1000 / 1000  # kg (채움 100%, PLA 1.24 g/cm3)
m_mount = vol(mount) * 1.24e-6
out["무게 kg"] = dict(수신기_S_배터리_포함=m_dev, 받이=round(m_mount, 3), 합=round(m_dev + m_mount, 3))
load = (m_dev + m_mount) * G
out["나사 여유 (수신기를 거꾸로 들어도)"] = round(4 * screws["M3 x 10"]["나사_하나의_뽑힘_힘_N"] / load, 1)

# ---------------------------------------------------------------- 3. 너트 자리의 힘 (손으로 1 N·m 조였을 때)
T_tight = 1.0; d = 6.35e-3; F = T_tight / (0.2 * d)
area = 0.866 * STAND_NUT_AF ** 2 - np.pi * (STAND_BOLT_D / 2) ** 2
hold = 0.3 * F * 6.0e-3                                             # 스탠드 어깨(반지름 6mm 쯤)와의 마찰로 버티는 비틀림
twist = 0.1235                                                      # 받침 모터가 낼 수 있는 가장 큰 비틀림 (tip_mujoco.json)
out["너트 자리"].update(조임력_N=round(F), 눌림_MPa=round(F / area, 1), 풀림을_버티는_비틀림_Nm=round(hold, 2), 받침_모터의_비틀림_Nm=twist, 여유=round(hold / twist, 1))
if F / area > 30: bad.append("너트 밑 눌림이 크다")

# ---------------------------------------------------------------- 4. 스탠드째 넘어짐
H = 2.40; com_off = 0.0192                                          # 안테나 높이, 수신기 무게중심이 축에서 벗어나는 가장 큰 값 (tipover.json)
stands = {"가벼운 조명 스탠드 (1.3kg, 다리 반지름 0.45m)": (1.3, 0.45, 0.9), "튼튼한 조명 스탠드 (2.5kg, 다리 반지름 0.55m)": (2.5, 0.55, 0.9)}
tip = {}
for name, (ms, rleg, hs) in stands.items():
    for bag in (0.0, 3.0, 5.0):
        r_in = rleg * 0.5                                           # 다리 셋이 만드는 세모의 안쪽 원 반지름 (다리 사이 방향이 가장 약하다)
        M = m_dev + m_mount + ms + bag
        h = ((m_dev + m_mount) * (H - 0.10) + ms * hs + bag * 0.25) / M
        push = (M * G * r_in - (m_dev + m_mount) * G * com_off) / H
        tip["%s%s" % (name, " + 모래주머니 %dkg" % bag if bag else "")] = dict(꼭대기를_미는_힘_N=round(push, 1), 넘어가는_기울기_도=round(float(np.degrees(np.arctan(r_in / h))), 1))
out["스탠드째 넘어짐 (수신기 2.4m)"] = tip
out["스스로 넘어지나"] = dict(무게중심이_벗어나는_몫="%.1f%%" % (100 * (m_dev + m_mount) * com_off / ((m_dev + m_mount + 1.3) * 0.225)), 설명="머리를 가장 깊이 숙여도 가벼운 스탠드의 버티는 힘의 이만큼만 쓴다. 스스로는 넘어지지 않는다")
out["압축봉 (바닥~천장)"] = "넘어지지 않는다. 클램프의 1/4인치 나사에 같은 받이로 조인다"
out["problems"] = bad
json.dump(out, open(os.path.join(paths.RES, "stand_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(k, ":", json.dumps(v, ensure_ascii=False))
print("문제 %d개" % len(bad))
