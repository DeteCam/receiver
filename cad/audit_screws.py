"""나사 전수 검사: 나사를 하나씩 모델에 넣어 본다.
    python3 audit_screws.py        ->  결과 폴더의 audit_screws.json
보는 것
  1) 나사 길(지름 2.5 심)에 플라스틱이 막고 있지 않은가  = 구멍이 있고, 깊이가 충분한가
  2) 지나가는 부품의 구멍이 헐거운가 (나사가 걸리지 않고 지나가야 한다)
  3) 나사산이 물리는 길이 (2.5mm 이상이어야 잡는다)
  4) 나사 끝이 어디서 끝나는가, 반대편으로 얼마나 나오는가
  5) 나사 머리가 다른 부품에 닿는가
  6) 드라이버가 들어갈 길(지름 7, 길이 90)이 그 조립 단계에서 비어 있는가
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *


def cyl(p, d, L, r, seg=24):
    """점 p 에서 방향 d 로 길이 L, 반지름 r 인 원기둥"""
    p = np.asarray(p, float); d = np.asarray(d, float); d = d / np.linalg.norm(d)
    x = np.cross([0, 0, 1.0], d)
    x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-9 else x / np.linalg.norm(x)
    y = np.cross(d, x)
    return M.cylinder(L, r, r, seg).transform(np.column_stack([x, y, d, p]))


mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
Hd = lambda m: P.pose_head(m, 0, 0)
Rt = lambda m: P.pose_head(m, 0, 0, 0.0)
PARTS = {   # 이름: (형상(세계 좌표, 받침 0도, 끄덕임 0도, 판 0도), 출력 부품인가)
    "받침 통": (P.tub("S"), True), "받침 덮개": (P.cover("S"), True), "고리": (P.clip(), True),
    "요크": (P.yoke(), True), "작은 기어 1": (P.pinion1(), True), "작은 기어 2": (P.pinion2(), True), "맞춤 링": (P.m2_ring(), True), "핀(오른쪽)": (P.pin(1), True), "핀(왼쪽)": (P.pin(-1), True),
    "모터 1": ((mb + mc).transform(P.motor1_T()), False), "모터 2": ((mb + mc).transform(P.motor2_T()), False),
    "머리 몸통": (Hd(P.head_shell()), True), "머리 뚜껑": (Hd(P.head_lid()), True), "모터 자리": (Hd(P.crown()), True), "부채꼴 기어": (Hd(P.sector_head()), True),
    "모터 3": (Hd(v1._m1(mb) + v1._m1(mc)), False), "파이": (Hd(pb + pc), False), "전원 플러그": (Hd(P.usb_plug_dummy()), False),
    "드라이버 3": (Hd((db + dc).transform(P.driver_T(0))), False), "드라이버 2": (Hd((db + dc).transform(P.driver_T(1))), False), "드라이버 1": (Hd((db + dc).transform(P.driver_T(2))), False),
    "회전 막대": (Rt(P.rotor_bar()), True), "판": (Rt(P.blade(150.0)), True), "균형 조각": (Rt(P.trim_plate()), True),
    "배터리": (P.battery_dummy("S"), False),
    "랜 플러그": (Hd(P.lan_plug_dummy()), False), "무선 동글": (Hd(P.dongle_dummy()), False), "탐지 동글": (Hd(P.dongle_dummy(DONGLE2)), False), "랜선 홈 마개": (P.lan_cap(), True), "버튼 모듈": (Hd(P.button_dummy()[0] + P.button_dummy()[1]), False),
}
BAL = Rt(P.balance_200())
px, pz = P.pin2_xz(); lx, lz = P.lock_xz(); zt = (POD_Z0 + BAR_Z0) / 2
hp = lambda p: np.array(Hd(M.sphere(0.01).translate(list(p))).bounding_box()).reshape(2, 3).mean(0)      # 머리 좌표의 점 -> 세계
hv = lambda v: hp(np.asarray(v, float)) - hp((0, 0, 0))
EAR = 0.8                                # 모터 귀(철판) 두께

NO_ROTOR = [k for k in PARTS if k not in ("회전 막대", "판", "균형 조각")]
# 이름, 머리 밑면 위치, 들어가는 방향, 길이, 지나가는 부품, 나사산이 물리는 부품, 조일 때 이미 붙어 있는 부품들(None = 전부), 머리 모양
S = []
for i, (x, y) in enumerate(POSTS):
    S.append(("받침 덮개 %d" % (i + 1), (x, y, COV_Z1), (0, 0, -1), 10, ["받침 덮개"], "받침 통", ["받침 통", "받침 덮개"], "접시"))
for i, (ex, ey) in enumerate(P.m1_ears()):
    S.append(("모터 1 귀 %d" % (i + 1), (ex, ey, DECK_Z1 + M1_PAD + EAR), (0, 0, -1), 6, ["모터 1"], "요크", ["요크", "모터 1"], "둥근"))
S.append(("받침 회전 멈춤", (YAW_STOP_XY[0], YAW_STOP_XY[1], DECK_Z1), (0, 0, -1), 10, [], "요크", ["요크", "모터 1"], "둥근"))
for i, (ex, ez) in enumerate(P.m2_ears()):
    S.append(("모터 2 귀 %d" % (i + 1), (ex, ARM_Y1 + EAR, ez), (0, -1, 0), 6, ["모터 2"], "요크", ["요크", "모터 1", "모터 2", "작은 기어 2", "맞춤 링", "받침 통", "받침 덮개"], "둥근"))
for i, (x, z) in enumerate(P.stop_screws_xz()):
    S.append(("끄덕임 멈춤 %d" % (i + 1), (x, -ARM_Y1, z), (0, 1, 0), 10, [], "요크", None, "둥근"))
S.append(("고정 나사(옮길 때)", (lx, ARM_Y1, lz), (0, -1, 0), 10, ["요크"], "부채꼴 기어", None, "둥근"))
for i, (sx, sy) in enumerate(CROWN_SCREWS):
    S.append(("모터 자리 %d" % (i + 1), hp((sx, sy, HZ1 + 2.0)), hv((0, 0, -1)), 8, ["모터 자리"], "머리 몸통", ["머리 몸통", "모터 자리", "모터 3", "파이", "드라이버 1", "드라이버 2", "드라이버 3"], "둥근"))
for i, (sx, sz) in enumerate(LID_SCREWS):
    S.append(("머리 뚜껑 %d" % (i + 1), hp((sx, HY0, sz)), hv((0, 1, 0)), 6, ["머리 뚜껑"], "머리 몸통", NO_ROTOR, "둥근"))       # 뚜껑은 머리를 요크에 올린 뒤 마지막에 닫는다
for i, (y, z) in enumerate(P.sector_mount_yz()):
    S.append(("부채꼴 기어 %d" % (i + 1), hp((HX + (SEC_Y1 - SEC_Y0) - SEC_CB_D, y, z)), hv((-1, 0, 0)), 6, ["부채꼴 기어"], "머리 몸통", ["머리 몸통", "모터 자리", "모터 3", "부채꼴 기어"], "둥근"))
for i, s in enumerate((1, -1)):
    S.append(("모터 3 귀 %d" % (i + 1), hp((-M_OFFSET, s * M_EAR, M1_ZT)), hv((0, 0, -1)), 6, ["모터 3"], "모터 자리", ["모터 자리", "모터 3"], "둥근"))
S.append(("판 고정", hp((COLLAR_X0 + 3.6, 0, (BAR_Z0 + BAR_Z1) / 2 + 6.0)), hv((0, 0, -1)), 8, ["판"], "회전 막대", None, "둥근"))
S.append(("균형 조각", hp((-POD_R1 - 3.0, 0, zt)), hv((1, 0, 0)), 8, ["균형 조각"], "회전 막대", None, "둥근"))

DRV_LEN = {"판 고정": 120.0}            # 판 고정 나사는 판 위끝 너머에서 내려와야 한다 (판 200 이면 약 10cm)
rows = []; problems = []
S2 = list(S) + [("판 고정 (판 200)",) + S[-2][1:]]
for name, p, d, L, through, thread, stage, head in S2:
    if name == "판 고정 (판 200)":
        PARTS["판"] = (Rt(P.blade(200.0)), True)
    p = np.asarray(p, float); d = np.asarray(d, float); d = d / np.linalg.norm(d)
    core = cyl(p, d, L, 1.25); shank = cyl(p, d, L, 1.5)
    ring = cyl(p, d, L, 1.5) - cyl(p - d * 0.5, d, L + 1, 1.31)
    area = np.pi * (1.5 ** 2 - 1.31 ** 2)
    blocked = {k: round((core ^ m).volume(), 2) for k, (m, pr) in PARTS.items() if (core ^ m).volume() > 0.05}
    rub = {k: round((shank ^ PARTS[k][0]).volume(), 2) for k in through if (shank ^ PARTS[k][0]).volume() > 0.3}
    eng = (ring ^ PARTS[thread][0]).volume() / area
    # 나사 끝이 물리는 부품 밖으로 얼마나 나오는가: 끝에서부터 심이 그 부품 안에 있지 않은 길이
    tip_free = 0.0
    for t in np.arange(L - 0.25, 0, -0.5):
        seg = cyl(p + d * t, d, 0.25, 1.6) - cyl(p + d * (t - 0.1), d, 0.5, 1.31)
        if (seg ^ PARTS[thread][0]).volume() / (np.pi * (1.6 ** 2 - 1.31 ** 2) * 0.25) > 0.5:
            break
        tip_free += 0.5
    hh = 2.6 if head == "둥근" else 0.2
    headc = cyl(p - d * hh, d, hh - 0.02, 3.0)
    head_hit = {k: round((headc ^ m).volume(), 2) for k, (m, pr) in PARTS.items() if (headc ^ m).volume() > 0.05}
    dl = DRV_LEN.get(name.split(" (")[0], 90.0)
    drv = cyl(p - d * (hh + dl), d, dl, 3.5)
    st = list(PARTS) if stage is None else stage
    acc = {k: round((drv ^ PARTS[k][0]).volume(), 1) for k in st if (drv ^ PARTS[k][0]).volume() > 0.5}
    ok = (not blocked) and (not rub) and eng >= 2.5 and (not head_hit) and (not acc)
    rows.append(dict(name=name, length=L, engage=round(eng, 1), tip_out=round(tip_free, 1), blocked=blocked, rub=rub, head_hit=head_hit, driver_blocked=acc, ok=bool(ok)))
    print("%-18s M3x%-2d 물림 %4.1f mm, 끝이 밖으로 %4.1f | 길 막힘 %s | 지나가는 구멍 걸림 %s | 머리 닿음 %s | 드라이버 길 %s %s" % (
        name, L, eng, tip_free, blocked or "-", rub or "-", head_hit or "-", acc or "비어 있음", "" if ok else "  <== 확인"))
    if not ok:
        problems.append(name)
    if name == "판 고정 (판 200)":
        PARTS["판"] = (Rt(P.blade(150.0)), True)
print("\n나사 %d개 가운데 확인할 것 %d개: %s" % (len(rows), len(problems), problems))
json.dump(rows, open(os.path.join(paths.RES, "audit_screws.json"), "w"), ensure_ascii=False, indent=1)
