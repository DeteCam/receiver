"""나사 전수 검사: 나사를 하나씩 모델에 넣어 본다.
    python3 audit_screws.py        ->  결과 폴더의 audit_screws.json
보는 것
  1) 나사 길(나사 심 굵기)에 플라스틱이 막고 있지 않은가  = 구멍이 있고, 깊이가 충분한가
  2) 지나가는 부품의 구멍이 헐거운가 (나사가 걸리지 않고 지나가야 한다)
  3) 나사산이 물리는 길이 (2.5mm 이상이어야 잡는다)
  4) 나사 끝이 어디서 끝나는가, 반대편으로 얼마나 나오는가
  5) 나사 머리가 다른 부품에 닿는가
  6) 드라이버가 들어갈 길(M3 는 지름 7, M2.5 는 지름 5, 길이 90)이 그 조립 단계에서 비어 있는가
나사는 세 가지: M3 (플라스틱에 스스로 나사산을 낸다), M2.5 (파이·드라이버 보드를 기둥에), M3 볼트 + 너트 (알루미늄 판, 축 고정)
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
N_COIN = P.COINS[(150.0, AL_T)]
PARTS = {   # 이름: (형상(세계 좌표, 받침 0도, 끄덕임 0도, 판 0도), 출력 부품인가)
    "받침 통": (P.tub("S"), True), "받침 덮개": (P.cover("S"), True), "고리": (P.clip(), True),
    "요크": (P.yoke(), True), "작은 기어 1": (P.pinion1(), True), "작은 기어 2": (P.pinion2(), True), "맞춤 링": (P.m2_ring(), True), "핀(오른쪽)": (P.pin(1), True), "핀(왼쪽)": (P.pin(-1), True),
    "핀 막이(오른쪽)": (P.pin_keeper(1), True), "핀 막이(왼쪽)": (P.pin_keeper(-1), True),
    "모터 1": ((mb + mc).transform(P.motor1_T()), False), "모터 2": ((mb + mc).transform(P.motor2_T()), False),
    "머리 몸통": (Hd(P.head_shell()), True), "머리 뚜껑": (Hd(P.head_lid()), True), "모터 자리": (Hd(P.crown()), True), "부채꼴 기어": (Hd(P.sector_head()), True), "마커 틀": (Hd(P.marker_frame()), True),
    "모터 3": (Hd(v1._m1(mb) + v1._m1(mc)), False), "파이": (Hd(pb + pc), False), "전원 플러그": (Hd(P.usb_plug_dummy()), False),
    "드라이버 3": (Hd((db + dc).transform(P.driver_T(0))), False), "드라이버 2": (Hd((db + dc).transform(P.driver_T(1))), False), "드라이버 1": (Hd((db + dc).transform(P.driver_T(2))), False),
    "회전 막대": (Rt(P.rotor_bar()), True), "알루미늄 판": (Rt(P.al_plate(150.0)), False), "마개": (Rt(P.coin_cap(N_COIN)), True), "동전": (Rt(P.coins(N_COIN)), False),
    "배터리": (P.battery_dummy("S"), False),
    "랜 플러그": (Hd(P.lan_plug_dummy()), False), "무선 동글": (Hd(P.dongle_dummy()), False), "탐지 동글": (Hd(P.dongle_dummy(DONGLE2)), False), "랜선 홈 마개": (P.lan_cap(), True), "버튼 모듈": (Hd(P.button_dummy()[0] + P.button_dummy()[1]), False),
}
px, pz = P.pin2_xz(); lx, lz = P.lock_xz(); kx, kz = P.keeper_xz()
hp = lambda p: np.array(Hd(M.sphere(0.01).translate(list(p))).bounding_box()).reshape(2, 3).mean(0)      # 머리 좌표의 점 -> 세계
hv = lambda v: hp(np.asarray(v, float)) - hp((0, 0, 0))
EAR = 0.8                                # 모터 귀(철판) 두께
SPEC = {3.0: dict(core=1.25, shank=1.5, pilot=1.31, head_r=3.0, head_h=2.6, drv=3.5), 2.5: dict(core=1.0, shank=1.25, pilot=1.06, head_r=2.4, head_h=2.0, drv=2.5)}

ROTOR = ("회전 막대", "알루미늄 판", "마개", "동전")
NO_ROTOR = [k for k in PARTS if k not in ROTOR]
INSIDE = ["머리 몸통", "파이", "드라이버 1", "드라이버 2", "드라이버 3"]          # 머리 속을 채우는 단계에서 이미 붙어 있는 것 (뚜껑은 아직 없다)
HEAD_ONLY = ["머리 몸통", "머리 뚜껑", "모터 자리", "모터 3", "부채꼴 기어", "마커 틀", "파이", "드라이버 1", "드라이버 2", "드라이버 3"]
# 이름, 머리 밑면 위치, 들어가는 방향, 길이, 지나가는 부품, 나사산이 물리는 부품, 조일 때 이미 붙어 있는 부품들(None = 전부), 머리 모양, 그 밖의 것
#   그 밖의 것: m = 나사 굵기(기본 3), nut = 너트가 놓이는 곳(머리 밑면에서의 거리: 가까운 면), sunk = 머리가 파 놓은 자리에 숨는가
S = []
for i, (x, y) in enumerate(POSTS):
    S.append(("받침 덮개 %d" % (i + 1), (x, y, COV_Z1), (0, 0, -1), 10, ["받침 덮개"], "받침 통", ["받침 통", "받침 덮개"], "접시", {}))
for i, (ex, ey) in enumerate(P.m1_ears()):
    S.append(("모터 1 귀 %d" % (i + 1), (ex, ey, DECK_Z1 + M1_PAD + EAR), (0, 0, -1), 6, ["모터 1"], "요크", ["요크", "모터 1"], "둥근", {}))
S.append(("받침 회전 멈춤", (YAW_STOP_XY[0], YAW_STOP_XY[1], DECK_Z1), (0, 0, -1), 10, [], "요크", ["요크", "모터 1"], "둥근", {}))
for i, (ex, ez) in enumerate(P.m2_ears()):
    S.append(("모터 2 귀 %d" % (i + 1), (ex, ARM_Y1 + EAR, ez), (0, -1, 0), 6, ["모터 2"], "요크", ["요크", "모터 1", "모터 2", "작은 기어 2", "맞춤 링", "받침 통", "받침 덮개"], "둥근", {}))
for i, (x, z) in enumerate(P.stop_screws_xz()):
    S.append(("끄덕임 멈춤 %d" % (i + 1), (x, -ARM_Y1, z), (0, 1, 0), 10, [], "요크", None, "둥근", {}))
S.append(("고정 나사(옮길 때)", (lx, ARM_Y1, lz), (0, -1, 0), 10, ["요크"], "부채꼴 기어", None, "둥근", {}))
for i, s_ in enumerate((1, -1)):                                           # 핀 막이: 머리가 파 놓은 자리(깊이 2.5)에 숨는다
    S.append(("핀 막이 (%s)" % ("오른쪽" if s_ > 0 else "왼쪽"), (kx, s_ * (ARM_Y1 + 3.9 - 2.5), kz), (0, -s_, 0), 6, ["핀 막이(%s)" % ("오른쪽" if s_ > 0 else "왼쪽")], "요크", None, "둥근", dict(sunk=True)))
for i, (sx, sy) in enumerate(CROWN_SCREWS):
    S.append(("모터 자리 %d" % (i + 1), hp((sx, sy, HZ1 + 2.0)), hv((0, 0, -1)), 8, ["모터 자리"], "머리 몸통", ["머리 몸통", "모터 자리", "모터 3", "파이", "드라이버 1", "드라이버 2", "드라이버 3"], "둥근", {}))
for i, (sx, sz) in enumerate(LID_SCREWS):
    S.append(("머리 뚜껑 %d" % (i + 1), hp((sx, HY0, sz)), hv((0, 1, 0)), 6, ["머리 뚜껑"], "머리 몸통", NO_ROTOR, "둥근", {}))       # 뚜껑은 머리를 요크에 올린 뒤 마지막에 닫는다
for i, (y, z) in enumerate(P.sector_mount_yz()):
    S.append(("부채꼴 기어 %d" % (i + 1), hp((HX + (SEC_Y1 - SEC_Y0) - SEC_CB_D, y, z)), hv((-1, 0, 0)), 6, ["부채꼴 기어"], "머리 몸통", ["머리 몸통", "모터 자리", "모터 3", "부채꼴 기어"], "둥근", {}))
for i, s_ in enumerate((1, -1)):
    S.append(("모터 3 귀 %d" % (i + 1), hp((-M_OFFSET, s_ * M_EAR, M1_ZT)), hv((0, 0, -1)), 6, ["모터 3"], "모터 자리", ["모터 자리", "모터 3"], "둥근", {}))
for i, (x, z) in enumerate(P.pi_hole_xz()):                               # 파이: M2.5 x 6, 앞에서 기둥으로
    S.append(("파이 %d" % (i + 1), hp((x, -PI_T / 2, z)), hv((0, 1, 0)), 6, ["파이"], "머리 몸통", ["머리 몸통", "파이"], "둥근", dict(m=2.5)))
yf = HY1 - HWB - DRV_POST_H - 1.6
for k in range(3):                                                         # 드라이버 3장: M2.5 x 6, 앞에서 기둥으로
    for i, (x, z) in enumerate(P.driver_holes(k)):
        S.append(("드라이버 %d-%d" % (3 - k, i + 1), hp((x, yf, z)), hv((0, 1, 0)), 6, ["드라이버 %d" % (3 - k)], "머리 몸통", INSIDE, "둥근", dict(m=2.5)))
for i, (sx, sz) in enumerate(P.marker_screws()):                           # 마커 틀: M3 x 6, 등판 겉에서
    S.append(("마커 틀 %d" % (i + 1), hp((sx, HY1 + MK_FRAME["t"], sz)), hv((0, -1, 0)), 6, ["마커 틀"], "머리 몸통", HEAD_ONLY, "둥근", {}))      # 머리를 요크에 올리기 전에 단다
xf = FL_X1 - FL_T
for i, (y, z) in enumerate(PLATE_BOLTS):                                   # 알루미늄 판: M3 x 8 볼트 + 너트. 판 바깥에서 넣고, 판 받이 안쪽의 육각 자리에 너트
    S.append(("판 볼트 %d" % (i + 1), hp((FL_X1 + AL_T, y, z)), hv((-1, 0, 0)), 8, ["알루미늄 판", "회전 막대"], None, None, "둥근", dict(nut=FL_X1 + AL_T - (xf + 2.2))))
zs = M1_ZT + 7.0
S.append(("축 고정", hp((0, 9.2, zs)), hv((0, -1, 0)), 8, ["회전 막대"], None, None, "둥근", dict(nut=9.2 - 6.5, tip="모터 축을 누른다")))
gap = P.coin_gap(N_COIN)
for i, s_ in enumerate((1, -1)):                                           # 균형추 마개: M3 x 8, 아래에서 위로
    S.append(("균형추 마개 %d" % (i + 1), hp((POD_XC, s_ * CAP_EAR_Y, POD_Z0 - gap - CAP_T)), hv((0, 0, 1)), 8, ["마개"], "회전 막대", None, "둥근", {}))

rows = []; problems = []
for name, p, d, L, through, thread, stage, head, opt in S:
    q = SPEC[opt.get("m", 3.0)]
    p = np.asarray(p, float); d = np.asarray(d, float); d = d / np.linalg.norm(d)
    core = cyl(p, d, L, q["core"]); shank = cyl(p, d, L, q["shank"])
    ring = cyl(p, d, L, q["shank"]) - cyl(p - d * 0.5, d, L + 1, q["pilot"])
    area = np.pi * (q["shank"] ** 2 - q["pilot"] ** 2)
    blocked = {k: round((core ^ m).volume(), 2) for k, (m, pr) in PARTS.items() if (core ^ m).volume() > 0.05}
    rub = {k: round((shank ^ PARTS[k][0]).volume(), 2) for k in through if (shank ^ PARTS[k][0]).volume() > 0.3}
    tip_free = 0.0; nut_hit = {}
    if thread is not None:
        eng = (ring ^ PARTS[thread][0]).volume() / area
        # 나사 끝이 물리는 부품 밖으로 얼마나 나오는가: 끝에서부터 심이 그 부품 안에 있지 않은 길이
        for t in np.arange(L - 0.25, 0, -0.5):
            seg = cyl(p + d * t, d, 0.25, q["shank"] + 0.1) - cyl(p + d * (t - 0.1), d, 0.5, q["pilot"])
            if (seg ^ PARTS[thread][0]).volume() / (np.pi * ((q["shank"] + 0.1) ** 2 - q["pilot"] ** 2) * 0.25) > 0.5:
                break
            tip_free += 0.5
    else:                                                                  # 볼트 + 너트: 너트(맞변 5.5, 두께 2.4)가 제자리에 들어가는가, 볼트 끝이 너트를 지나가는가
        n0 = opt["nut"]; nut = (cyl(p + d * n0, d, 2.4, 5.5 / 2 / np.cos(np.radians(30.0)), 6) - cyl(p + d * (n0 - 0.5), d, 3.4, 1.5))
        # 육각 방향을 맞춘다: 판 볼트는 변이 위아래로 수평, 축 고정 나사는 X 로 5.8 폭의 홈
        best = None
        for a in (0.0, 30.0):
            nn = nut.translate(list(-p)).rotate(list(np.degrees(a * np.radians(1.0)) * d)).translate(list(p)) if a else nut
            hit = {k: round((nn ^ m).volume(), 2) for k, (m, pr) in PARTS.items() if (nn ^ m).volume() > 0.3}
            if best is None or sum(hit.values()) < sum(best.values()):
                best = hit
        nut_hit = best
        eng = max(0.0, min(2.4, L - n0)); tip_free = max(0.0, L - n0 - 2.4)
    hh = q["head_h"] if head == "둥근" else 0.2
    headc = cyl(p - d * hh, d, hh - 0.02, q["head_r"])
    head_hit = {k: round((headc ^ m).volume(), 2) for k, (m, pr) in PARTS.items() if (headc ^ m).volume() > 0.05}
    drv = cyl(p - d * (hh + 90.0), d, 90.0, q["drv"])
    st = list(PARTS) if stage is None else stage
    acc = {k: round((drv ^ PARTS[k][0]).volume(), 1) for k in st if (drv ^ PARTS[k][0]).volume() > 0.5}
    ok = (not blocked) and (not rub) and eng >= (2.4 if thread is None else 2.5) and (not head_hit) and (not acc) and (not nut_hit)
    rows.append(dict(name=name, m=opt.get("m", 3.0), length=L, nut=thread is None, engage=round(eng, 1), tip_out=round(tip_free, 1), blocked=blocked, rub=rub, head_hit=head_hit, nut_hit=nut_hit,
                     driver_blocked=acc, ok=bool(ok), note=opt.get("tip", "")))
    print("%-16s M%gx%-2d %s %4.1f mm, 끝이 밖으로 %4.1f | 길 막힘 %s | 지나가는 구멍 걸림 %s | 머리 닿음 %s%s | 드라이버 길 %s %s" % (
        name, opt.get("m", 3.0), L, "너트 물림" if thread is None else "물림", eng, tip_free, blocked or "-", rub or "-", head_hit or "-",
        (" | 너트 자리 걸림 %s" % (nut_hit or "-")) if thread is None else "", acc or "비어 있음", "" if ok else "  <== 확인"))
    if not ok:
        problems.append(name)
# 마커 종이를 나중에 바꿀 때: 머리를 재는 자세(54.7도)로 숙이면 등판이 위를 본다. 그 자세에서 드라이버 길이 비는지 본다 (요크·받침은 그대로, 머리 쪽만 돌린다)
FIXED = union([PARTS[k][0] for k in ("받침 통", "받침 덮개", "요크", "모터 1", "모터 2", "핀 막이(오른쪽)", "핀 막이(왼쪽)")])
tilt_block = 0.0
for sx, sz in P.marker_screws():
    drv = P.pose_head(cy(3.5, HY1 + MK_FRAME["t"] + 2.6, HY1 + MK_FRAME["t"] + 92.6, sx, sz), 0, TILT3)
    tilt_block = max(tilt_block, (drv ^ FIXED).volume())
print("마커 틀 나사: 머리를 54.7도로 숙였을 때 드라이버 길이 요크·받침에 막히는 부피 = %.1f mm3 (0 이면 요크에 올린 뒤에도 조일 수 있다)" % tilt_block)
rows.append(dict(name="마커 틀 (54.7도로 숙이고)", m=3.0, length=6, nut=False, engage=0.0, tip_out=0.0, blocked={}, rub={}, head_hit={}, nut_hit={}, driver_blocked={} if tilt_block < 0.5 else {"요크·받침": round(tilt_block, 1)},
                 ok=bool(tilt_block < 0.5), note="종이를 바꿀 때", extra=True))
by = {}
for r in rows:
    if r.get("extra"):
        continue
    k = "M%g x %d%s" % (r["m"], r["length"], " + 너트" if r["nut"] else ""); by[k] = by.get(k, 0) + 1
print("\n나사 %d개 가운데 확인할 것 %d개: %s" % (len([r for r in rows if not r.get("extra")]), len(problems), problems))
print("종류별:", by)
json.dump(rows, open(os.path.join(paths.RES, "audit_screws.json"), "w"), ensure_ascii=False, indent=1)
