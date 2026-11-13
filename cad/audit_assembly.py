"""조립 경로 검사: 조립 안내의 순서대로 부품을 제자리까지 밀어 넣으며, 가는 길에 걸리는 곳이 있는지 본다.
    python3 audit_assembly.py        ->  결과 폴더의 audit_assembly.json
'가는 길' = 제자리에서 1mm 넘게 떨어진 모든 위치. 여기서 겹치면 그 방향으로는 끼울 수 없다는 뜻이다.
'제자리' 겹침은 일부러 꽉 끼우는 자리(축, 핀)에서만 조금 나온다.
딸깍 끼우는 고리는 가는 길에서 통과 겹치는 것이 정상이라 따로 본다 (벌어지는 양)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *

mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
motor = mb + mc
rows = []


def sweep(group, name, moving, fixed, d, start, how, step=1.0, allow_seat=1.0, allow_path=0.5, note=""):
    d = np.asarray(d, float)
    worst = (0.0, None)
    for t in np.arange(start, 0.99, -step):
        v = (moving.translate(list(d * t)) ^ fixed).volume()
        if v > worst[0]:
            worst = (v, t)
    seat = (moving ^ fixed).volume()
    ok = worst[0] < allow_path and seat <= allow_seat
    rows.append(dict(group=group, step=name, how=how, path_overlap_mm3=round(worst[0], 2), at_mm=None if worst[1] is None else float(worst[1]),
                     seat_overlap_mm3=round(max(seat, 0.0), 2), ok=bool(ok), note=note))
    print("%s %-30s %-22s 가는 길 겹침 %7.2f mm3%s | 제자리 겹침 %5.2f %s %s" % (
        group, name, how, worst[0], "" if worst[1] is None else " (제자리에서 %.0fmm)" % worst[1], max(seat, 0.0), "" if ok else "  <== 확인", note))
    return ok


tub, cov, yoke = P.tub("S"), P.cover("S"), P.yoke()
m1 = motor.transform(P.motor1_T()); m2 = motor.transform(P.motor2_T())
stop = cz(1.5, DECK_Z1 - 10.0, DECK_Z1, *YAW_STOP_XY) + cz(3.0, DECK_Z1, DECK_Z1 + 2.6, *YAW_STOP_XY)

# ---- 1 받침
cap = P.lan_cap()
sweep("1", "랜선 홈 마개 -> 통 (덮개 덮기 전)", cap, tub, (0, 0, 1), 16, "위에서 벽에 끼워 내린다", note="랜선을 받침 안으로 넣어 쓸 때는 끼우지 않는다")
sweep("1", "받침 덮개 -> 통", cov, tub + cap, (0, 0, 1), 30, "위에서 내린다")
sweep("1", "배터리 -> 통 (덮개 닫은 채)", P.battery_dummy("S"), tub + cov, (0, -1, 0), 130, "옆 창으로 민다")
sweep("1", "배터리 L -> 통 L", P.battery_dummy("L"), P.tub("L") + P.cover("L"), (0, -1, 0), 170, "옆 창으로 민다")
usb_micro = box(-5.75, 5.75, -3.75, 3.75, 0, 20)                               # 마이크로 5핀 플러그의 손잡이(11.5 x 7.5)
sweep("1", "전원선 플러그 -> 가운데 통", usb_micro.translate([0, 0, TUB_H - 25]), cov, (0, 0, -1), 0, "아래에서 위로", note="통 구멍 지름 %.0f, 플러그 대각선 %.1f" % (2 * TUBE_RI, np.hypot(11.5, 7.5)))
v = max((usb_micro.translate([0, 0, z]) ^ cov).volume() for z in np.arange(TUB_H - 25, TUBE_Z1 + 5, 1.0))
rows[-1]["path_overlap_mm3"] = round(v, 2); rows[-1]["ok"] = bool(v < 0.5)
rj45 = box(-6.75, 6.75, -7.25, 7.25, 0, 30)                                    # 덮개(부트) 달린 랜 플러그 13.5 x 14.5
v = max((rj45.translate([0, 0, z]) ^ cov).volume() for z in np.arange(TUB_H - 35, TUBE_Z1 + 5, 1.0))
rows.append(dict(group="1", step="랜선 플러그 -> 가운데 통", how="아래에서 위로", path_overlap_mm3=round(v, 2), seat_overlap_mm3=0.0, ok=bool(v < 0.5),
                 note="통 구멍 지름 %.1f, 플러그 대각선 %.1f" % (2 * TUBE_RI, np.hypot(13.5, 14.5))))
print("1 %-30s 가는 길 겹침 %.2f mm3 (구멍 %.1f, 대각선 %.1f) %s" % ("랜선 플러그 -> 가운데 통", v, 2 * TUBE_RI, np.hypot(13.5, 14.5), "" if v < 0.5 else " <== 확인"))
both = usb_micro.translate([0, 3.2, 0]) + cz(3.0, 0, 30, 0, -6.0)              # 랜선(지름 6)이 이미 지나가 있을 때 그 옆으로 전원선 플러그
v = max((both.translate([0, 0, z]) ^ cov).volume() for z in np.arange(TUB_H - 35, TUBE_Z1 + 5, 1.0))
rows.append(dict(group="1", step="전원선 플러그 -> 가운데 통 (랜선 옆으로)", how="아래에서 위로", path_overlap_mm3=round(v, 2), seat_overlap_mm3=0.0, ok=bool(v < 0.5), note="랜선 지름 6 으로 봤다"))
print("1 %-30s 가는 길 겹침 %.2f mm3 %s" % ("전원선 플러그 (랜선 옆으로)", v, "" if v < 0.5 else " <== 확인"))

# ---- 2 요크
sweep("2", "모터 1 -> 회전판", m1, yoke, (0, 0, 1), 30, "위에서 내린다", allow_seat=5.0)
sweep("2", "작은 기어 1 -> 모터 1 축", P.pinion1(), yoke + m1, (0, 0, -1), 20, "아래에서 민다")
sx, sy = P.m1_shaft()
best = min(((cov ^ P.pinion(ph).translate([sx, sy, DECK_Z0 - 6.0])).volume(), ph) for ph in np.arange(0, 36, 0.5))
ysub = union([yoke, m1, P.pinion(best[1]).translate([sx, sy, DECK_Z0 - 6.0]), stop])
sweep("2", "요크 묶음 -> 받침", ysub, tub + cov, (0, 0, 1), 30, "위에서 내린다", allow_seat=0.5, note="전원선이 통에 꿰인 채")
clip = P.clip()
sweep("2", "고리 -> 가운데 통 (통 말고 다른 것)", clip, union([yoke, m1, stop]), (-1, 0, 0), 42, "뒤에서 앞으로 민다", note="상자 안에서. 고리 앞쪽은 앞벽의 문 구멍으로 조금 나온다")
ro = TUBE_RO - 1.5 + 0.15
spread = (TUBE_RO - 1.5) - (ro - 0.9)
vmax = max((clip.translate([-t, 0, 0]) ^ cov).volume() for t in np.arange(45, 0.9, -1.0))
ok = spread <= 1.0 and (clip ^ cov).volume() < 0.5
rows.append(dict(group="2", step="고리 -> 가운데 통 (딸깍)", how="뒤에서 앞으로 민다", path_overlap_mm3=round(vmax, 2), seat_overlap_mm3=round((clip ^ cov).volume(), 2), ok=bool(ok),
                 note="입구 %.1f, 홈 지름 %.1f: 양쪽이 %.2f mm 씩 벌어졌다가 제자리에서 풀린다" % (2 * (ro - 0.9), 2 * (TUBE_RO - 1.5), spread)))
print("2 %-30s 입구 %.1f, 홈 지름 %.1f -> 한쪽 %.2f mm 씩 벌어짐 (제자리 겹침 %.2f) %s" % ("고리 -> 가운데 통 (딸깍)", 2 * (ro - 0.9), 2 * (TUBE_RO - 1.5), spread, (clip ^ cov).volume(), "" if ok else " <== 확인"))

# ---- 3 머리 속
shell, lid, crown, sec = P.head_shell(), P.head_lid(), P.crown(), P.sector_head()
pi = pb + pc
drvs = [(db + dc).transform(P.driver_T(k)) for k in range(3)]
drv = union(drvs)
keep = union([P.driver_keepout(k) for k in range(3)] + [P.gpio_keepout()])
m3 = v1._m1(motor)
plug = P.usb_plug_dummy() ^ box(-100, 100, -100, 100, BZ1 - 16, 100)
sweep("3", "파이 -> 몸통 못", pi, shell, (0, -1, 0), 28, "앞에서 뒤로 민다", allow_seat=0.5)
for k in range(3):
    sweep("3", "드라이버 %d -> 뒷벽 (파이가 있는 채)" % (3 - k), drvs[k], shell + pi, (0, -1, 0), 30, "앞에서 뒤로 곧게")
    side = min(drvs[k].translate([0, -t, 0]).min_gap(shell + pi, 5.0) for t in (26.0, 20.0, 14.0))
    rows[-1]["note"] = "들어가는 길의 옆 틈 %.2f mm" % side; print("     옆 틈 %.2f mm" % side)
sweep("3", "ㄱ자 젠더 -> 파이 전원 단자", plug, shell + pi, (-1, 0, 0), 12, "왼쪽 창으로 옆에서", allow_seat=0.5)
bpcb, bsw = P.button_dummy(); btn = bpcb + bsw
sweep("3", "버튼 모듈 -> 받침 살 (파이가 있는 채)", btn, union([shell, pi, plug]), (0, -1, 0), 20, "앞에서 뒤로 댄다", allow_seat=0.5)

# ---- 4 머리 겉
jst = box(-7.45, 7.45, -2.9, 2.9, 0, 7.0).translate([(CABLE_HOLE[0] + CABLE_HOLE[1]) / 2, (CABLE_HOLE[2] + CABLE_HOLE[3]) / 2, HZ1 + 12.0])
v = max((jst.translate([0, 0, -t]) ^ crown).volume() for t in np.arange(0, 22, 0.5))
rows.append(dict(group="4", step="모터 3 흰 플러그 -> 모터 자리의 네모 구멍", how="위에서 아래로 곧게", path_overlap_mm3=round(v, 2), seat_overlap_mm3=0.0, ok=bool(v < 0.5),
                 note="플러그 14.9 x 5.8, 구멍 %.0f x %.0f" % (CABLE_HOLE[1] - CABLE_HOLE[0], CABLE_HOLE[3] - CABLE_HOLE[2])))
print("4 %-30s 가는 길 겹침 %.2f mm3 %s" % ("흰 플러그 -> 네모 구멍", v, "" if v < 0.5 else " <== 확인"))
sweep("4", "모터 3 -> 모터 자리", m3, crown, (0, 0, 1), 30, "위에서 내린다", allow_seat=5.0)
sweep("4", "모터 자리(모터 3) -> 몸통", crown + m3, union([shell, pi, drv, keep, plug, btn]), (0, 0, 1), 25, "위에서 내린다", allow_seat=0.5)
sweep("4", "부채꼴 기어 -> 머리 옆면", sec, union([shell, crown]), (1, 0, 0), 20, "옆에서 댄다", allow_seat=0.5)

# ---- 5 머리 올리기 (뚜껑은 아직 열어 둔 채)
ring = P.m2_ring(); shaft2 = v1.motor_shaft_dummy().transform(P.motor2_T())
sweep("5", "맞춤 링 -> 모터 2 축 둘레 턱", ring, m2 + shaft2, (0, -1, 0), 30, "넓은 구멍을 모터 쪽으로 해서 씌운다", note="턱 지름 9, 링 구멍 %.1f" % (2 * M2_RING_BOSS[0]))
sweep("5", "작은 기어 2 -> 모터 2 축 (팔에 달기 전에)", P.pinion2(), m2 + ring, (0, -1, 0), 30, "축 끝에서 턱까지 민다", note="링과 기어 사이 %.1f mm" % (ARM_Y1 - M2_RING_L - PIN2_Y1))
sweep("5", "모터 2 + 링 + 작은 기어 -> 오른쪽 팔", m2 + ring + P.pinion2(), yoke, (0, 1, 0), 30, "바깥에서 구멍으로 넣는다", allow_seat=5.0,
      note="구멍 지름 %.1f, 링 지름 %.1f, 기어 바깥 지름 %.1f" % (2 * M2_HOLE_R, 2 * M2_RING_R, 2 * P_RA))
jst5 = box(-7.45, 7.45, -2.9, 2.9, 0, 7.0)                                         # 모터 전선 끝의 흰 5핀 플러그 14.9 x 5.8 (세워서 지나간다)
x0, x1, z0, z1 = P.M2_SLOT
jv = max((jst5.rotate([90, 0, 0]).rotate([0, 90, 0]).translate([(x0 + x1) / 2, CHEEK_Y + 6.0 - t, (z0 + z1) / 2 - 7.45 + 7.45]) ^ yoke).volume() for t in np.arange(0, 16, 0.5))
rows.append(dict(group="5", step="모터 2 흰 플러그 -> 칸막이 구멍", how="기어 칸에서 상자 안으로", path_overlap_mm3=round(jv, 2), seat_overlap_mm3=0.0, ok=bool(jv < 0.5),
                 note="플러그 14.9 x 5.8, 구멍 %.0f x %.0f" % (x1 - x0, z1 - z0)))
print("5 %-30s 가는 길 겹침 %.2f mm3 %s" % ("모터 2 흰 플러그 -> 칸막이 구멍", jv, "" if jv < 0.5 else " <== 확인"))
lanp = P.lan_plug_dummy(); dongle = P.dongle_dummy()
sweep("5", "무선 동글 -> 파이 USB (머리를 올리기 전)", dongle, union([shell, pi, plug, btn, lanp]), (0, 0, -1), 30, "아래에서 위로 꽂는다", allow_seat=120.0,
      note="파이 단자와 겹치는 부피는 플러그가 들어간 것이라 뺀다" )
rows[-1]["seat_overlap_mm3"] = round((dongle ^ union([shell, plug, btn, lanp])).volume(), 2); rows[-1]["ok"] = bool(rows[-1]["seat_overlap_mm3"] < 0.5)
dongle2 = P.dongle_dummy(DONGLE2)
sweep("5", "탐지 동글 -> 파이 가운데 USB (머리를 올리기 전)", dongle2, union([shell, pi, plug, btn, lanp, dongle]), (0, 0, -1), 30, "아래에서 위로 꽂는다", allow_seat=120.0,
      note="먼저 꽂은 무선 동글 옆(틈 2mm)으로 들어간다" )
rows[-1]["seat_overlap_mm3"] = round((dongle2 ^ union([shell, plug, btn, lanp, dongle])).volume(), 2); rows[-1]["ok"] = bool(rows[-1]["seat_overlap_mm3"] < 0.5)
HEADALL = union([shell, crown, sec, m3, pi, drv, plug, btn, lanp, dongle, dongle2])         # 뚜껑 없이. 랜선과 동글 둘도 꽂은 채
YSET = union([yoke, m1, m2, P.pinion2(), ring])
worst = 0.0; trace = []
for dz in list(np.arange(60.0, 10.0, -5.0)) + list(np.arange(10.0, -0.01, -0.5)):
    best = min(((P.pose_head(HEADALL, 0, p).translate([0, 0, dz]) ^ YSET).volume(), p) for p in np.arange(-5.0, 5.01, 0.25))
    worst = max(worst, best[0]); trace.append((float(dz), float(best[1]), round(best[0], 2)))
ok = worst < 3.0
rows.append(dict(group="5", step="머리 -> 요크 (모터 2와 작은 기어가 달린 채)", how="위에서 내리며 조금 끄덕인다", path_overlap_mm3=round(worst, 2), seat_overlap_mm3=trace[-1][2], ok=bool(ok),
                 note="이가 맞는 각을 ±5도 안에서 찾았을 때"))
print("5 %-30s %-22s 가는 길 겹침 %7.2f mm3 (각을 ±5도 안에서 맞췄을 때) %s" % ("머리 -> 요크", "내리며 조금 끄덕인다", worst, "" if ok else "  <== 확인"))
HEADY = P.pose_head(HEADALL, 0, 0)
sweep("5", "오른쪽 핀 -> 축 구멍", P.pin(1), union([yoke, HEADY, m2]), (0, 1, 0), 25, "바깥에서 민다", allow_seat=3.0)
sweep("5", "왼쪽 핀 -> 축 구멍", P.pin(-1), union([yoke, HEADY]), (0, -1, 0), 25, "바깥에서 민다", allow_seat=3.0)
YH = P.yoke_to_head(union([YSET, P.pin(1), P.pin(-1)]))                            # 요크 쪽 부품을 머리 좌표로 (끄덕임 0도)
sweep("5", "뚜껑 -> 몸통 (머리가 요크에 앉은 채)", lid, union([shell, pi, drv, keep, plug, btn, crown, m3, sec, YH]), (0, -1, 0), 40, "앞에서 뒤로 민다", allow_seat=0.5)

# ---- 6 판
bar = P.rotor_bar()
sweep("6", "회전 막대 -> 모터 3 축", bar, union([crown, m3, shell, lid]), (0, 0, 1), 25, "위에서 누른다", allow_seat=1.0)
sweep("6", "판 -> 막대 끝", P.blade(150.0), union([bar, crown, shell, lid]), (1, 0, 0), 14, "바깥에서 안으로 민다", allow_seat=0.5)
sweep("6", "판 200 -> 막대 끝", P.blade(200.0), union([bar, crown, shell, lid]), (1, 0, 0), 14, "바깥에서 안으로 민다", allow_seat=0.5)
sweep("6", "균형 조각 -> 균형추", P.trim_plate(), bar, (-1, 0, 0), 10, "바깥에서 댄다", allow_seat=0.5)
bad = [r["step"] for r in rows if not r["ok"]]
print("\n조립 단계 %d개 가운데 확인할 것 %d개: %s" % (len(rows), len(bad), bad))
json.dump(rows, open(os.path.join(paths.RES, "audit_assembly.json"), "w"), ensure_ascii=False, indent=1)
