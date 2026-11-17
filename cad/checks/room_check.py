"""자리 검토: 사 오는 부품(모터 3, 드라이버 3, 파이, 배터리, 전선)이 다 들어가는지 숫자로 확인한다.
    python3 room_check.py     ->  결과 폴더의 room_check.json, 화면에 표
틈은 모델끼리의 가장 가까운 거리(mm). 0 은 일부러 맞닿는 자리(나사로 붙는 면, 얹히는 면)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import parts2 as P
from cadlib import *
from p2 import *
import parts as v1

import paths
OUT = paths.RES + '/'
CABLE_STOCK = 240.0          # 28BYJ-48 에 달려 나오는 전선 길이 (보통 23~25cm) [실측]


def gap(a, b, search=60.0):
    if (a ^ b).volume() > 0.3:
        return -1.0
    return float(a.min_gap(b, search))


def bb(m):
    b = m.bounding_box()
    return [round(b[3] - b[0], 1), round(b[4] - b[1], 1), round(b[5] - b[2], 1)]


def head_pt(p, pitch):
    """머리 좌표의 점 -> 요크 좌표"""
    x, y, z = p
    x, y, z = x, y - Y_T, z - S_T
    x, y = -y, x
    pr = np.radians(pitch)
    x, z = x * np.cos(pr) + z * np.sin(pr), -x * np.sin(pr) + z * np.cos(pr)
    return np.array([x, y, z + H_T])


def path_len(pts):
    pts = [np.asarray(p, float) for p in pts]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))


def main():
    rows = []; res = {}
    mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
    shell, lid, crown = P.head_shell(), P.head_lid(), P.crown()
    headp = shell + lid + crown
    lidflat = box(-HX, HX, HY0, HYL, HZ0, HZ1)                 # 뚜껑의 평평한 판 (버튼 기판을 누르는 살은 뺀 것)

    # ---------- 머리 안 ----------
    pi = pb + pc
    drv = [(db + dc).transform(P.driver_T(k)) for k in range(3)]
    keep = [P.driver_keepout(k) for k in range(3)]
    rows.append(("파이 3B+", "머리 안 (뒷벽의 기둥 4개에 M2.5 나사)", bb(pi), "뚜껑 안쪽 면까지 %.1f, 모터 자리 바닥까지 %.1f" % (gap(pc, lidflat), gap(pi, crown))))
    rows.append(("GPIO에 꽂은 점퍼선", "파이 오른쪽 변 (선이 꺾이는 높이 23)", bb(P.gpio_keepout()), "뚜껑 안쪽 면까지 %.1f" % gap(P.gpio_keepout(), lidflat)))
    for k in range(3):
        others = [pi] + [drv[j] for j in range(3) if j != k]
        rows.append(("드라이버 %d" % (3 - k), "머리 안 오른쪽 (뒷벽의 기둥 4개에 M2 나사)", bb(drv[k]),
                     "몸통까지 %.1f, 옆 부품까지 %.1f, 점퍼선 꽂고 뚜껑 안쪽 면까지 %.1f" % (gap(drv[k], shell), min(gap(drv[k], o) for o in others), gap(keep[k], lidflat))))
    rows.append(("전원 플러그 (ㄱ자)", "파이 왼쪽 변, 선은 아래로", bb(P.usb_plug_dummy()), "옆벽까지 %.1f" % gap(P.usb_plug_dummy(), shell)))
    bp, bs = P.button_dummy()
    rows.append(("전원 버튼 모듈 (KY-004)", "머리 안 왼쪽, 버튼은 앞을 본다 (뚜껑의 살이 눌러 고정)", bb(bp + bs), "버튼 몸통에서 뚜껑 안쪽 면까지 %.1f, 버튼 머리는 뚜껑 겉면보다 %.1f 안쪽, 파이 부품까지 %.1f" % (
        (BTN_PCB_Y - 3.5) - HYL, (BTN_PCB_Y - 5.0) - HY0, gap(bp + bs, pi))))
    rows.append(("랜선 플러그", "파이 아래 단자, 아래로 나온다", bb(P.lan_plug_dummy()), "머리 치마 밖으로 %.0f mm" % (HZ0 - (BZ0 - 27.0))))
    xd = P.xiao_dummy(); fd = P.fpc_dummy()
    rows.append(("ESP32 (XIAO ESP32-C5)", "뚜껑 안쪽 왼쪽, 세워서 홈 받침 2개 + 막음쇠 (점퍼선 소켓이 +X, 파이 GPIO 쪽)", bb(xd), "파이 부품까지 %.1f, 버튼 모듈까지 %.1f, 점퍼선 소켓 끝에서 GPIO 점퍼선 자리까지 %.1f" % (
        gap(xd, pi), gap(xd, union(list(P.button_dummy()))), gap(xd, P.gpio_keepout()))))
    rows.append(("ESP32 안테나 (FPC)", "머리 안 천장(모터 자리 밑면), 판 회전축 위 파이 안테나에서 %.1f mm 위" % (HZ1 + ESP_ANT["pocket"] - ESP_ANT["t"] / 2), bb(fd), "파이 기판 윗변까지 %.1f, 뒷벽까지 %.1f" % (gap(fd, pb), gap(fd, shell))))
    a = np.array([-ESP_ANT["L"] / 2 + 3.0, ESP_ANT["yc"], HZ1]); xb = P._xiao_box(); b_ = np.array([xb[0] - 1.5, xb[2] + 4.0, xb[5] - 3.0])
    res["coax_mm"] = round(path_len([a, np.array([a[0], -24.0, a[2] - 4.0]), b_]) + 15.0, 0)        # FPC 끝 -> 뚜껑 쪽으로 -> XIAO 의 안테나 단자 (+여유 15)
    m3 = v1._m1(mb) + v1._m1(mc)
    rows.append(("모터 3 (판 회전)", "머리 위 모터 자리", bb(m3), "나사 기둥에 얹힘, 둥근 벽까지 %.1f" % (SEAT_R - 1.4 - M_BODY_R - M_OFFSET)))
    xi = HX - HW
    res["head_inner"] = [2 * xi, (HY1 - HWB) - HYL, HZ1 - HZ0]
    res["driver_slot_x"] = [BX1, xi - BOSS_T]

    # ---------- 요크 위 ----------
    m1 = (mb + mc).transform(P.motor1_T()); m2 = (mb + mc).transform(P.motor2_T())
    HEAD = union([headp, P.sector_head(), pi] + drv + keep + [P.usb_plug_dummy(), P.lan_plug_dummy(), m3] + list(P.button_dummy()))
    ROT = P.rotor_set(150.0, low_cap=True)
    g1h = g2h = g1r = g2r = 1e9
    for pitch in list(np.arange(PITCH_MIN, PITCH_MAX + 0.1, 5.0)) + [TILT3]:
        H = P.pose_head(HEAD, 0, pitch)
        g1h = min(g1h, gap(m1, H)); g2h = min(g2h, gap(m2, H))
        for spin in np.arange(0, 360, 15.0):
            R = P.pose_head(ROT, 0, pitch, spin)
            g1r = min(g1r, gap(m1, R)); g2r = min(g2r, gap(m2, R))
    rows.append(("모터 1 (받침 회전)", "회전판 위, 축은 아래로", bb(m1), "머리까지 %.1f, 도는 판까지 %.1f (모든 자세)" % (g1h, g1r)))
    rows.append(("모터 2 (끄덕임)", "오른쪽 팔 바깥면", bb(m2), "머리까지 %.1f, 도는 판까지 %.1f (모든 자세)" % (g2h, g2r)))

    # ---------- 받침 ----------
    for size in ("S", "L"):
        bat = P.battery_dummy(size); tub = P.tub(size); cov = P.cover(size)
        b = BASE[size]
        rows.append(("배터리 (받침 %s)" % size, "받침 통 바닥, 옆 창으로 넣음 (문턱 2개가 앞을 막는다)", [b["bw"], b["bl"], BATT_T],
                     "통까지 %.1f, 덮개까지 %.1f (선이 지나는 자리)" % (gap(bat, tub), gap(bat, cov))))
        ua = P.usb_a_dummy(size); R_ = b["R"]
        out_ = max(0.0, max(np.hypot(x, y) for x in (-14.0, 14.0) for y in (ua.bounding_box()[1],)) - R_)
        rows.append(("ㄱ자 USB 플러그 (받침 %s)" % size, "배터리 끝, 창 안쪽", bb(ua), "받침 바깥면에서 %.1f 안쪽 (밖으로 나온 길이 %.1f)" % (R_ - np.hypot(14.0, ua.bounding_box()[1]), out_)))
        res["battery_%s" % size] = dict(max=[b["bw"], b["bl"], BATT_T], headroom=TUB_H - 2.0)

    # ---------- 전선 길이 (가장 길어지는 자세) ----------
    sx, sy = P.m1_shaft(); ux, uy = M1_BODY_DIR
    m1_cap = np.array([sx + ux * (M_OFFSET + 17.0), sy + uy * (M_OFFSET + 17.0), DECK_Z1 + 8.0])        # 모터 1 전선이 나오는 곳 (상자 안)
    cx_, cz_ = P.m2_cap_xz()
    m2_cap = np.array([cx_, ARM_Y1 + 8.0, cz_])                                                        # 모터 2 전선이 나오는 곳 (팔 뒤 아래 트인 자리 앞)
    x0, x1, z0, z1 = P.M2_SLOT
    via2 = [np.array([cx_, ARM_Y0 - 3.0, cz_]), np.array([(x0 + x1) / 2, CHEEK_Y - 4.0, (z0 + z1) / 2]), np.array([-40.0, 30.0, 60.0])]   # 팔 안쪽 -> 칸막이 구멍 -> 상자 안
    cables = {}
    for name, start, via, k in (("모터 1 → 드라이버 1", m1_cap, [], 2), ("모터 2 → 드라이버 2", m2_cap, via2, 1)):
        worst = 0
        for pitch in np.arange(PITCH_MIN, PITCH_MAX + 0.1, 5.0):
            skirt = head_pt((DRV_X0 + 17.5, -12.0, HZ0), pitch)                                        # 머리 아래 열린 곳
            sock = head_pt((DRV_X0 + 17.5, -6.0, DRV_ZC[k] - 10.0), pitch)                             # 드라이버의 흰 소켓
            worst = max(worst, path_len([start] + via + [skirt, sock]))
        cables[name] = worst
    cables["모터 3 → 드라이버 3"] = path_len([(-25.0, 0.0, 14.0), (-21.0, -15.0, HZ1), (-21.0, -15.0, -4.0), (DRV_X0 + 17.5, -6.0, DRV_ZC[0] - 10.0)])
    worst = 0; lanw = 0
    for pitch in np.arange(PITCH_MIN, PITCH_MAX + 0.1, 5.0):
        worst = max(worst, path_len([(0, 0, TUBE_Z1), head_pt((BX0 - 6.6, -4.2, HZ0), pitch), head_pt((BX0 - 6.6, -4.2, BZ1 - 10.6), pitch)]))
        lanw = max(lanw, path_len([(0, 0, TUBE_Z1), head_pt((BX0 + 10.25, -8.0, BZ0 - 30.0), pitch), head_pt((BX0 + 10.25, -8.0, BZ0 - 2.0), pitch)]))
    # 전원선: 배터리 끝(ㄱ자 플러그) -> 배터리 위 -> 가운데 통 -> 머리.  랜선(받침 안에 들어가는 길이): 뒤 홈 -> 기둥 -> 가운데 통 -> 머리
    usb = {s: path_len([(0.0, -BASE[s]["R"] + BATT_IN - 8.0, 10.0), (20.0, -BASE[s]["R"] + BATT_IN + 6.0, 24.0), (0, 0, 26.0), (0, 0, TUBE_Z1)]) + worst for s in ("S", "L")}
    lan = {s: path_len([(-BASE[s]["R"], 0.0, 25.0), (-50.0, 24.0, 22.0), (0, 0, 26.0), (0, 0, TUBE_Z1)]) + lanw for s in ("S", "L")}
    res["lan_inside"] = {k: round(v) for k, v in lan.items()}
    res["cables"] = {k: round(v) for k, v in cables.items()}
    res["cable_stock"] = CABLE_STOCK
    res["usb_path"] = {k: round(v) for k, v in usb.items()}
    res["tube_bore"] = 2 * TUBE_RI
    res["rows"] = [dict(item=a, where=b, size=c, room=d) for a, b, c, d in rows]

    print("%-22s %-34s %-22s %s" % ("부품", "자리", "크기 (mm)", "여유"))
    for a, b, c, d in rows:
        print("%-22s %-34s %-22s %s" % (a, b, " × ".join("%g" % v for v in c), d))
    print("\n머리 안: 폭 %.0f × 깊이 %.1f × 높이 %.0f.  드라이버 자리 폭 %.0f (보드 35)" % (res["head_inner"][0], res["head_inner"][1], res["head_inner"][2], xi - BOSS_T - BX1))
    print("전선 (가장 길어지는 자세에서 필요한 길이 / 모터에 달린 전선 %d)" % CABLE_STOCK)
    for k, v in cables.items():
        print("   %-20s %4.0f mm  -> 남는 길이 %4.0f" % (k, v, CABLE_STOCK - v))
    print("   전원선 (배터리 -> 파이): 받침 S %.0f mm, L %.0f mm  -> 50cm 선이면 된다. 가운데 통 구멍 지름 %.1f" % (usb["S"], usb["L"], 2 * TUBE_RI))
    print("   랜선이 수신기 안에 들어가는 길이 (뒤 홈 -> 파이): 받침 S %.0f mm, L %.0f mm" % (lan["S"], lan["L"]))
    json.dump(res, open(OUT + "room_check.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
