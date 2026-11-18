"""패널 수신기의 기계 검사: 부품끼리 겹치지 않는지, 틈이 얼마인지, 세웠을 때 넘어지지 않는지, 자석이 버티는지.

    python3 checks/panel_check.py        ->  results/panel_check.json

겹침은 형상의 교집합 부피로 본다 (0.5 mm3 넘으면 겹친 것으로 본다). 무게는 print/summary.json (슬라이싱 결과) 과 사 오는 부품의 어림값이다.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
import panel as P
import paths

TOL = 0.5
HW_G = dict(reflector=53.0, films=17.0, rf=96.0, xiao=4.0, cell=55.0, magnets=28.0, wires=190.0, screws=12.0, motor=37.0, bearing=60.0)   # 사 오는 부품의 무게 어림 (g): 반사판 Al 140x140x1, 모듈 8 x 12, 선 = 변환 단자 34 + SMA 선 12 + 위상선


def vol(a, b):
    return float((a ^ b).volume())


def bbox(m):
    b = m.bounding_box(); return [round(float(v), 2) for v in b]


def main():
    out = dict(overlaps=[], gaps={}, sizes={}, problems=[])
    pa = P.panel_assembly("plate"); pf = P.panel_assembly("foot"); hub = P.hub_assembly()
    pairs = [(pa, a, b) for a in pa for b in pa if a < b] + [(pf, "foot", b) for b in pf if b != "foot"] + [(hub, a, b) for a in hub for b in hub if a < b]
    allow = {("films", "ant_plate"), ("ant_plate", "reflector"), ("ant_plate", "films"), ("ref", "ant_plate")}   # 필름은 받침판 홈에 붙고, 받침판 다리는 반사판에 선다
    for A, a, b in pairs:
        v = vol(A[a][0], A[b][0])
        if v > TOL and (a, b) not in allow and (b, a) not in allow:
            out["overlaps"].append([a, b, round(v, 1)])
    if out["overlaps"]:
        out["problems"].append("겹침 %d쌍" % len(out["overlaps"]))
    # 틈
    out["gaps"] = dict(plate_side=P.MX, antenna_to_cover=P.AIR, antenna_to_reflector=P.RG, behind_reflector=P.CAV,
                       xiao_side=0.25, cell_fence=0.3, usb_window=[P.USB_C[0], P.USB_C[1]], foot_to_tray=P.FT_GAP,
                       battery_to_deck=round(P.DECK_Z - (P.HFLOOR + P.BATT[2] - 1.0), 2), pi_to_lid=round(P.HFLOOR + P.HH - 3.0 - (P.DECK_Z + P.DECK_T + P.PI_STAND + P.PI[2] + 16.0), 2))
    if out["gaps"]["pi_to_lid"] < 1.0:
        out["problems"].append("파이 위가 뚜껑에 닿는다")
    # 크기
    for nm, fn in (("tray", P.tray), ("bezel", P.bezel), ("mount_plate", P.mount_plate), ("foot", P.foot), ("hub_box", P.hub_box), ("hub_lid", P.hub_lid)):
        b = fn().bounding_box(); out["sizes"][nm] = [round(b[3] - b[0], 1), round(b[4] - b[1], 1), round(b[5] - b[2], 1)]
    out["sizes"]["panel_with_plate"] = [round(2 * P.OX, 1), round(P.OY1 - P.OY0, 1), round(P.ZC - P.ZO + P.MT_T, 1)]
    out["sizes"]["hub"] = [round(P.HL + 2 * P.HWALL, 1), round(P.HW_ + 2 * P.HWALL, 1), round(P.HFLOOR + P.HH + 2.4, 1)]
    # 무게와 무게 중심 (패널 좌표, 단위 mm). 뽑는 부품은 슬라이싱한 무게, 무게 중심은 형상의 중심으로 본다
    sp = os.path.join(paths.SLICE, "summary.json"); pr = json.load(open(sp, encoding="utf-8"))["parts"] if os.path.exists(sp) else {}
    g = lambda k, d: pr.get(k, {}).get("g", d)
    def cen(m):
        b = m.bounding_box(); return np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])
    items = [(g("01_panel_tray", 174.0), cen(pa["tray"][0])), (g("02_panel_bezel", 69.0), cen(pa["bezel"][0])), (g("09_ant_plate", 44.0) + HW_G["films"], cen(pa["ant_plate"][0])), (HW_G["reflector"], cen(pa["reflector"][0])), (HW_G["rf"], cen(pa["rf"][0])),
             (HW_G["xiao"], cen(pa["xiao"][0])), (HW_G["cell"], cen(pa["cell"][0])), (HW_G["wires"] + HW_G["screws"], np.array([0, 0, -14.0]))]
    w0 = sum(w for w, _ in items); c0 = sum(w * c for w, c in items) / w0
    wp = w0 + g("03_mount_plate", 41.0) + HW_G["magnets"]; wf = w0 + g("04_foot", 66.0)
    out["mass_g"] = dict(panel_with_plate=round(wp), panel_with_foot=round(wf), printed_panel=round(g("01_panel_tray", 174.0) + g("02_panel_bezel", 69.0) + g("09_ant_plate", 44.0) + g("03_mount_plate", 41.0) + g("05_xiao_clip", 0.8)))
    out["centre_mm"] = [round(float(v), 1) for v in c0]
    # 받침 발 위에서: 발 밑면 y, 앞끝 z, 뒤끝 z
    yb = P.OY0 - P.FT_GAP - P.FT_T; zf, zb = P.FT_FRONT, P.ZO - P.FT_T - P.FT_D; cf = (w0 * c0 + g("04_foot", 66.0) * cen(pf["foot"][0])) / wf; h = cf[1] - yb
    out["tip_deg"] = dict(forward=round(float(np.degrees(np.arctan2(zf - cf[2], h))), 1), backward=round(float(np.degrees(np.arctan2(cf[2] - zb, h))), 1),
                          sideways=round(float(np.degrees(np.arctan2(P.FT_W / 2, h))), 1))
    if min(out["tip_deg"].values()) < 10:
        out["problems"].append("받침 발 위에서 10도 안에 넘어진다")
    # 회전 받침: 겹침, 패널을 얹은 무게 중심이 받침대 원 안에 드는가, 몇 도까지 기울여도 넘어지지 않는가
    T = P.turntable_assembly(); place = P.panel_on_turntable()
    tn = ("turn_base", "turn_top", "turn_hub", "motor", "bearing")
    ok_t = {("turn_top", "bearing"), ("turn_base", "bearing"), ("turn_base", "motor"), ("turn_top", "foot"), ("turn_top", "turn_hub"), ("turn_hub", "motor")}   # 맞닿는 면 (발 밑면 = 기운 면), 둥글게 그린 축이 D 구멍의 평면에 걸치는 것
    tov = []
    for a in tn:
        for b in list(T):
            if a < b or b not in tn:
                if a == b: continue
                v = vol(T[a][0], T[b][0])
                if v > TOL and (a, b) not in ok_t and (b, a) not in ok_t: tov.append([a, b, round(v, 1)])
    out["turntable_overlaps"] = tov
    if tov: out["problems"].append("회전 받침 겹침 %d쌍" % len(tov))
    def tcen(m): return cen(place(m))
    pts = [(w, tcen(m)) for w, m in ((g("01_panel_tray", 174.0), pf["tray"][0]), (g("02_panel_bezel", 69.0), pf["bezel"][0]), (g("09_ant_plate", 44.0) + HW_G["films"], pf["ant_plate"][0]), (HW_G["reflector"], pf["reflector"][0]), (HW_G["rf"], pf["rf"][0]), (HW_G["xiao"], pf["xiao"][0]), (HW_G["cell"], pf["cell"][0]), (HW_G["wires"] + HW_G["screws"], pf["rf"][0]), (g("04_foot", 66.0), pf["foot"][0]))]
    pts += [(g("11_turn_top", 84.0), cen(T["turn_top"][0])), (g("10_turn_base", 97.0), cen(T["turn_base"][0])), (g("12_turn_hub", 3.0), cen(T["turn_hub"][0])), (HW_G["motor"], cen(T["motor"][0])), (HW_G["bearing"], cen(T["bearing"][0]))]
    W = sum(w for w, _ in pts); C = sum(w * c for w, c in pts) / W; zb = min(T[k][0].bounding_box()[2] for k in T)
    off = float(np.hypot(C[0], C[1])); hgt = float(C[2] - zb)
    out["turntable"] = dict(mass_g=round(W), centre_mm=[round(float(v), 1) for v in C], offset_from_axis_mm=round(off, 1), base_radius_mm=P.TB_R,
                            tip_deg_worst=round(float(np.degrees(np.arctan2(P.TB_R - off, hgt))), 1))
    if off > P.TB_R * 0.7: out["problems"].append("회전 받침: 무게 중심이 받침대 끝에 가깝다")
    # 자석: 접시 구멍 네오디뮴 지름 20 x 3 한 개가 두꺼운 철판을 당기는 힘을 3 kgf 로 본다 (판매처 표기). 칠판은 철판이 얇아 절반, 마찰 0.2~0.3
    pull = 3.0; n = len(P.MAGS); hold = [n * pull * 0.5 * mu for mu in (0.2, 0.3)]
    out["magnet"] = dict(count=n, pull_each_kgf=pull, thin_steel_factor=0.5, friction=[0.2, 0.3], slide_hold_kg=[round(v, 2) for v in hold], panel_kg=round(wp / 1000, 2),
                         margin=[round(v / (wp / 1000), 1) for v in hold], note="당기는 힘과 마찰은 어림값이다. 실물에서 잰다")
    if hold[0] / (wp / 1000) < 1.5:
        out["problems"].append("자석 여유가 1.5배가 안 된다")
    json.dump(out, open(os.path.join(paths.RES, "panel_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("겹침 %d쌍 %s" % (len(out["overlaps"]), out["overlaps"][:6])); print("크기", out["sizes"]["panel_with_plate"], out["sizes"]["hub"]); print("무게", out["mass_g"], "무게 중심", out["centre_mm"])
    print("넘어지는 각", out["tip_deg"]); print("회전 받침", out["turntable"], out["turntable_overlaps"]); print("자석", out["magnet"]["slide_hold_kg"], "kg, 여유", out["magnet"]["margin"], "배"); print("문제 %d개 %s" % (len(out["problems"]), out["problems"]))


if __name__ == "__main__":
    main()
