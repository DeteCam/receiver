"""DeteCam 5.0 (Palm-12) 기계 간섭·벽면 클리어런스·무게 자동 검사 (Pan ±30° x Tilt 0°/-15°/-25° 전 조합 검증)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import palm_node as P
from cadlib import box
import paths

TOL = 0.5  # mm3
HW_MASS_G = {
    "radxa_rock3a": 55.0,
    "ax210": 4.0,
    "rf_array_switch_pcb_112mm": 28.0,
    "nicu_reflector_sheet": 5.0,
    "neodymium_magnets_x4": 16.0,
    "nano_gel_pad_and_fasteners": 16.0,
}


def check_pose(pan_deg, tilt_deg):
    A = P.node_assembly(pan_deg=pan_deg, tilt_deg=tilt_deg)
    # 가상 벽면(z <= WALL_Z0 - 0.1) 추가: 회전 시 퍽 모서리가 벽에 닿는지 엄격 검사!
    wall_slab = box(-300.0, 300.0, -300.0, 300.0, P.WALL_Z0 - 20.0, P.WALL_Z0 - 0.1)
    names = sorted(A.keys())
    overlaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = float((A[a][0] ^ A[b][0]).volume())
            if v > TOL:
                overlaps.append([a, b, round(v, 2)])
    # 벽면 침범 검사
    wall_hits = []
    for n in names:
        v = float((A[n][0] ^ wall_slab).volume())
        if v > TOL:
            wall_hits.append([n, round(v, 2)])
    bbs = [A[n][0].bounding_box() for n in names]
    x0 = min(b[0] for b in bbs); x1 = max(b[3] for b in bbs)
    y0 = min(b[1] for b in bbs); y1 = max(b[4] for b in bbs)
    z0 = min(b[2] for b in bbs); z1 = max(b[5] for b in bbs)
    return dict(
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        bbox_mm=[round(x1 - x0, 1), round(y1 - y0, 1), round(z1 - z0, 1)],
        overlaps=overlaps,
        wall_hits=wall_hits,
    )


def main():
    # 핵심 극한 각도 포함 9개 조합 전수 검사
    poses = [
        (0, 0), (30, 0), (-30, 0),
        (0, -15), (30, -15), (-30, -15),
        (0, -25), (30, -25), (-30, -25),
    ]
    reports = [check_pose(p, t) for p, t in poses]

    summ_p = os.path.join(paths.SLICE, "summary.json")
    parts_p = os.path.join(paths.RES, "parts.json")
    printed_g = None
    if os.path.exists(summ_p):
        s = json.load(open(summ_p, encoding="utf-8"))
        if s.get("plates"):
            printed_g = round(sum(pl["g"] for pl in s["plates"].values() if pl.get("ok")), 1)
    if printed_g is None and os.path.exists(parts_p):
        rows = json.load(open(parts_p, encoding="utf-8"))
        printed_g = round(sum(r["volume_cm3"] * r["qty_per_node"] * 0.62 for r in rows), 1)
    hw_g = sum(HW_MASS_G.values())
    total_g = round((printed_g or 100.0) + hw_g, 1)

    problems = []
    for r in reports:
        if r["overlaps"]:
            problems.append(f"자세(pan={r['pan_deg']}, tilt={r['tilt_deg']}) 부품 겹침: {r['overlaps']}")
        if r["wall_hits"]:
            problems.append(f"자세(pan={r['pan_deg']}, tilt={r['tilt_deg']}) 벽면 충돌: {r['wall_hits']}")
    if total_g > 300.0:
        problems.append(f"노드 무게 초과: {total_g} g > 300 g")

    res = dict(
        puck_outer_mm=[P.PUCK_W, P.PUCK_W, round(P.RF_SHELL_Z1 - P.CHASSIS_Z0, 1)],
        wall_standoff_mm=round(P.CHASSIS_Z0 - P.WALL_Z0, 1),
        rf_cavity_mm=round(P.PCB_Z0 - (P.REF_Z + 0.25), 2),
        poses=reports,
        mass_g=dict(printed_pla=printed_g, hardware=hw_g, total_node=total_g, breakdown=HW_MASS_G),
        problems=problems,
    )
    os.makedirs(paths.RES, exist_ok=True)
    with open(os.path.join(paths.RES, "mech_check.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"Puck: {res['puck_outer_mm']} mm | Wall standoff: {res['wall_standoff_mm']} mm | RF cavity: {res['rf_cavity_mm']} mm | Mass: {total_g} g")
    print(f"Tested {len(reports)} Pan/Tilt poses -> problems: {len(problems)}")
    if problems:
        for pr in problems:
            print("  FAIL:", pr)
    sys.exit(len(problems))


if __name__ == "__main__":
    main()
