"""
DeteCam 5.0 (Palm-12) — 손바닥 크기(120 x 120 x 29.6 mm) 벽면 부착·2축 각도 조절 수신기 CAD
(단위: mm, 좌표계: X=벽면 가로, Y=벽면 세로(위=+Y), +Z=방 안쪽(전면 레이돔), -Z=벽면쪽)

핵심 특징:
  - 외형: 120 x 120 x 29.6 mm (손바닥 크기, 노드 총중량 ~215 g)
  - 안테나: 112 x 112 mm 4층 통합 RF PCB (테두리 12소자 Abracon AANI-CH-0078 + 기준 1소자 + PE42442 x4 + AD8317)
  - 벽면 후면 차폐: RF PCB 뒤 11.4 mm 공극 + 110 x 110 mm Ni-Cu 차폐 시트 (콘크리트·타일·거울·철제 사물함 간섭 차단)
  - 전자부: Radxa ROCK 3A (RK3568, 85x56 mm, 58x49 mm 홀, 팬리스) + Intel AX210NGW (Radxa X4 드롭인 호환)
  - 직교 2축 벽면 스위블 짐벌 (GoPro형 직교 힌지 + 손조임 M4 노브 2개, 공구 불필요):
      * 05_wall_base (72x72 mm): Ø15x3 mm 네오디뮴 자석 4개 + 재사용 나노겔 패드면 + 키홀 + Y축 좌우 Pan 귀(±30°)
      * 04_swivel_yoke: 후면 Y축 Pan 혀(±30°) + 전면 X축 Tilt 혀(0°/-15°/-25°), 벽 분리 시 탁상/교탁 스탠드 겸용!
      * 03_rear_cover (104x104 mm 인셋): 방열 슬롯 + X축 상하 Tilt 귀
  - Bambu Lab X1 Carbon 단 1판(256x256 mm)에 노드 1기분 7개 출력물 전체 수납, 서포트 0% (처짐 0.0 mm²)!
"""
import json
import os
import numpy as np
from manifold3d import Manifold as M, CrossSection as CS
from cadlib import box, cz, cy, cx, union, rrect, poly, extrude_z, teardrop_cs

# ── 주요 치수 (mm) ────────────────────────────────────────────
PUCK_W = 120.0
PUCK_R = 10.0
WALL_T = 3.5
INN_W = PUCK_W - 2 * WALL_T   # 113.0 mm (112mm PCB + 0.5mm/측 여유)

# Z 적층 (+Z = 전면 방 안쪽, -Z = 후면 벽쪽)
RADOME_T = 1.4
RF_SHELL_Z0 = 0.0
RF_SHELL_Z1 = 16.0
PCB_Z0 = 11.4
PCB_Z1 = 13.0
REF_Z = 0.0

CHASSIS_Z1 = 0.0
BULKHEAD_T = 2.0
CHASSIS_Z0 = -14.6            # 총 본체 두께 = 29.6 mm!

COVER_W = 104.0               # 인셋 후면 커버 (X1C 1판에 7개 부품 전부 수납!)
COVER_R = 6.0
COVER_T = 2.6
COVER_Z0 = CHASSIS_Z0
COVER_Z1 = CHASSIS_Z0 + COVER_T  # -12.0

CORNER_XY = [(51.0, 51.0), (-51.0, 51.0), (-51.0, -51.0), (51.0, -51.0)]
LID_XY = [(44.5, 44.5), (-44.5, 44.5), (-44.5, -44.5), (44.5, -44.5)]

# Radxa ROCK 3A / X4 공통 폼팩터 (85 x 56 mm, 58 x 49 mm 마운팅 홀)
SBC_W, SBC_H, SBC_T = 85.0, 56.0, 1.6
SBC_HOLES = [(-29.0, -24.5), (29.0, -24.5), (29.0, 24.5), (-29.0, 24.5)]
SBC_STANDOFF_H = 3.0
SBC_Z1 = -BULKHEAD_T - SBC_STANDOFF_H  # -5.0
SBC_Z0 = SBC_Z1 - SBC_T                # -6.6

# 직교 2축 벽면 짐벌 (X축 = 상하 Tilt, Y축 = 좌우 Pan)
TILT_EAR_GAP = 14.8           # 요크 전면 혀 폭(14.0) + 좌우 0.4mm 유격
TILT_EAR_T = 5.0
TILT_PIVOT_Z = COVER_Z0 - 15.0    # -29.6 mm (본체 후면에서 15mm 뒤)

PAN_EAR_GAP = 16.4            # 요크 후면 혀 높이(15.6) + 상하 0.4mm 유격
PAN_EAR_T = 5.0
PAN_PIVOT_Z = TILT_PIVOT_Z - 26.0 # -55.6 mm (Tilt-Pan 축간 26mm)

WALL_PLATE_T = 6.0
WALL_BASE_W = 72.0
WALL_PLATE_Z1 = PAN_PIVOT_Z - 13.0 # -68.6 mm (Pan 귀 높이 13mm)
WALL_Z0 = WALL_PLATE_Z1 - WALL_PLATE_T  # -74.6 mm (벽면 접촉면, 총 벽면 이격 60mm -> ±30°/-25° 동시회전 무간섭!)


def _load_palm12_coords():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", "palm_array_results.json")
    if os.path.exists(p):
        data = json.load(open(p, encoding="utf-8"))
        k = "Palm-12-Edge (114mm 테두리12소자 SMD)"
        if k in data:
            return np.array(data[k]["coords_mm"], dtype=float)
    return np.array([
        [-33.8, 53.0], [-6.3, 53.0], [32.2, 53.0],
        [-30.1, -53.0], [-0.9, -53.0], [29.3, -53.0],
        [-53.0, -35.6], [-53.0, -8.4], [-53.0, 28.0],
        [53.0, -36.7], [53.0, -4.2], [53.0, 36.1],
    ], dtype=float)


ANT_XY = _load_palm12_coords()
REF_ANT_XY = np.array([46.0, 46.0])


# ── 1) 출력 부품 6종 (노드당 7개: 노브 2개 포함, X1C 단 1판, 서포트 0%) ──

def rf_shell():
    """01_rf_shell: 전면 RF 투과 레이돔 + 12mm 안테나 공극 외곽 쉘 (전면 바닥 배치 -> 처짐 0.0)."""
    outer_cs = rrect(-PUCK_W / 2, PUCK_W / 2, -PUCK_W / 2, PUCK_W / 2, PUCK_R)
    inner_cs = rrect(-INN_W / 2, INN_W / 2, -INN_W / 2, INN_W / 2, PUCK_R - WALL_T)

    radome = extrude_z(outer_cs, RF_SHELL_Z1 - RADOME_T, RF_SHELL_Z1)
    # 전면 중앙 34x34mm ArUco 마커 부착 가이드 홈 (깊이 0.25mm < 0.35mm -> 서포트 0)
    aruco_frame = (
        extrude_z(rrect(-19.0, 19.0, -19.0, 19.0, 1.5), RF_SHELL_Z1 - 0.25, RF_SHELL_Z1 + 0.1)
        - extrude_z(rrect(-17.6, 17.6, -17.6, 17.6, 1.0), RF_SHELL_Z1 - 0.25, RF_SHELL_Z1 + 0.1)
    )
    wall = extrude_z(outer_cs - inner_cs, RF_SHELL_Z0, RF_SHELL_Z1 - RADOME_T)

    bosses, holes = [], []
    for bx, by in CORNER_XY:
        bosses.append(cz(4.4, RF_SHELL_Z0, RF_SHELL_Z1 - RADOME_T, bx, by))
        holes.append(cz(1.6, RF_SHELL_Z0 - 0.2, 10.0, bx, by))
    # 레이돔 내면 PCB 0.4mm 이격 스페이서 4개
    for sx, sy in [(38.0, 38.0), (-38.0, 38.0), (-38.0, -38.0), (38.0, -38.0)]:
        bosses.append(cz(3.0, PCB_Z1, RF_SHELL_Z1 - RADOME_T, sx, sy))

    return union([radome, wall] + bosses) - union(holes + [aruco_frame])


def sbc_chassis():
    """02_sbc_chassis: Ni-Cu 차폐 격벽 + ROCK 3A/X4 베이 + 인셋 커버 턱 (격벽 바닥 배치 -> 처짐 0.0)."""
    outer_cs = rrect(-PUCK_W / 2, PUCK_W / 2, -PUCK_W / 2, PUCK_W / 2, PUCK_R)
    bay_w = 98.0
    bay_cs = rrect(-bay_w / 2, bay_w / 2, -bay_w / 2, bay_w / 2, 5.0)
    lid_rec_w = COVER_W + 0.6
    lid_rec_cs = rrect(-lid_rec_w / 2, lid_rec_w / 2, -lid_rec_w / 2, lid_rec_w / 2, COVER_R + 0.3)

    bulkhead = extrude_z(outer_cs, -BULKHEAD_T, CHASSIS_Z1)
    mid_wall = extrude_z(outer_cs - bay_cs, COVER_Z1, -BULKHEAD_T)
    rim_wall = extrude_z(outer_cs - lid_rec_cs, CHASSIS_Z0, COVER_Z1)

    standoffs, cuts = [], []
    for hx, hy in SBC_HOLES:
        standoffs.append(cz(3.2, SBC_Z1, -BULKHEAD_T, hx, hy))
        cuts.append(cz(1.25, SBC_Z1 - 0.2, -0.8, hx, hy))

    # 격벽 중앙 RF 동축 + GPIO 통과 슬롯
    cuts.append(box(-7.0, 7.0, -5.0, 5.0, -BULKHEAD_T - 0.5, CHASSIS_Z1 + 0.5))
    # 4코너 쉘 결합 M3 관통홀 + 후면 카운터보어 (출력 시 위쪽으로 열려 처짐 0)
    for bx, by in CORNER_XY:
        cuts.append(cz(1.75, CHASSIS_Z0 - 0.5, CHASSIS_Z1 + 0.5, bx, by))
        cuts.append(cz(3.1, CHASSIS_Z0 - 0.2, CHASSIS_Z0 + 3.2, bx, by))
    # 인셋 커버 고정 M3 탭홀 4개
    for lx, ly in LID_XY:
        cuts.append(cz(1.5, COVER_Z1 - 0.2, -3.0, lx, ly))
    # 하단(-Y) USB-C 전원 케이블 개방 U-노치 (처짐 0)
    cuts.append(box(-8.0, 8.0, -PUCK_W / 2 - 1.0, -bay_w / 2 + 1.0, CHASSIS_Z0 - 0.5, -5.0))
    # 좌우 측면 대류 방열 개방 U-슬롯 (처짐 0)
    for sx in (-1, 1):
        for vy in (-26.0, -13.0, 0.0, 13.0, 26.0):
            x0 = (bay_w / 2 - 1.0) if sx > 0 else (-PUCK_W / 2 - 1.0)
            x1 = (PUCK_W / 2 + 1.0) if sx > 0 else (-bay_w / 2 + 1.0)
            cuts.append(box(x0, x1, vy - 2.5, vy + 2.5, CHASSIS_Z0 - 0.5, -5.5))

    return union([bulkhead, mid_wall, rim_wall] + standoffs) - union(cuts)


def rear_cover():
    """03_rear_cover: 인셋 후면 커버(104x104 mm) + X축 상하 틸트 귀 2개 (내측 바닥 배치 -> 처짐 0.0)."""
    plate_cs = rrect(-COVER_W / 2, COVER_W / 2, -COVER_W / 2, COVER_W / 2, COVER_R)
    plate = extrude_z(plate_cs, COVER_Z0, COVER_Z1)
    pad = extrude_z(rrect(-20.0, 20.0, -13.0, 13.0, 4.0), COVER_Z0 - 2.0, COVER_Z0)

    # X축 틸트 힌지 귀 (Y-Z 평면 단면 -> X축 돌출)
    # 출력 회전 [180, 0, 0] 시 -Z가 +Z_print(위)이므로 물방울 구멍 꼭지 toward_deg = -90°!
    ear_prof = (
        rrect(-9.5, 9.5, TILT_PIVOT_Z, COVER_Z0, 2.0)
        + CS.circle(9.5).translate([0.0, TILT_PIVOT_Z])
    )
    td = teardrop_cs(2.2, -90.0).translate([0.0, TILT_PIVOT_Z])
    T_yz = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    ear_one = M.extrude(ear_prof - td, TILT_EAR_T).transform(T_yz)
    ears = [
        ear_one.translate([-TILT_EAR_GAP / 2 - TILT_EAR_T, 0.0, 0.0]),
        ear_one.translate([TILT_EAR_GAP / 2, 0.0, 0.0]),
    ]

    cuts = [cz(1.75, COVER_Z0 - 0.5, COVER_Z1 + 0.5, lx, ly) for lx, ly in LID_XY]
    for vy in (-34.0, -24.0, 24.0, 34.0):
        for vx in (-26.0, 0.0, 26.0):
            cuts.append(extrude_z(rrect(vx - 8.5, vx + 8.5, vy - 2.0, vy + 2.0, 1.5), COVER_Z0 - 0.5, COVER_Z1 + 0.5))

    return union([plate, pad] + ears) - union(cuts)


def swivel_yoke():
    """04_swivel_yoke: 직교 2축 스위블 요크 (후면 Y축 Pan 혀 ±30° + 전면 X축 Tilt 혀 0°/-15°/-25°).
    단면이 평탄 바닥(z=PAN_PIVOT_Z-9.0)에서 45° 경사로만 변하므로 [0,0,0] 배치 시 처짐 0.0!"""
    # 후면 Pan 혀: Y 두께 = 15.6mm (PAN_EAR_GAP 16.4mm에 양측 0.4mm 유격), X 폭 = 18.0mm
    # X-Z 평면 단면을 Y 방향으로 압출!
    pan_w_y = 15.6
    # Z 범위: PAN_PIVOT_Z - 9.0 .. PAN_PIVOT_Z + 12.0 (중앙 허브와 연결)
    pan_prof = (
        rrect(-9.0, 9.0, PAN_PIVOT_Z - 9.0, PAN_PIVOT_Z + 12.0, 2.0)
        + CS.circle(9.0).translate([0.0, PAN_PIVOT_Z])
    )
    # 출력 회전 [0, 0, 0] (+Z 위) -> Y축 수평 구멍 물방울 꼭지 toward_deg = +90°!
    td_pan = teardrop_cs(2.2, 90.0).translate([0.0, PAN_PIVOT_Z])
    # T_xz: 2D (u, w) -> 3D (x=u, y=const, z=w)
    T_xz = np.array([[1, 0, 0, 0], [0, 0, 1, -pan_w_y / 2], [0, 1, 0, 0]], dtype=np.float64)
    pan_tab = M.extrude(pan_prof - td_pan, pan_w_y).transform(T_xz)

    # 전면 Tilt 혀: X 두께 = 14.0mm (TILT_EAR_GAP 14.8mm에 양측 0.4mm 유격), Y 폭 = 18.0mm
    # Y-Z 평면 단면을 X 방향으로 압출!
    # 45° 경사 램프로 pan_w_y(15.6) -> tilt_h_y(18.0) 자연 연결 (y 편차 1.2mm -> z 2.0mm 이상 경사면 < 31°!)
    tilt_w_x = 14.0
    z_mid = 0.5 * (PAN_PIVOT_Z + TILT_PIVOT_Z)  # -42.6
    tilt_prof = (
        poly([
            (-7.8, z_mid - 2.0), (7.8, z_mid - 2.0),
            (9.0, z_mid + 1.5), (9.0, TILT_PIVOT_Z),
            (-9.0, TILT_PIVOT_Z), (-9.0, z_mid + 1.5),
        ])
        + CS.circle(9.0).translate([0.0, TILT_PIVOT_Z])
    )
    td_tilt = teardrop_cs(2.2, 90.0).translate([0.0, TILT_PIVOT_Z])
    T_yz = np.array([[0, 0, 1, -tilt_w_x / 2], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    tilt_tab = M.extrude(tilt_prof - td_tilt, tilt_w_x).transform(T_yz)

    # 중앙 브릿지 (X: -7..+7, Y: -7.8..+7.8, Z: PAN_PIVOT_Z+4 .. z_mid+2)
    bridge = box(-7.0, 7.0, -7.8, 7.8, PAN_PIVOT_Z + 4.0, z_mid + 2.0)

    # 출력 바닥면 평탄화: PAN_PIVOT_Z - 8.2 아래를 평평하게 잘라 베드 접지 면적 확보!
    yoke = union([pan_tab, tilt_tab, bridge])
    flat_cut = box(-30.0, 30.0, -30.0, 30.0, PAN_PIVOT_Z - 15.0, PAN_PIVOT_Z - 8.2)
    return yoke - flat_cut


def wall_base():
    """05_wall_base: 벽면 부착판(72x72 mm) — 자석 4개 + 재사용 나노겔 패드면 + 키홀 + Y축 좌우 Pan 귀 2개.
    벽면 접촉면(z=WALL_Z0)을 바닥에 두고 [0,0,0] 출력 -> 처짐 0.0!"""
    w = WALL_BASE_W
    base_cs = rrect(-w / 2, w / 2, -w / 2, w / 2, 8.0)
    plate = extrude_z(base_cs, WALL_Z0, WALL_PLATE_Z1)

    # Y축 좌우 Pan 힌지 귀 2개 (상단/하단, X-Z 평면 단면 -> Y축 압출)
    # 출력 시 +Z 위 -> 물방울 구멍 꼭지 toward_deg = +90°!
    ear_prof = (
        rrect(-9.5, 9.5, WALL_PLATE_Z1 - 1.0, PAN_PIVOT_Z, 2.0)
        + CS.circle(9.5).translate([0.0, PAN_PIVOT_Z])
    )
    td = teardrop_cs(2.2, 90.0).translate([0.0, PAN_PIVOT_Z])
    T_xz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)
    ear_one = M.extrude(ear_prof - td, PAN_EAR_T).transform(T_xz)
    ears = [
        ear_one.translate([0.0, -PAN_EAR_GAP / 2 - PAN_EAR_T, 0.0]),
        ear_one.translate([0.0, PAN_EAR_GAP / 2, 0.0]),
    ]

    cuts = []
    # 4코너 Ø15.4 x 3.4 mm 네오디뮴 자석 포켓 (위에서 파임 -> 벽쪽 0.9mm 얇은 보호막 유지, 처짐 0.0!)
    for mx, my in [(22.5, 22.5), (-22.5, 22.5), (-22.5, -22.5), (22.5, -22.5)]:
        cuts.append(cz(7.75, WALL_Z0 + 0.9, WALL_PLATE_Z1 + 0.5, mx, my))
    # 상단 중앙 키홀 걸이 구멍 (자석 포켓 및 상단 Pan 귀 y=13.2와 간섭 없이 y=24.5..31.5에 배치!)
    cuts.append(cz(3.6, WALL_Z0 - 0.5, WALL_PLATE_Z1 + 0.5, 0.0, 24.5))
    cuts.append(extrude_z(rrect(-1.9, 1.9, 24.5, 31.0, 1.8), WALL_Z0 - 0.5, WALL_PLATE_Z1 + 0.5))

    return union([plate] + ears) - union(cuts)


def thumb_knob():
    """06_thumb_knob: M4 손조임 6엽 노브 (노드당 2개: Pan/Tilt 무공구 잠금, 처짐 0.0)."""
    body_cs = CS.circle(8.0)
    for k in range(6):
        a = np.radians(k * 60.0)
        body_cs = body_cs + CS.circle(3.1).translate([7.6 * np.cos(a), 7.6 * np.sin(a)])
    body = extrude_z(body_cs, 0.0, 9.0)
    hex_r = 7.2 / np.sqrt(3.0)
    hex_pts = [(hex_r * np.cos(np.radians(60 * i)), hex_r * np.sin(np.radians(60 * i))) for i in range(6)]
    nut_pocket = extrude_z(poly(hex_pts), 5.4, 9.5)
    bore = cz(2.2, -0.5, 9.5)
    return body - union([bore, nut_pocket])


# ── 2) 구매 전자·기구 부품 (간섭 검사 및 조립 시각화용) ────────

def hw_rf_pcb():
    p = extrude_z(rrect(-56.0, 56.0, -56.0, 56.0, 5.0), PCB_Z0, PCB_Z1)
    cuts = [cz(5.0, PCB_Z0 - 0.5, PCB_Z1 + 0.5, bx, by) for bx, by in CORNER_XY]
    return p - union(cuts)


def hw_smd_antennas():
    ants = []
    for x, y in ANT_XY:
        if abs(y) > 45.0:
            ants.append(box(x - 3.0, x + 3.0, y - 1.2, y + 1.2, PCB_Z1, PCB_Z1 + 1.1))
        else:
            ants.append(box(x - 1.2, x + 1.2, y - 3.0, y + 3.0, PCB_Z1, PCB_Z1 + 1.1))
    rx, ry = REF_ANT_XY
    ants.append(box(rx - 2.5, rx + 2.5, ry - 1.2, ry + 1.2, PCB_Z1, PCB_Z1 + 1.1))
    return union(ants)


def hw_rf_ics():
    return union([
        box(-2.5, 2.5, -2.5, 2.5, PCB_Z1, PCB_Z1 + 0.9),
        box(-22.0, -17.0, 15.0, 20.0, PCB_Z1, PCB_Z1 + 0.9),
        box(17.0, 22.0, 15.0, 20.0, PCB_Z1, PCB_Z1 + 0.9),
        box(-2.5, 2.5, -22.0, -17.0, PCB_Z1, PCB_Z1 + 0.9),
        box(16.0, 21.0, -22.0, -17.0, PCB_Z1, PCB_Z1 + 0.9),
    ])


def hw_nicu_reflector():
    s = extrude_z(rrect(-55.0, 55.0, -55.0, 55.0, 4.0), REF_Z + 0.05, REF_Z + 0.25)
    cuts = [cz(5.0, REF_Z - 0.5, REF_Z + 1.0, bx, by) for bx, by in CORNER_XY]
    cuts.append(box(-7.5, 7.5, -5.5, 5.5, REF_Z - 0.5, REF_Z + 1.0))
    return s - union(cuts)


def hw_rock3a():
    pcb = box(-SBC_W / 2, SBC_W / 2, -SBC_H / 2, SBC_H / 2, SBC_Z0, SBC_Z1)
    soc = box(-12.0, 12.0, -12.0, 12.0, SBC_Z0 - 3.2, SBC_Z0)
    return union([pcb, soc])


def hw_ax210():
    return box(14.0, 36.0, -15.0, 15.0, SBC_Z0 - 2.2, SBC_Z0 - 0.2)


def hw_magnets():
    return union([cz(7.5, WALL_Z0 + 1.0, WALL_Z0 + 4.0, mx, my)
                  for mx, my in [(22.5, 22.5), (-22.5, 22.5), (-22.5, -22.5), (22.5, -22.5)]])


# ── 3) 직교 2축 운동학 조립체 (벽면 고정, Pan=Y축, Tilt=X축!) ───

def _rot_about(m, axis_deg, pivot):
    p = np.asarray(pivot, dtype=float)
    return m.translate(-p).rotate(axis_deg).translate(p)


def node_assembly(pan_deg=0.0, tilt_deg=0.0):
    """
    벽면(z=WALL_Z0)에 고정된 완전 조립 DeteCam Palm-12 노드.
    - pan_deg  : 좌우 회전 각도 (±30°, Y축 회전)
    - tilt_deg : 상하 숙임 각도 (0° / -15° / -25°, 음수 = 아래쪽 -Y 숙임 -> rx = -tilt_deg)
    """
    pan_piv = [0.0, 0.0, PAN_PIVOT_Z]
    tilt_piv = [0.0, 0.0, TILT_PIVOT_Z]

    # 1) 벽면 고정부
    wb = wall_base()
    mags = hw_magnets()

    # 2) 요크 + 팬 노브: Y축 Pan 회전
    yk = _rot_about(swivel_yoke(), [0.0, pan_deg, 0.0], pan_piv)
    knob_pan = thumb_knob().rotate([-90.0, 0.0, 0.0]).translate([0.0, PAN_EAR_GAP / 2 + PAN_EAR_T + 0.5, PAN_PIVOT_Z])
    knob_pan = _rot_about(knob_pan, [0.0, pan_deg, 0.0], pan_piv)

    # 3) 퍽 본체 + 틸트 노브: 먼저 Tilt 피벗에서 X축 회전 -> 그 다음 Pan 피벗에서 Y축 회전!
    def xform_puck(m):
        m1 = _rot_about(m, [-tilt_deg, 0.0, 0.0], tilt_piv)
        return _rot_about(m1, [0.0, pan_deg, 0.0], pan_piv)

    knob_tilt = thumb_knob().rotate([0.0, 90.0, 0.0]).translate([TILT_EAR_GAP / 2 + TILT_EAR_T + 0.5, 0.0, TILT_PIVOT_Z])

    return {
        "rf_shell":       (xform_puck(rf_shell()),        "#1E3A5F", 1.0),
        "sbc_chassis":    (xform_puck(sbc_chassis()),     "#1B2A57", 1.0),
        "rear_cover":     (xform_puck(rear_cover()),      "#274472", 1.0),
        "swivel_yoke":    (yk,                            "#3A5F8A", 1.0),
        "wall_base":      (wb,                            "#2E4057", 1.0),
        "knob_pan":       (knob_pan,                      "#E07A5F", 1.0),
        "knob_tilt":      (xform_puck(knob_tilt),         "#E07A5F", 1.0),
        "rf_pcb":         (xform_puck(hw_rf_pcb()),       "#1D7A46", 1.0),
        "smd_antennas":   (xform_puck(hw_smd_antennas()), "#F59E0B", 1.0),
        "rf_ics":         (xform_puck(hw_rf_ics()),       "#111827", 1.0),
        "nicu_reflector": (xform_puck(hw_nicu_reflector()), "#CBD5E1", 1.0),
        "rock3a":         (xform_puck(hw_rock3a()),       "#0F766E", 1.0),
        "ax210":          (xform_puck(hw_ax210()),        "#0EA5E9", 1.0),
        "magnets":        (mags,                          "#94A3B8", 1.0),
    }
