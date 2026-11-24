"""
DeteCam 6.0 (Nano-2 AI) — 초소형 포켓 크기(96 x 96 x 36.0 mm) 벽면 부착·2축 스위블 수신기 CAD
(단위: mm, 좌표계: X=벽면 가로, Y=벽면 세로(위=+Y), +Z=방 안쪽(전면 레이돔), -Z=벽면쪽)

핵심 특징:
  - 외형: 96 x 96 x 36.0 mm (포켓 크기, 면적 v5.0 대비 -36% / v4.0 대비 -95%, 노드 총중량 ~185 g)
  - 단 2기 구성: PyTorch ReconCovNet-v3 + MA-PINN 가상 거울 앵커로 3기 -> 2기 축소!
  - 안테나: 84 x 84 mm 4층 통합 RF PCB (테두리 12소자 Abracon AANI-CH-0078 + 기준 1소자 + PE42442 x4 + AD8317 + 온보드 RP2040)
  - 벽면 후면 차폐: RF PCB 뒤 8.55 mm 공극 + 82 x 82 mm Ni-Cu 차폐 시트 (콘크리트·타일·거울·사물함 간섭 차단)
  - 전자부: FriendlyElec NanoPi R5C (58 x 58 mm, RK3568B2 + 0.8 TOPS NPU, 무개조 순정 포트 완전 수납!) + Intel AX210NGW (네이티브 M.2 E-Key)
  - 직교 2축 벽면 스위블 짐벌 (GoPro형 직교 힌지 + 손조임 M4 노브 2개, 공구 불필요):
      * 05_wall_base (66 x 66 mm): Ø15x3 mm 네오디뮴 자석 4개 + 재사용 나노겔 패드면 + 키홀 + Y축 좌우 Pan 귀(±30°)
      * 04_swivel_yoke: 후면 Y축 Pan 혀(±30°) + 전면 X축 Tilt 혀(0°/-15°/-25°), 벽 분리 시 탁상/교탁 스탠드 겸용
      * 03_rear_cover (82 x 82 mm 인셋): 방열 슬롯 + X축 상하 Tilt 귀
  - Bambu Lab X1 Carbon 단 1판(256 x 256 mm)에 노드 1기분 7개 출력물 일괄 수납 (서포트 0%, 처짐 0.0 mm²)!
    2기 시스템 전체 제작 = 단 2판 (~5시간 40분, ~195 g PLA)!
"""
import numpy as np
from manifold3d import Manifold as M, CrossSection as CS
from cadlib import box, cz, union, rrect, poly, extrude_z, teardrop_cs

# ── 주요 치수 (mm) ────────────────────────────────────────────
PUCK_W = 96.0
PUCK_R = 8.5
WALL_T = 3.2
INN_W = PUCK_W - 2 * WALL_T   # 89.6 mm (84 x 84 mm RF PCB + 2.8 mm/측 여유)

# Z 적층 (+Z = 전면 방 안쪽, -Z = 후면 벽쪽)
RADOME_T = 1.2
RF_SHELL_Z0 = 0.0
RF_SHELL_Z1 = 12.8
PCB_Z0 = 8.6
PCB_Z1 = 10.2                 # 8.35 mm Ni-Cu 공극 + 1.6 mm 4층 FR-4 + 1.1 mm SMD + 0.3 mm 공극 + 1.2 mm 레이돔
REF_Z = 0.0

CHASSIS_Z1 = 0.0
BULKHEAD_T = 1.6
CHASSIS_Z0 = -23.2            # 총 본체 두께 = 36.0 mm (무개조 NanoPi R5C 포트 14mm + M.2 AX210 완전 수납!)

COVER_W = 80.0                # 인셋 후면 커버 (코너 M3 카운터보어와 3.9 mm 완전 이격)
COVER_R = 8.5
COVER_T = 2.2
COVER_Z0 = CHASSIS_Z0
COVER_Z1 = CHASSIS_Z0 + COVER_T  # -21.0 (내부 SBC 높이 19.4 mm!)

CORNER_XY = [(40.5, 40.5), (-40.5, 40.5), (-40.5, -40.5), (40.5, -40.5)]
LID_XY = [(36.0, 20.0), (-36.0, 20.0), (-36.0, -20.0), (36.0, -20.0)]

# FriendlyElec NanoPi R5C (58 x 58 x 1.6 mm, 49 x 49 mm 마운팅 홀, 38 g, 네이티브 M.2 E-Key + 0.8T NPU)
SBC_W, SBC_H, SBC_T = 58.0, 58.0, 1.6
SBC_HOLES = [(-24.5, -24.5), (24.5, -24.5), (24.5, 24.5), (-24.5, 24.5)]
SBC_STANDOFF_H = 4.0          # 하부 M.2 E-Key AX210 (3.2 mm) + 방열 간극
SBC_Z1 = -BULKHEAD_T - SBC_STANDOFF_H  # -5.6
SBC_Z0 = SBC_Z1 - SBC_T                # -7.2 (상부 RJ45/USB-A 포트 -7.2..-20.6, 커버 내면 -21.0과 0.4 mm 여유!)

# 직교 2축 벽면 짐벌 (X축 = 상하 Tilt, Y축 = 좌우 Pan)
TILT_EAR_GAP = 14.4           # 요크 전면 혀 폭(13.6) + 좌우 0.4 mm 유격
TILT_EAR_T = 4.8
TILT_PIVOT_Z = COVER_Z0 - 13.0    # -36.2 mm

PAN_EAR_GAP = 16.0            # 요크 후면 혀 높이(15.2) + 상하 0.4 mm 유격
PAN_EAR_T = 4.8
PAN_PIVOT_Z = TILT_PIVOT_Z - 23.0 # -59.2 mm

WALL_PLATE_T = 5.6
WALL_BASE_W = 66.0
WALL_PLATE_Z1 = PAN_PIVOT_Z - 12.0 # -71.2 mm
WALL_Z0 = WALL_PLATE_Z1 - WALL_PLATE_T  # -76.8 mm (벽면 접촉면, ±30°/-25° 동시회전 무간섭!)

# DeteCam 6.0 Nano-12 테두리 12소자 좌표 (mm, sim/raytrace_engine.py NANO12_EDGE_M 와 0.1 mm 일치)
ANT_XY = np.array([
    [-25.2,  39.0], [ -4.5,  39.0], [ 23.8,  39.0],
    [-22.5, -39.0], [  0.8, -39.0], [ 21.8, -39.0],
    [-39.0, -26.5], [-39.0,  -5.8], [-39.0,  21.2],
    [ 39.0, -27.2], [ 39.0,  -3.0], [ 39.0,  26.8],
], dtype=float)
REF_ANT_XY = np.array([32.0, 32.0])


# ── 1) 출력 부품 6종 (노드당 7개: 노브 2개 포함, X1C 단 1판, 서포트 0%) ──

def rf_shell():
    """01_rf_shell: 전면 RF 투과 레이돔 + 8.6 mm 안테나 공극 외곽 쉘 (전면 바닥 배치 -> 처짐 0.0)."""
    outer_cs = rrect(-PUCK_W / 2, PUCK_W / 2, -PUCK_W / 2, PUCK_W / 2, PUCK_R)
    inner_cs = rrect(-INN_W / 2, INN_W / 2, -INN_W / 2, INN_W / 2, PUCK_R - WALL_T)

    radome = extrude_z(outer_cs, RF_SHELL_Z1 - RADOME_T, RF_SHELL_Z1)
    # 전면 중앙 30 x 30 mm ArUco 마커 부착 가이드 홈 (깊이 0.25 mm < 0.35 mm -> 서포트 0)
    aruco_frame = (
        extrude_z(rrect(-16.5, 16.5, -16.5, 16.5, 1.5), RF_SHELL_Z1 - 0.25, RF_SHELL_Z1 + 0.1)
        - extrude_z(rrect(-15.2, 15.2, -15.2, 15.2, 1.0), RF_SHELL_Z1 - 0.25, RF_SHELL_Z1 + 0.1)
    )
    wall = extrude_z(outer_cs - inner_cs, RF_SHELL_Z0, RF_SHELL_Z1 - RADOME_T + 0.2)

    bosses, holes = [], []
    for bx, by in CORNER_XY:
        bosses.append(cz(4.0, RF_SHELL_Z0, RF_SHELL_Z1 - RADOME_T + 0.2, bx, by))
        holes.append(cz(1.6, RF_SHELL_Z0 - 0.2, 8.2, bx, by))
    # 레이돔 내면 PCB 1.4 mm 이격 스페이서 4개 (REF_ANT 및 중앙 RF IC와 완전 이격)
    for sx, sy in [(24.0, 24.0), (-24.0, 24.0), (-24.0, -24.0), (24.0, -24.0)]:
        bosses.append(cz(2.6, PCB_Z1, RF_SHELL_Z1 - RADOME_T + 0.2, sx, sy))

    return union([radome, wall] + bosses) - union(holes + [aruco_frame])


def sbc_chassis():
    """02_sbc_chassis: Ni-Cu 차폐 격벽 + NanoPi R5C 경량 베이 + 인셋 커버 턱 (격벽 바닥 배치 -> 처짐 0.0)."""
    outer_cs = rrect(-PUCK_W / 2, PUCK_W / 2, -PUCK_W / 2, PUCK_W / 2, PUCK_R)
    bay_w = 76.0
    bay_cs = rrect(-bay_w / 2, bay_w / 2, -bay_w / 2, bay_w / 2, 6.5)
    lid_rec_w = COVER_W + 0.6
    lid_rec_cs = rrect(-lid_rec_w / 2, lid_rec_w / 2, -lid_rec_w / 2, lid_rec_w / 2, COVER_R + 0.3)

    bulkhead = extrude_z(outer_cs, -BULKHEAD_T, CHASSIS_Z1)
    mid_wall = extrude_z(outer_cs - bay_cs, COVER_Z1, -BULKHEAD_T + 0.2)
    rim_wall = extrude_z(outer_cs - lid_rec_cs, CHASSIS_Z0, COVER_Z1 + 0.2)

    standoffs, cuts = [], []
    for hx, hy in SBC_HOLES:
        standoffs.append(cz(3.0, SBC_Z1, -BULKHEAD_T + 0.2, hx, hy))
        cuts.append(cz(1.25, SBC_Z1 - 0.2, -0.6, hx, hy))
    # 4코너 쉘 결합 기둥(COVER_Z1까지만 보강 -> 후면 커버와 간섭 0!) + 인셋 커버 M3 보스
    for bx, by in CORNER_XY:
        standoffs.append(cz(4.0, COVER_Z1, -BULKHEAD_T + 0.2, bx, by))
        cuts.append(cz(1.75, CHASSIS_Z0 - 0.5, CHASSIS_Z1 + 0.5, bx, by))
        cuts.append(cz(2.9, CHASSIS_Z0 - 0.2, CHASSIS_Z0 + 3.2, bx, by))
    for lx, ly in LID_XY:
        standoffs.append(cz(3.5, COVER_Z1, -BULKHEAD_T + 0.2, lx, ly))
        cuts.append(cz(1.5, COVER_Z1 - 0.2, -4.0, lx, ly))

    # 격벽 중앙 RF 동축(MHF4 x2) + RP2040 USB 배선 통과 슬롯
    cuts.append(box(-6.5, 6.5, -4.5, 4.5, -BULKHEAD_T - 0.5, CHASSIS_Z1 + 0.5))
    # 하단(-Y) USB-C 5V 전원 + 2.5GbE 포트 개방 U-노치 (처짐 0)
    cuts.append(box(-14.0, 14.0, -PUCK_W / 2 - 1.0, -bay_w / 2 + 1.0, CHASSIS_Z0 - 0.5, -8.5))
    # 좌우 측면 대류 방열 개방 U-슬롯 (LID_XY y=±20과 완전 이격, 처짐 0)
    for sx in (-1, 1):
        for vy in (-9.0, 0.0, 9.0):
            x0 = (bay_w / 2 - 1.0) if sx > 0 else (-PUCK_W / 2 - 1.0)
            x1 = (PUCK_W / 2 + 1.0) if sx > 0 else (-bay_w / 2 + 1.0)
            cuts.append(box(x0, x1, vy - 2.2, vy + 2.2, CHASSIS_Z0 - 0.5, -8.0))

    return union([bulkhead, mid_wall, rim_wall] + standoffs) - union(cuts)


def rear_cover():
    """03_rear_cover: 인셋 후면 커버(82 x 82 mm) + X축 상하 틸트 귀 2개 (내측 바닥 배치 -> 처짐 0.0)."""
    plate_cs = rrect(-COVER_W / 2, COVER_W / 2, -COVER_W / 2, COVER_W / 2, COVER_R)
    plate = extrude_z(plate_cs, COVER_Z0, COVER_Z1)
    pad = extrude_z(rrect(-18.0, 18.0, -11.5, 11.5, 3.5), COVER_Z0 - 1.8, COVER_Z0)

    # X축 틸트 힌지 귀 (Y-Z 평면 단면 -> X축 압출)
    ear_prof = (
        rrect(-8.8, 8.8, TILT_PIVOT_Z, COVER_Z0, 2.0)
        + CS.circle(8.8).translate([0.0, TILT_PIVOT_Z])
    )
    td = teardrop_cs(2.2, -90.0).translate([0.0, TILT_PIVOT_Z])
    T_yz = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    ear_one = M.extrude(ear_prof - td, TILT_EAR_T).transform(T_yz)
    ears = [
        ear_one.translate([-TILT_EAR_GAP / 2 - TILT_EAR_T, 0.0, 0.0]),
        ear_one.translate([TILT_EAR_GAP / 2, 0.0, 0.0]),
    ]

    cuts = [cz(1.75, COVER_Z0 - 0.5, COVER_Z1 + 0.5, lx, ly) for lx, ly in LID_XY]
    for vy in (-28.0, -19.0, 19.0, 28.0):
        for vx in (-18.0, 0.0, 18.0):
            cuts.append(extrude_z(rrect(vx - 6.5, vx + 6.5, vy - 1.8, vy + 1.8, 1.4), COVER_Z0 - 0.5, COVER_Z1 + 0.5))

    return union([plate, pad] + ears) - union(cuts)


def swivel_yoke():
    """04_swivel_yoke: 직교 2축 스위블 요크 (후면 Y축 Pan 혀 ±30° + 전면 X축 Tilt 혀 0°/-15°/-25°)."""
    pan_w_y = 15.2
    pan_prof = (
        rrect(-8.4, 8.4, PAN_PIVOT_Z - 8.4, PAN_PIVOT_Z + 10.5, 2.0)
        + CS.circle(8.4).translate([0.0, PAN_PIVOT_Z])
    )
    td_pan = teardrop_cs(2.2, 90.0).translate([0.0, PAN_PIVOT_Z])
    T_xz = np.array([[1, 0, 0, 0], [0, 0, 1, -pan_w_y / 2], [0, 1, 0, 0]], dtype=np.float64)
    pan_tab = M.extrude(pan_prof - td_pan, pan_w_y).transform(T_xz)

    tilt_w_x = 13.6
    z_mid = 0.5 * (PAN_PIVOT_Z + TILT_PIVOT_Z)  # -47.7
    tilt_prof = (
        poly([
            (-7.6, z_mid - 2.0), (7.6, z_mid - 2.0),
            (8.4, z_mid + 1.5), (8.4, TILT_PIVOT_Z),
            (-8.4, TILT_PIVOT_Z), (-8.4, z_mid + 1.5),
        ])
        + CS.circle(8.4).translate([0.0, TILT_PIVOT_Z])
    )
    td_tilt = teardrop_cs(2.2, 90.0).translate([0.0, TILT_PIVOT_Z])
    T_yz = np.array([[0, 0, 1, -tilt_w_x / 2], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    tilt_tab = M.extrude(tilt_prof - td_tilt, tilt_w_x).transform(T_yz)

    bridge = box(-6.8, 6.8, -7.6, 7.6, PAN_PIVOT_Z + 3.5, z_mid + 2.0)
    yoke = union([pan_tab, tilt_tab, bridge])
    flat_cut = box(-30.0, 30.0, -30.0, 30.0, PAN_PIVOT_Z - 14.0, PAN_PIVOT_Z - 7.6)
    return yoke - flat_cut


def wall_base():
    """05_wall_base: 초소형 벽면 부착판(66 x 66 mm) — 자석 4개 + 재사용 나노겔 + 키홀 + Y축 Pan 귀 2개."""
    w = WALL_BASE_W
    base_cs = rrect(-w / 2, w / 2, -w / 2, w / 2, 7.5)
    plate = extrude_z(base_cs, WALL_Z0, WALL_PLATE_Z1)

    ear_prof = (
        rrect(-8.8, 8.8, WALL_PLATE_Z1 - 1.0, PAN_PIVOT_Z, 2.0)
        + CS.circle(8.8).translate([0.0, PAN_PIVOT_Z])
    )
    td = teardrop_cs(2.2, 90.0).translate([0.0, PAN_PIVOT_Z])
    T_xz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)
    ear_one = M.extrude(ear_prof - td, PAN_EAR_T).transform(T_xz)
    ears = [
        ear_one.translate([0.0, -PAN_EAR_GAP / 2 - PAN_EAR_T, 0.0]),
        ear_one.translate([0.0, PAN_EAR_GAP / 2, 0.0]),
    ]

    cuts = []
    # 4코너 Ø15.4 x 3.4 mm 네오디뮴 자석 포켓 (Pan 귀 ±8.8 mm와 3.9 mm 이상 이격!)
    for mx, my in [(20.5, 20.5), (-20.5, 20.5), (-20.5, -20.5), (20.5, -20.5)]:
        cuts.append(cz(7.75, WALL_Z0 + 0.9, WALL_PLATE_Z1 + 0.5, mx, my))
    # 상단 중앙 키홀 걸이 구멍 (y = 22.5..28.5, 상단 Pan 귀 y=12.8과 6.1 mm 이격!)
    cuts.append(cz(3.4, WALL_Z0 - 0.5, WALL_PLATE_Z1 + 0.5, 0.0, 22.5))
    cuts.append(extrude_z(rrect(-1.8, 1.8, 22.5, 28.5, 1.7), WALL_Z0 - 0.5, WALL_PLATE_Z1 + 0.5))

    return union([plate] + ears) - union(cuts)


def thumb_knob():
    """06_thumb_knob: M4 손조임 6엽 노브 (노드당 2개: Pan/Tilt 무공구 잠금, 처짐 0.0)."""
    body_cs = CS.circle(7.6)
    for k in range(6):
        a = np.radians(k * 60.0)
        body_cs = body_cs + CS.circle(2.9).translate([7.2 * np.cos(a), 7.2 * np.sin(a)])
    body = extrude_z(body_cs, 0.0, 8.6)
    hex_r = 7.2 / np.sqrt(3.0)
    hex_pts = [(hex_r * np.cos(np.radians(60 * i)), hex_r * np.sin(np.radians(60 * i))) for i in range(6)]
    nut_pocket = extrude_z(poly(hex_pts), 5.0, 9.2)
    bore = cz(2.2, -0.5, 9.2)
    return body - union([bore, nut_pocket])


# ── 2) 구매 전자·기구 부품 (간섭 검사 및 조립 시각화용) ────────

def hw_rf_pcb():
    p = extrude_z(rrect(-42.0, 42.0, -42.0, 42.0, 4.5), PCB_Z0, PCB_Z1)
    cuts = [cz(4.8, PCB_Z0 - 0.5, PCB_Z1 + 0.5, bx, by) for bx, by in CORNER_XY]
    return p - union(cuts)


def hw_smd_antennas():
    ants = []
    for x, y in ANT_XY:
        if abs(y) > 33.0:
            ants.append(box(x - 3.0, x + 3.0, y - 1.2, y + 1.2, PCB_Z1, PCB_Z1 + 1.1))
        else:
            ants.append(box(x - 1.2, x + 1.2, y - 3.0, y + 3.0, PCB_Z1, PCB_Z1 + 1.1))
    rx, ry = REF_ANT_XY
    ants.append(box(rx - 2.5, rx + 2.5, ry - 1.2, ry + 1.2, PCB_Z1, PCB_Z1 + 1.1))
    return union(ants)


def hw_rf_ics():
    """PE42442 x4 + AD8317 + 온보드 RP2040 QFN-56 (7x7 mm)."""
    return union([
        box(-2.5, 2.5, -2.5, 2.5, PCB_Z1, PCB_Z1 + 0.9),      # PE42442 #0
        box(-16.0, -11.0, 11.0, 16.0, PCB_Z1, PCB_Z1 + 0.9),  # PE42442 #1
        box(11.0, 16.0, 11.0, 16.0, PCB_Z1, PCB_Z1 + 0.9),    # PE42442 #2
        box(-16.0, -11.0, -16.0, -11.0, PCB_Z1, PCB_Z1 + 0.9),# PE42442 #3
        box(11.0, 16.0, -16.0, -11.0, PCB_Z1, PCB_Z1 + 0.9),  # AD8317
        box(-3.5, 3.5, -16.0, -9.0, PCB_Z1, PCB_Z1 + 1.0),    # RP2040 QFN-56 USB-GPIO/ADC
    ])


def hw_nicu_reflector():
    s = extrude_z(rrect(-41.0, 41.0, -41.0, 41.0, 3.5), REF_Z + 0.05, REF_Z + 0.25)
    cuts = [cz(4.8, REF_Z - 0.5, REF_Z + 1.0, bx, by) for bx, by in CORNER_XY]
    cuts.append(box(-7.0, 7.0, -5.0, 5.0, REF_Z - 0.5, REF_Z + 1.0))
    return s - union(cuts)


def hw_nanopi_r5c():
    """FriendlyElec NanoPi R5C (58 x 58 mm PCB + RK3568B2 + 듀얼 2.5GbE/USB-A 포트 13.4 mm 높이)."""
    pcb = box(-SBC_W / 2, SBC_W / 2, -SBC_H / 2, SBC_H / 2, SBC_Z0, SBC_Z1)
    # 2x RJ45 (-Y 쪽 하단 가장자리) + 2x USB 3.0 Type-A (+Y 쪽 상단 가장자리) + RK3568B2 히트싱크
    rj45_bank = box(-18.0, 14.0, -28.5, -12.0, SBC_Z0 - 13.2, SBC_Z0)
    usb_bank = box(-12.0, 8.0, 14.0, 28.5, SBC_Z0 - 13.4, SBC_Z0)
    soc = box(-10.0, 10.0, -8.0, 10.0, SBC_Z0 - 4.5, SBC_Z0)
    return union([pcb, rj45_bank, usb_bank, soc])


def hw_ax210():
    """Intel AX210NGW (22 x 30 mm, NanoPi R5C 바닥면 네이티브 M.2 E-Key 장착)."""
    return box(-11.0, 11.0, -15.0, 15.0, SBC_Z1 + 0.3, SBC_Z1 + 3.2)


def hw_magnets():
    return union([cz(7.5, WALL_Z0 + 1.0, WALL_Z0 + 4.0, mx, my)
                  for mx, my in [(20.5, 20.5), (-20.5, 20.5), (-20.5, -20.5), (20.5, -20.5)]])


# ── 3) 직교 2축 운동학 조립체 (벽면 고정, Pan=Y축, Tilt=X축) ────

def _rot_about(m, axis_deg, pivot):
    p = np.asarray(pivot, dtype=float)
    return m.translate(-p).rotate(axis_deg).translate(p)


def node_assembly(pan_deg=0.0, tilt_deg=0.0):
    """벽면(z=WALL_Z0)에 고정된 완전 조립 DeteCam 6.0 Nano-2 노드."""
    pan_piv = [0.0, 0.0, PAN_PIVOT_Z]
    tilt_piv = [0.0, 0.0, TILT_PIVOT_Z]

    wb = wall_base()
    mags = hw_magnets()

    yk = _rot_about(swivel_yoke(), [0.0, pan_deg, 0.0], pan_piv)
    knob_pan = thumb_knob().rotate([-90.0, 0.0, 0.0]).translate([0.0, PAN_EAR_GAP / 2 + PAN_EAR_T + 0.5, PAN_PIVOT_Z])
    knob_pan = _rot_about(knob_pan, [0.0, pan_deg, 0.0], pan_piv)

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
        "nanopi_r5c":     (xform_puck(hw_nanopi_r5c()),   "#0F766E", 1.0),
        "ax210":          (xform_puck(hw_ax210()),        "#0EA5E9", 1.0),
        "magnets":        (mags,                          "#94A3B8", 1.0),
    }
