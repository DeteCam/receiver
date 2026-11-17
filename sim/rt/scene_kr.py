"""교실(한국 교실 7.5 x 9.0 x 2.7 m 40석 = kr40, 대학 강의실 9 x 12 x 3 m 72석 = lec72)을 Sionna RT 장면으로 만든다. 좌표: x = 칠판 벽을 따라, y = 앞(칠판 0) -> 뒤(9.0), z = 위."""
import os, tempfile
import numpy as np
import trimesh
import mitsuba as mi
import sionna.rt as rt

ROOMS = {   # 방마다: 폭, 깊이, 천장, 칸(x), 줄(y), 비운 자리, 칠판 x 범위, 뒷벽 사물함, 복도 쪽 문 y
    "kr40": dict(W=7.5, D=9.0, CEIL=2.7, XS=0.6 + np.arange(6) * 1.1 + 0.025, YS=2.0 + np.arange(7) * 0.95, skip=((6, 0), (6, 5)), board=(1.35, 6.15), lockers=True, doors=((0.4, 1.3), (7.7, 8.6))),
    "lec72": dict(W=9.0, D=12.0, CEIL=3.0, XS=0.9 + np.arange(8) * 1.03, YS=2.4 + np.arange(9) * 1.0, skip=(), board=(1.5, 7.5), lockers=False, doors=((0.4, 1.4), (10.5, 11.5))),   # 대학 강의실 72석
}
ROOM = "kr40"


def set_room(name):
    """방을 고른다 (모듈 전역을 바꾼다)"""
    global ROOM, W, D, CEIL, XS, YS, SEATS
    r = ROOMS[name]; ROOM = name; W, D, CEIL, XS, YS = r["W"], r["D"], r["CEIL"], r["XS"], r["YS"]
    SEATS = np.array([(x, y) for j, y in enumerate(YS) for i, x in enumerate(XS) if (j, i) not in r["skip"]])       # 책상 가운데


set_room("kr40")
DESK_H, DESK_W, DESK_D = 0.74, 0.65, 0.45
BODY_DY = 0.38            # 책상 가운데에서 몸통 가운데까지 (뒤쪽)
TMP = tempfile.mkdtemp(prefix="rtscene_")


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo); m.apply_translation((lo + hi) / 2); return m


def _prism(cx, cy, rx, ry, z0, z1, n=8, rot=0.0):
    a = rot + np.arange(n) * 2 * np.pi / n
    poly = np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(poly), z1 - z0); m.apply_translation((0, 0, z0)); return m


def _quad(p0, p1, p2, p3):
    return trimesh.Trimesh(vertices=np.array([p0, p1, p2, p3], float), faces=[[0, 1, 2], [0, 2, 3]], process=False)


def people_layout(rng=None, jitter=0.0, pose=0.0):
    """자리마다 사람: (몸통 가운데 x, y, 앉은 머리 높이, 몸 돌린 각, 머리를 숙인 만큼). jitter: 자세 흩어짐 (m). pose: 0 이면 모두 같은 자세, 1 이면 바로 앉은 사람부터 깊이 숙인 사람까지 섞는다"""
    out = []
    for (x, y) in SEATS:
        dx = dy = 0.0; h = 1.32; rot = 0.0; ln = 0.10
        if rng is not None and jitter > 0:
            dx, dy = rng.normal(0, jitter, 2); h = 1.32 + rng.normal(0, 0.05); rot = rng.normal(0, 0.25)
            if pose > 0:
                ln = rng.uniform(0.05, 0.05 + 0.25 * pose); h -= 0.5 * (ln - 0.10)          # 숙일수록 머리가 낮아진다
        out.append((x + dx, y + BODY_DY + dy, h, rot, ln))
    return out


LEAN = 0.10           # 글씨를 쓰느라 머리를 앞으로 숙인 만큼 (m)


def build(people=True, rng=None, jitter=0.0, empty_seats=(), ceil_mat="concrete", lean=LEAN, layout=None):
    """장면을 만든다. people=False 면 빈 교실. empty_seats: 비운 자리 번호들. layout: people_layout() 의 결과 (카메라 점과 같은 것을 쓰려고)"""
    scene = rt.load_scene()
    M = {
        "concrete": rt.ITURadioMaterial("m_concrete", "concrete", thickness=0.2),
        "ceil": rt.ITURadioMaterial("m_ceil", ceil_mat, thickness=0.2 if ceil_mat == "concrete" else 0.012),
        "glass": rt.ITURadioMaterial("m_glass", "glass", thickness=0.006),
        "metal": rt.ITURadioMaterial("m_metal", "metal", thickness=0.002),
        "wood_door": rt.ITURadioMaterial("m_door", "wood", thickness=0.04),
        "desk": rt.ITURadioMaterial("m_desk", "chipboard", thickness=0.018),
        "body": rt.RadioMaterial("m_body", thickness=0.2, relative_permittivity=40.0, conductivity=1.5),
    }
    objs = []; n = [0]

    def add(mesh, mat):
        n[0] += 1; f = os.path.join(TMP, "o%04d_%d.ply" % (n[0], os.getpid())); mesh.export(f)
        objs.append(rt.SceneObject(fname=f, name="o%04d" % n[0], radio_material=M[mat]))

    # 바닥, 천장 (안쪽을 보는 넓은 판)
    add(_quad((0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0)), "concrete")
    add(_quad((0, 0, CEIL), (0, D, CEIL), (W, D, CEIL), (W, 0, CEIL)), "ceil")
    # 앞벽 y=0 (칠판은 벽 앞 2cm 철판)
    add(_quad((0, 0, 0), (0, 0, CEIL), (W, 0, CEIL), (W, 0, 0)), "concrete")
    R_ = ROOMS[ROOM]; b0, b1 = R_["board"]
    add(_box((b0, 0.0, 0.9), (b1, 0.02, 2.1)), "metal")
    # 뒷벽 y=D + 철제 사물함 (깊이 0.5, 높이 1.2)
    add(_quad((0, D, 0), (W, D, 0), (W, D, CEIL), (0, D, CEIL)), "concrete")
    if R_["lockers"]:
        add(_box((0.0, D - 0.5, 0.0), (W, D, 1.2)), "metal")
    # 창 쪽 벽 x=0: 아래 0.9 콘크리트, 0.9~2.4 유리 (y 0.5~8.5), 위와 양끝 콘크리트
    def wall_x(x, pieces, flip):
        for (y0, y1, z0, z1, mat) in pieces:
            q = _quad((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))
            if flip: q.invert()
            add(q, mat)
    wall_x(0.0, [(0, D, 0, 0.9, "concrete"), (0, 0.5, 0.9, 2.4, "concrete"), (D - 0.5, D, 0.9, 2.4, "concrete"), (0.5, D - 0.5, 0.9, 2.4, "glass"), (0, D, 2.4, CEIL, "concrete")], False)
    # 복도 쪽 벽 x=W: 앞문·뒷문(나무), 복도 창(유리 1.8~2.4), 나머지 콘크리트
    (f0, f1), (k0, k1) = R_["doors"]
    wall_x(W, [(0, f0, 0, 2.1, "concrete"), (f0, f1, 0, 2.1, "wood_door"), (f1, k0, 0, 1.8, "concrete"), (k0, k1, 0, 2.1, "wood_door"), (k1, D, 0, 2.1, "concrete"),
               (f1, f1 + 0.3, 1.8, 2.1, "concrete"), (k0 - 0.3, k0, 1.8, 2.1, "concrete"), (0, f1 + 0.3, 2.1, 2.4, "concrete"), (k0 - 0.3, D, 2.1, 2.4, "concrete"),
               (f1 + 0.3, k0 - 0.3, 1.8, 2.4, "glass"), (0, D, 2.4, CEIL, "concrete")], True)
    # 교탁 (나무 상자)
    add(_box((W / 2 - 0.4, 0.9, 0.0), (W / 2 + 0.4, 1.4, 1.0)), "wood_door")
    # 책상: 나무 상판 + 그 아래 철제 서랍 판
    for k, (x, y) in enumerate(SEATS):
        add(_box((x - DESK_W / 2, y - DESK_D / 2, DESK_H - 0.018), (x + DESK_W / 2, y + DESK_D / 2, DESK_H)), "desk")
        add(_box((x - 0.27, y - 0.19, DESK_H - 0.14), (x + 0.27, y + 0.19, DESK_H - 0.135)), "metal")
    # 사람: 몸통(팔각 기둥, 폭 0.42 x 깊이 0.24) + 머리(앞으로 조금 숙임)
    if people:
        for k, (bx, by, h, rot, *ln) in enumerate(layout if layout is not None else people_layout(rng, jitter)):
            if k in empty_seats: continue
            lk = ln[0] if ln else lean
            sh = h - 0.27                                    # 어깨 높이
            add(_prism(bx, by, 0.21, 0.12, 0.44, sh, 8, rot), "body")
            add(_prism(bx, by - lk, 0.085, 0.10, sh, h, 8, rot), "body")
            add(_box((bx - 0.19, by - 0.42, 0.44), (bx + 0.19, by, 0.60)), "body")     # 허벅지 (책상 밑)
    scene.edit(add=objs)
    return scene


def cam_points(layout=None, lean=LEAN):
    """자리마다 카메라가 있을 법한 점: 책상 위 2곳, 가슴, 안경. (자리 수, 4, 3). layout 을 주면 가슴·안경 점이 그 사람의 몸을 따라간다"""
    P = np.zeros((len(SEATS), 4, 3))
    for k, (x, y) in enumerate(SEATS):
        bx, by, h = (layout[k][0], layout[k][1], layout[k][2]) if layout is not None else (x, y + BODY_DY, 1.32)
        lk = layout[k][4] if layout is not None and len(layout[k]) > 4 else lean
        P[k, 0] = (x - 0.15, y - 0.05, 0.80)                 # 책상 위 왼쪽 (필통)
        P[k, 1] = (x + 0.20, y + 0.05, 0.78)                 # 책상 위 오른쪽
        P[k, 2] = (bx, by - 0.12 - 0.03, h - 0.30)           # 가슴 (단추): 몸통 앞면에서 3cm
        P[k, 3] = (bx, by - lk - 0.10 - 0.02, h - 0.11)      # 안경: 머리 앞면에서 2cm, 눈높이
    return P


def exam_spots():
    """시험 중에 수신기를 놓을 수 있는 곳 (둘레와 교탁 쪽). kr40 은 firmware 검사(exam_mode_check.py)와 같은 18곳"""
    if ROOM == "kr40":
        return [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)] + [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + \
               [(x, 8.6) for x in (2.0, 3.75, 5.5)] + [(x, 1.3) for x in (1.5, 3.75, 6.0)]
    ys = np.arange(2.5, D - 1.0, 1.8)
    return [(0.5, 0.5), (W - 0.5, 0.5), (0.5, D - 0.4), (W - 0.5, D - 0.4)] + [(0.3, float(y)) for y in ys] + [(W - 0.3, float(y)) for y in ys] + \
           [(float(x), D - 0.4) for x in (W * 0.25, W * 0.5, W * 0.75)] + [(float(x), 1.3) for x in (W * 0.2, W * 0.5, W * 0.8)]
