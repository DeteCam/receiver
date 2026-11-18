"""숙소 방 하나(모텔·원룸 크기 3.6 x 5.4 x 2.4 m, 현관 옆에 욕실)를 Sionna RT 장면으로 만든다.
좌표: x = 방의 폭 (0 = 침대 머리 쪽 벽, W = TV 쪽 벽), y = 현관(0) -> 창(D), z = 위.

몰카를 숨길 만한 곳(hide_points), 위치를 고를 때 훑는 겉면의 점(surface_grid), 패널을 두는 곳(PLACES)을 함께 준다.
"""
import os, tempfile
import numpy as np

W, D, CEIL = 3.6, 5.4, 2.4
BX, BY = 1.7, 2.2                       # 욕실: x 0~BX, y 0~BY (벽 두께 0.1)
TMP = tempfile.mkdtemp(prefix="rtroom_")

# 패널을 두는 곳: (이름, 자리, 바라보는 점). 0~4 는 방 가운데에 서서 패널을 들고 다섯 쪽으로 돌린 것, 10 은 현관 통로(욕실 벽과 옆벽 사이)에 서서 현관 쪽을 본 것
MID = (1.9, 2.95, 1.3)
PLACES = [
    ("가운데에서 창 쪽", MID, (1.9, D, 1.3)), ("가운데에서 현관 쪽", MID, (2.4, 0.0, 1.3)), ("가운데에서 TV 쪽", MID, (W, 2.95, 1.3)),
    ("가운데에서 침대 머리 쪽", MID, (0.0, 3.4, 1.0)), ("가운데에서 천장 쪽", MID, (1.9, 3.3, CEIL)),
    ("책상 위 (받침 발)", (3.30, 3.40, 0.95), (1.2, 3.2, 1.3)), ("창턱", (1.90, D - 0.10, 1.10), (1.9, 2.0, 1.3)), ("현관문 안쪽", (2.85, 0.12, 1.50), (1.9, 3.6, 1.2)),
    ("욕실 문 앞에서 욕실 쪽", (2.25, 1.55, 1.40), (0.6, 1.0, 1.6)), ("욕실 안 세면대 위", (1.45, 0.35, 1.20), (0.5, 1.4, 1.8)),
    ("현관 통로에서 현관 쪽", (2.75, 2.00, 1.40), (2.4, 0.1, 1.9)),
    # 11~14: 방 구석 천장 가까이(2.1 m)에 고정해 방 대각선 쪽을 본 것 (2026-11-19)
    ("구석: 창-침대 쪽", (0.15, D - 0.15, 2.10), (1.9, 2.9, 1.0)), ("구석: 창-책상 쪽", (W - 0.15, D - 0.15, 2.10), (1.7, 2.9, 1.0)),
    ("구석: 현관-옷장 쪽", (W - 0.15, 0.15, 2.10), (1.9, 3.5, 1.0)), ("구석: 욕실벽-침대 쪽", (0.15, BY + 0.20, 2.10), (2.2, 4.2, 1.0)),
]


def _box(lo, hi):
    import trimesh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo); m.apply_translation((lo + hi) / 2); return m


def _quad(p0, p1, p2, p3):
    import trimesh
    return trimesh.Trimesh(vertices=np.array([p0, p1, p2, p3], float), faces=[[0, 1, 2], [0, 2, 3]], process=False)


def build(door_open=False):
    """장면을 만든다. door_open: 욕실 문을 열어 둔다 (문짝을 뺀다)"""
    import sionna.rt as rt
    scene = rt.load_scene()
    M = {
        "concrete": rt.ITURadioMaterial("m_concrete", "concrete", thickness=0.2),
        "ceil": rt.ITURadioMaterial("m_ceil", "ceiling_board", thickness=0.012),
        "brick": rt.ITURadioMaterial("m_brick", "brick", thickness=0.1),
        "glass": rt.ITURadioMaterial("m_glass", "glass", thickness=0.006),
        "metal": rt.ITURadioMaterial("m_metal", "metal", thickness=0.002),
        "wood": rt.ITURadioMaterial("m_wood", "wood", thickness=0.04),
        "board": rt.ITURadioMaterial("m_board", "chipboard", thickness=0.018),
    }
    objs = []; n = [0]

    def add(mesh, mat):
        n[0] += 1; f = os.path.join(TMP, "r%04d_%d.ply" % (n[0], os.getpid())); mesh.export(f)
        objs.append(rt.SceneObject(fname=f, name="r%04d" % n[0], radio_material=M[mat]))

    add(_quad((0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0)), "concrete")                       # 바닥
    add(_quad((0, 0, CEIL), (0, D, CEIL), (W, D, CEIL), (W, 0, CEIL)), "ceil")               # 천장 (석고보드)
    def wall_y(y, pieces, back):
        for (x0, x1, z0, z1, mat) in pieces:
            add(_quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)) if back else _quad((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0)), mat)
    def wall_x(x, pieces, flip):
        for (y0, y1, z0, z1, mat) in pieces:
            q = _quad((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))
            if flip: q.invert()
            add(q, mat)
    wall_y(0.0, [(0, 2.4, 0, CEIL, "concrete"), (2.4, 3.3, 0, 2.05, "wood"), (2.4, 3.3, 2.05, CEIL, "concrete"), (3.3, W, 0, CEIL, "concrete")], False)      # 현관 벽 (문은 나무)
    wall_y(D, [(0, 0.9, 0, CEIL, "concrete"), (0.9, 2.9, 0, 0.9, "concrete"), (0.9, 2.9, 0.9, 2.1, "glass"), (0.9, 2.9, 2.1, CEIL, "concrete"), (2.9, W, 0, CEIL, "concrete")], True)   # 창 벽
    wall_x(0.0, [(0, D, 0, CEIL, "concrete")], False); wall_x(W, [(0, D, 0, CEIL, "concrete")], True)
    # 욕실 벽 (벽돌 0.1) 과 문
    add(_box((BX - 0.05, 0.0, 0.0), (BX + 0.05, 1.2, CEIL)), "brick"); add(_box((BX - 0.05, 1.9, 0.0), (BX + 0.05, BY + 0.05, CEIL)), "brick")
    add(_box((BX - 0.05, 1.2, 2.0), (BX + 0.05, 1.9, CEIL)), "brick"); add(_box((0.0, BY - 0.05, 0.0), (BX - 0.05, BY + 0.05, CEIL)), "brick")
    if not door_open:
        add(_box((BX - 0.02, 1.2, 0.0), (BX + 0.02, 1.9, 2.0)), "wood")
    add(_box((0.0, 0.4, 1.0), (0.01, 1.0, 1.7)), "metal")                                     # 욕실 거울
    # 가구: 침대, 협탁, 책상과 냉장고, TV, 현관 쪽 거울, 옷장
    add(_box((0.15, 3.0, 0.0), (2.15, 4.6, 0.45)), "wood"); add(_box((0.10, 2.45, 0.0), (0.50, 2.90, 0.55)), "wood")
    add(_box((3.05, 2.6, 0.72), (W, 4.2, 0.75)), "board"); add(_box((3.10, 4.3, 0.0), (3.58, 4.8, 0.80)), "metal")
    add(_box((3.55, 2.95, 1.00), (3.59, 3.95, 1.58)), "metal"); add(_box((3.585, 1.40, 0.70), (W, 2.00, 1.90)), "metal")
    add(_box((1.80, 0.05, 0.0), (2.30, 0.60, 2.00)), "wood")
    scene.edit(add=objs)
    return scene


# 숨길 만한 곳: 이름 -> 점을 하나 뽑는 함수. 벽·천장에서 4cm 띄운다 (물건 속에 든 카메라)
def _wall_clock(r):
    k = r.integers(4)
    if k == 0: return (0.04, r.uniform(2.4, 5.2), r.uniform(1.5, 2.0))
    if k == 1: return (W - 0.04, r.uniform(0.3, 1.2), r.uniform(1.5, 2.0))
    if k == 2: return (W - 0.04, r.uniform(4.3, 5.2), r.uniform(1.5, 2.0))
    return (r.uniform(0.1, 0.8), D - 0.04, r.uniform(1.5, 2.0))


def _bath_shelf(r):
    if r.random() < 0.5: return (0.04, r.uniform(0.2, 2.0), r.uniform(1.1, 1.9))
    return (r.uniform(0.2, 1.5), 0.04, r.uniform(1.1, 1.9))


def _entrance(r):
    if r.random() < 0.5: return (r.uniform(1.85, 2.25), r.uniform(0.15, 0.55), 2.04)       # 옷장 위
    return (r.uniform(2.45, 3.25), 0.05, r.uniform(2.10, 2.30))                             # 현관문 위


HIDE = {
    "천장 화재감지기": lambda r: (r.uniform(1.9, 3.2), r.uniform(2.6, 4.8), CEIL - 0.04),
    "벽시계·액자": _wall_clock,
    "TV·셋톱박스 둘레": lambda r: (W - r.uniform(0.08, 0.45), r.uniform(2.7, 4.1), r.uniform(0.78, 0.98)),
    "에어컨·커튼 상자": lambda r: (r.uniform(1.0, 2.8), D - 0.12, r.uniform(2.15, 2.32)),
    "침대 머리맡 (콘센트·충전기·스탠드)": lambda r: (r.uniform(0.04, 0.30), r.uniform(2.5, 4.6), r.uniform(0.25, 0.95)),
    "창가 화분·휴지통": lambda r: (r.uniform(2.5, 3.4), r.uniform(4.9, 5.3), r.uniform(0.1, 0.9)),
    "현관 쪽 (옷장 위·문 위)": _entrance,
    "욕실 환풍기": lambda r: (r.uniform(0.5, 1.2), r.uniform(0.6, 1.6), CEIL - 0.04),
    "욕실 선반·거울 둘레": _bath_shelf,
}
CATS = list(HIDE)

# 겉면에 붙지 않은 곳 (물건 속, 공중). 부피 계산(vol_study.py)에서만 쓴다 (2026-11-19)
HIDE_FREE = {
    "책상 위 물건 (겉면에서 10~40 cm 위)": lambda r: (r.uniform(3.15, 3.50), r.uniform(2.7, 4.1), r.uniform(0.85, 1.15)),
    "침대 옆 협탁 위 물건": lambda r: (r.uniform(0.20, 0.60), r.uniform(2.35, 2.90), r.uniform(0.50, 0.90)),
    "걸어 둔 옷·가방 (공중)": lambda r: (r.uniform(2.50, 3.30), r.uniform(0.60, 1.80), r.uniform(0.90, 1.70)),
    "바닥에 둔 가방·상자": lambda r: (r.uniform(1.00, 3.20), r.uniform(2.40, 3.00), r.uniform(0.15, 0.45)),
}


def hide_points(rng, per=16, free=False):
    """숨긴 카메라 점 [C,3] 과 그 종류 번호 [C]. free: 겉면에 붙지 않은 곳도 넣는다 (종류 이름은 all_cats(True))"""
    H = dict(HIDE, **HIDE_FREE) if free else HIDE
    P = [H[c](rng) for c in H for _ in range(per)]
    return np.array(P, float), np.repeat(np.arange(len(H)), per)


def all_cats(free=False):
    return list(HIDE) + (list(HIDE_FREE) if free else [])


def surface_grid(step=0.2, off=0.04):
    """위치를 고를 때 훑는 점: 벽·천장·욕실 벽·가구 겉면에서 off 만큼 띄운 점들 [N,3]"""
    X = []
    xs = np.arange(0.1, W - 0.05, step); ys = np.arange(0.1, D - 0.05, step); zs = np.arange(0.1, CEIL - 0.05, step)
    X += [(x, y, CEIL - off) for x in xs for y in ys]                                           # 천장
    X += [(off, y, z) for y in ys for z in zs] + [(W - off, y, z) for y in ys for z in zs]      # 두 옆벽
    X += [(x, off, z) for x in xs for z in zs] + [(x, D - off, z) for x in xs for z in zs]      # 현관 벽, 창 벽
    by = np.arange(0.1, BY - 0.05, step); bx = np.arange(0.1, BX - 0.05, step)
    X += [(BX - 0.05 - off, y, z) for y in by for z in zs] + [(BX + 0.05 + off, y, z) for y in by for z in zs]      # 욕실 벽 안쪽·바깥쪽
    X += [(x, BY - 0.05 - off, z) for x in bx for z in zs] + [(x, BY + 0.05 + off, z) for x in bx for z in zs]
    X += [(x, y, 0.45 + off) for x in np.arange(0.25, 2.15, step) for y in np.arange(3.1, 4.6, step)]              # 침대 위
    X += [(x, y, 0.75 + off) for x in np.arange(3.15, W, step) for y in np.arange(2.7, 4.2, step)]                 # 책상 위
    X += [(x, y, 0.80 + off) for x in (3.2, 3.4) for y in (4.4, 4.6)] + [(x, y, 2.0 + off) for x in (1.9, 2.1) for y in (0.2, 0.4)] + [(x, y, 0.55 + off) for x in (0.2, 0.4) for y in (2.55, 2.75)]
    X += [(x, y, z) for x in (2.6, 3.0, 3.4) for y in (4.95, 5.2) for z in (0.15, 0.45, 0.75)]                     # 창가 바닥 쪽
    return np.array(X, float)
