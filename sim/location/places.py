"""여러 곳에서 잘 되는지 본다: 변기 칸, 욕실, 모텔 방, 강의실.
    python3 places.py [횟수] [곳 이름 ...]        결과: places_result.json

room.py 의 방 모형(벽·바닥·천장 반사를 거울상으로 계산)에 곳마다의 크기, 재질, 물건, 카메라를 숨길 만한 자리, 수신기를 놓을 수 있는 자리를 넣었다.
한 번의 실험 = 카메라를 숨길 만한 자리 한 곳에 놓고, 수신기를 첫 자리에 놓아 재고, 장치가 권하는 자리로 옮겨 가며 4자리까지 잰다.
위치 계산은 장치와 같은 코드·같은 설정(locate.py)이다. 폰 AR 이 재는 수신기 자리의 오차도 넣었다 (twopos.POSE).

곳마다 넣은 것
  변기 칸  0.9 x 1.5m. 옆 칸막이와 문은 합성수지 판 13mm(또는 철판), 바닥에서 12cm 와 2m 위는 뚫려 있다. 뒷벽은 타일. 사람은 칸 밖에 있다.
  욕실     1.6 x 2.2m. 타일, 세면대 위 거울. 사람은 밖에 있다.
  모텔 방  3.3 x 5.0m. 침대, 벽걸이 TV(금속), 화장대 거울, 유리창. 사람이 방 안에 있을 때와 나갔을 때.
  강의실   7 x 9m. 칠판(철판), 유리창, 책상 25개. 카메라까지 1~10m. 사람은 방 안에 있다.
넣지 않은 것: 변기·세면대·책상 같은 물건의 정확한 모양(약한 흩어짐으로만), 칸막이 너머 옆 칸에서 되돌아오는 파, 편파의 자세한 계산.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import RoomWorld, ROOMS
from fusion import Fusion, consistency, origin
import montecarlo as mc
import locate as K
from locate3d import measure, PLANS
from estimator import azel

HERE = os.path.dirname(os.path.abspath(__file__))
TILT3 = mc.TILT3
U = lambda rng, a, b: float(rng.uniform(a, b))

# ---------------------------------------------------------------- 방 모형 (room.ROOMS 에 더한다)
def _stall(mat):
    gaps = []
    for wall, n in (("x0", 1.5), ("x1", 1.5), ("y0", 0.9)):
        gaps.append(dict(wall=wall, rect=(0.0, n, 0.0, 0.12), mat="open")); gaps.append(dict(wall=wall, rect=(0.0, n, 2.0, 2.5), mat="open"))
    return dict(L=(0.9, 1.5, 2.5), wall=mat, floor="tile", ceil="concrete", mat=dict(y1="tile"), clutter=(-20.0, -12.0), absorb=(0.9, 1.0), patches=gaps, heights=((0.42, 1.0),))

ROOMS["변기 칸·수지"] = _stall("hpl13")
ROOMS["변기 칸·철판"] = _stall("metal")
for _k in ("수지", "철판"):                           # 문을 활짝 연 것: 문 쪽 벽(y0)이 통째로 뚫린다
    _r = dict(ROOMS["변기 칸·" + _k]); _r["mat"] = dict(y1="tile", y0="open"); ROOMS["변기 칸·%s·문열림" % _k] = _r
ROOMS["욕실"] = ROOMS["화장실"]
ROOMS["모텔 방"] = dict(L=(3.3, 5.0, 2.4), wall="brick", floor="concrete", ceil="plaster", clutter=(-18.0, -10.0), absorb=(0.6, 1.0),
                      patches=[dict(wall="x1", rect=(2.4, 3.6, 0.9, 1.6), mat="metal"),        # 벽걸이 TV
                               dict(wall="x1", rect=(1.1, 1.9, 0.8, 1.6), mat="metal"),        # 화장대 거울
                               dict(wall="y1", rect=(0.65, 2.65, 0.9, 2.2), mat="glass6"),     # 유리창
                               dict(wall="y0", rect=(0.2, 1.1, 0.0, 2.0), mat="wood40"),       # 문
                               dict(wall="z0", rect=(0.0, 2.05, 2.2, 3.8), mat="open")],       # 침대가 덮은 바닥 (파를 흩어 버린다)
                      heights=((0.0, 1.0),))
ROOMS["강의실"] = dict(L=(7.0, 9.0, 2.7), wall="concrete", floor="concrete", ceil="plaster", clutter=(-14.0, -8.0), absorb=(0.8, 1.0),
                     patches=[dict(wall="y0", rect=(1.7, 5.3, 0.9, 2.1), mat="metal"),          # 칠판 (철판)
                              dict(wall="x1", rect=(1.0, 8.0, 0.9, 2.3), mat="glass6"),         # 창
                              dict(wall="x0", rect=(0.3, 1.3, 0.0, 2.1), mat="wood40"), dict(wall="x0", rect=(7.7, 8.7, 0.0, 2.1), mat="wood40")],
                     heights=((0.72, 1.0),))
DESKS = [(x, y) for x in (1.0, 2.2, 3.5, 4.8, 6.0) for y in (2.5, 3.6, 4.7, 5.8, 6.9)]


def on_wall(rng, L, z, walls="x0 x1 y0 y1", off=0.03):
    w = str(rng.choice(walls.split())); zz = U(rng, *z)
    if w[0] == "x":
        return np.array([off if w == "x0" else L[0] - off, U(rng, 0.05, L[1] - 0.05), zz])
    return np.array([U(rng, 0.05, L[0] - 0.05), off if w == "y0" else L[1] - off, zz])


# ---------------------------------------------------------------- 카메라를 숨길 만한 자리 (비율, 이름, 자리)
def cams_stall(rng):
    L = (0.9, 1.5, 2.5); s = rng.choice([0.03, 0.87])
    return [(30, "칸막이·문", on_wall(rng, L, (0.2, 1.9), "x0 x1 y0")), (12, "뒷벽", on_wall(rng, L, (0.3, 2.2), "y1")),
            (12, "천장", np.array([U(rng, 0.2, 0.7), U(rng, 0.3, 1.3), 2.47])), (12, "휴지걸이", np.array([s, U(rng, 0.75, 0.95), U(rng, 0.6, 0.8)])),
            (12, "휴지통", np.array([U(rng, 0.03, 0.22), U(rng, 0.55, 0.8), U(rng, 0.1, 0.32)])),
            (12, "변기 뒤·옆", np.array([rng.choice([U(rng, 0.1, 0.25), U(rng, 0.65, 0.8)]), U(rng, 1.15, 1.45), U(rng, 0.1, 0.5)])),
            (10, "물탱크 위", np.array([U(rng, 0.25, 0.65), U(rng, 1.35, 1.47), U(rng, 0.85, 1.2)]))]


def cams_bath(rng):
    L = (1.6, 2.2, 2.3)
    return [(25, "벽", on_wall(rng, L, (0.2, 2.2))), (12, "천장", np.array([U(rng, 0.3, 1.3), U(rng, 0.4, 1.8), 2.27])),
            (12, "거울 둘레", np.array([0.03, U(rng, 0.5, 1.2), U(rng, 0.9, 1.75)])), (12, "세면대 위 물건", np.array([U(rng, 0.05, 0.45), U(rng, 0.55, 1.15), U(rng, 0.85, 1.05)])),
            (12, "변기 위 선반", np.array([U(rng, 1.0, 1.45), U(rng, 2.05, 2.17), U(rng, 0.85, 1.5)])), (10, "샤워기", np.array([U(rng, 0.05, 0.3), U(rng, 2.1, 2.17), U(rng, 1.6, 2.0)])),
            (9, "세면대 아래", np.array([U(rng, 0.05, 0.45), U(rng, 0.55, 1.15), U(rng, 0.15, 0.6)])), (8, "수건걸이", np.array([U(rng, 1.2, 1.57), U(rng, 0.03, 0.6), U(rng, 1.2, 1.7)]))]


def cams_motel(rng):
    L = (3.3, 5.0, 2.4); side = rng.choice([0, 1])
    return [(18, "TV·셋톱박스", np.array([U(rng, 3.0, 3.27), U(rng, 2.5, 3.5), U(rng, 0.55, 1.65)])), (10, "천장 감지기", np.array([U(rng, 1.2, 2.2), U(rng, 2.2, 3.2), 2.37])),
            (14, "협탁 위 (시계·등)", np.array([U(rng, 0.05, 0.4), U(rng, 1.8, 2.15) if side else U(rng, 3.85, 4.2), U(rng, 0.5, 0.8)])),
            (10, "화장대 위 물건", np.array([U(rng, 2.9, 3.25), U(rng, 1.05, 2.15), U(rng, 0.75, 1.0)])), (6, "거울 둘레", np.array([3.27, U(rng, 1.1, 1.9), U(rng, 0.8, 1.6)])),
            (10, "콘센트·스위치", on_wall(rng, L, (0.25, 0.4) if rng.random() < 0.6 else (1.1, 1.3))), (8, "에어컨", np.array([U(rng, 0.8, 1.8), 4.95, U(rng, 2.1, 2.3)])),
            (6, "옷장 위", np.array([U(rng, 0.05, 0.55), U(rng, 0.25, 1.25), U(rng, 1.9, 2.0)])), (6, "침대 머리판", np.array([0.05, U(rng, 2.3, 3.7), U(rng, 0.5, 1.0)])),
            (12, "벽 아무 데나", on_wall(rng, L, (0.3, 2.3)))]


def cams_class(rng):
    L = (7.0, 9.0, 2.7); dx, dy = DESKS[int(rng.integers(len(DESKS)))]
    return [(10, "벽시계 (앞)", np.array([U(rng, 3.0, 4.0), 0.03, U(rng, 2.1, 2.4)])), (10, "프로젝터", np.array([U(rng, 3.0, 4.0), U(rng, 2.5, 4.0), U(rng, 2.3, 2.67)])),
            (10, "천장", np.array([U(rng, 0.5, 6.5), U(rng, 0.5, 8.5), 2.67])), (15, "책상 아래", np.array([dx + U(rng, -0.25, 0.25), dy + U(rng, -0.18, 0.18), U(rng, 0.5, 0.68)])),
            (10, "책상 위 물건", np.array([dx + U(rng, -0.25, 0.25), dy + U(rng, -0.18, 0.18), U(rng, 0.72, 0.9)])),
            (15, "뒤 선반", np.array([U(rng, 0.5, 6.5), U(rng, 8.6, 8.95), U(rng, 1.2, 1.6)])), (10, "콘센트", on_wall(rng, L, (0.25, 0.45))),
            (10, "교탁", np.array([U(rng, 3.1, 3.9), U(rng, 0.7, 1.3), U(rng, 0.9, 1.2)])), (10, "벽 아무 데나", on_wall(rng, L, (0.5, 2.5)))]


# ---------------------------------------------------------------- 수신기를 놓을 수 있는 자리 (x, y, z) 와 첫 자리로 고를 비율
def grid_pts(x, y, z, step):
    return [(float(a), float(b), z) for a in np.arange(x[0], x[1] + 1e-9, step) for b in np.arange(y[0], y[1] + 1e-9, step)]

SPOTS = {
    "stall": ([(0.45, 1.05, 0.42)] + [(0.25, 0.35, 0.0), (0.65, 0.35, 0.0), (0.45, 0.30, 0.0), (0.45, 0.60, 0.0), (0.27, 0.60, 0.0), (0.63, 0.60, 0.0)], [0.6] + [0.4 / 6] * 6),
    "bath": ([(0.9, 0.9, 0.0), (0.8, 0.45, 0.0), (1.1, 0.6, 0.0), (0.8, 1.25, 0.0), (1.25, 1.15, 0.0), (0.6, 1.3, 0.0), (0.45, 1.75, 0.0), (1.22, 1.8, 0.42), (0.3, 0.85, 0.85)],
             [0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0, 0.2, 0.1]),
    "motel": (grid_pts((2.25, 2.75), (0.5, 4.6), 0.0, 0.3) + grid_pts((0.9, 2.1), (0.4, 1.6), 0.0, 0.3) + grid_pts((0.7, 2.1), (4.2, 4.7), 0.0, 0.35) + [(3.07, 1.6, 0.75), (0.22, 1.97, 0.5)], None),
    "class": ([(x, y, 0.72) for x, y in DESKS] + [(3.5, 1.0, 1.0)] + [(x, y, 0.0) for x in (1.6, 2.85, 4.15, 5.4) for y in (2.0, 3.1, 4.2, 5.3, 6.4, 7.5)], None),
}
PLACES = {
    "변기 칸 (합성수지 칸막이)": dict(room="변기 칸·수지", spots="stall", cams=cams_stall, order=4, user=None, min_cam=0.35),
    "변기 칸 (철판 칸막이)": dict(room="변기 칸·철판", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35),
    "욕실": dict(room="욕실", spots="bath", cams=cams_bath, order=3, user=None, min_cam=0.4),
    "모텔 방 (사람은 나가 있음)": dict(room="모텔 방", spots="motel", cams=cams_motel, order=3, user=None, min_cam=0.5),
    "모텔 방 (사람이 방 안에)": dict(room="모텔 방", spots="motel", cams=cams_motel, order=3, user=(1.2, 2.5), min_cam=0.5),
    "강의실": dict(room="강의실", spots="class", cams=cams_class, order=3, user=(1.5, 3.0), min_cam=0.5),
    # ---- 더 해 본 것 (좁은 곳에서 나아지는 방법 찾기)
    "변기 칸 (합성수지) 여섯 번 재기": dict(room="변기 칸·수지", spots="stall", cams=cams_stall, order=4, user=None, min_cam=0.35, plan="tilt6"),
    "변기 칸 (합성수지) 문 열고": dict(room="변기 칸·수지·문열림", spots="stall", cams=cams_stall, order=4, user=None, min_cam=0.35),
    "변기 칸 (합성수지) 경고 뜬 자리는 덜 믿기": dict(room="변기 칸·수지", spots="stall", cams=cams_stall, order=4, user=None, min_cam=0.35, qflag=0.5),
    "변기 칸 (합성수지) 사람이 칸 안에": dict(room="변기 칸·수지", spots="stall", cams=cams_stall, order=4, user=(0.3, 0.55), min_cam=0.35),
    "변기 칸 (철판) 여섯 번 재기": dict(room="변기 칸·철판", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35, plan="tilt6"),
    "변기 칸 (철판) 문 열고": dict(room="변기 칸·철판·문열림", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35),
    "변기 칸 (철판) 문 열고 여섯 번": dict(room="변기 칸·철판·문열림", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35, plan="tilt6"),
    "욕실 여섯 번 재기": dict(room="욕실", spots="bath", cams=cams_bath, order=3, user=None, min_cam=0.4, plan="tilt6"),
    "욕실 경고 뜬 자리는 덜 믿기": dict(room="욕실", spots="bath", cams=cams_bath, order=3, user=None, min_cam=0.4, qflag=0.5),
    "모텔 방 여섯 번 재기": dict(room="모텔 방", spots="motel", cams=cams_motel, order=3, user=None, min_cam=0.5, plan="tilt6"),
    # ---- 여섯 번 재면 한 자리의 측정을 너무 믿게 되는지: 한 자리의 스캔들은 같은 반사를 겪어 함께 틀린다. 그래서 장치(locate.py 의 SIG_MANY)는 여섯 번 넘게 잰 자리의
    #      흔들림을 루트 2배 넓게 본다. 아래는 그렇게 하지 않았을 때(many=1.0)와 견주려고 둔 것이다
    "변기 칸 (합성수지) 여섯 번 재기 (넓히지 않고)": dict(room="변기 칸·수지", spots="stall", cams=cams_stall, order=4, user=None, min_cam=0.35, plan="tilt6", many=1.0),
    "변기 칸 (철판) 여섯 번 재기 (넓히지 않고)": dict(room="변기 칸·철판", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35, plan="tilt6", many=1.0),
    "변기 칸 (철판) 문 열고 여섯 번 (넓히지 않고)": dict(room="변기 칸·철판·문열림", spots="stall", cams=cams_stall, order=6, user=None, min_cam=0.35, plan="tilt6", many=1.0),
    "욕실 여섯 번 재기 (넓히지 않고)": dict(room="욕실", spots="bath", cams=cams_bath, order=3, user=None, min_cam=0.4, plan="tilt6", many=1.0),
    "모텔 방 여섯 번 재기 (넓히지 않고)": dict(room="모텔 방", spots="motel", cams=cams_motel, order=3, user=None, min_cam=0.5, plan="tilt6", many=1.0),
}
BASE_PLACES = ["변기 칸 (합성수지 칸막이)", "변기 칸 (철판 칸막이)", "욕실", "모텔 방 (사람은 나가 있음)", "모텔 방 (사람이 방 안에)", "강의실"]
NPOS = 4


def user_spot(rng, base, L, rng_m):
    """재는 동안 사람이 서 있는 자리: 수신기에서 rng_m 만큼 떨어진 아무 곳 (카메라가 어디 있는지 모르니 가끔은 길을 막는다)"""
    for _ in range(40):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(*rng_m)
        q = np.asarray(base)[:2] + r * np.array([np.cos(a), np.sin(a)])
        if 0.2 <= q[0] <= L[0] - 0.2 and 0.2 <= q[1] <= L[1] - 0.2:
            return q
    return np.clip(q, 0.2, np.array(L[:2]) - 0.2)


def post_of(grid, Ms, qflag=None, sig=None):
    """장치와 같은 계산. qflag 를 주면 '스캔끼리 안 맞는다'는 경고가 뜬 자리는 틀렸을 확률을 qflag 로 높여서(덜 믿고) 곱한다. sig 를 주면 골 가운데의 흔들림을 그 배수로 본다"""
    sig = K.SIG_SCALE if sig is None else sig
    if qflag is None:
        return K.posterior(grid, Ms, None, K.Q_BAD, sig)
    p = np.ones(len(grid.X))
    for m in Ms:
        p *= K.likelihood(grid, m, K.Q_BAD if m["ok"] else qflag, sig)
    return p / p.sum()


def settings(name):
    """이 곳을 어떤 설정으로 돌렸는지 (결과 파일에 같이 적는다)"""
    Pl = PLACES[name]; plan = Pl.get("plan", "tilt3"); many = len(PLANS[plan]) >= K.MANY_N
    return dict(plan=plan, sig_many=round(float(Pl.get("many", K.SIG_MANY)), 3) if many else 1.0, qflag=Pl.get("qflag"))


def run(name, shape="W", n=200, seed=71, plan=None, npos=NPOS):
    many0 = K.SIG_MANY; K.SIG_MANY = PLACES[name].get("many", many0)
    try:
        return _run(name, shape, n, seed, plan, npos)
    finally:
        K.SIG_MANY = many0


def _run(name, shape, n, seed, plan, npos):
    Pl = PLACES[name]; R = ROOMS[Pl["room"]]; L = np.array(R["L"]); grid = K.Grid(L)
    plan = plan or Pl.get("plan", "tilt3"); qflag = Pl.get("qflag"); sig = Pl.get("sig")
    S = K.Session((0, 0, 0), tuple(L)); kk = float(min(1.0, max(0.4, min(L[0], L[1]) / 2.0))); good_len = float(min(K.GOOD_LEN, K.GOOD_FRAC * max(L[0], L[1])))
    spots, wts = SPOTS[Pl["spots"]]; spots = np.array(spots, dtype=float)
    wts = np.ones(len(spots)) / len(spots) if wts is None else np.array(wts) / np.sum(wts)
    fus = Fusion(); frames = PLANS[plan]; rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        opts = Pl["cams"](rng); w = np.array([o[0] for o in opts], dtype=float); j = int(rng.choice(len(opts), p=w / w.sum()))
        label, cam = opts[j][1], opts[j][2]
        hB = origin(frames)
        ok0 = [k for k in range(len(spots)) if wts[k] > 0 and np.linalg.norm(spots[k] + hB - cam) >= Pl["min_cam"]]
        if not ok0:
            continue
        p0 = spots[int(rng.choice(ok0, p=wts[ok0] / wts[ok0].sum()))]
        absorb = [None]
        def world_at(p):
            w_ = RoomWorld(rng, Pl["room"], shape, cam=cam, base=p, order=Pl["order"])
            if absorb[0] is None:
                absorb[0] = w_.absorb
            w_.absorb = absorb[0]
            if Pl["user"] is not None:
                w_.person = user_spot(rng, p, L, Pl["user"])
            return w_
        wA = world_at(p0); A = measure(wA, fus, rng, frames)
        row = dict(label=label, dist=float(np.linalg.norm(cam - wA.ref)), el=float(azel(wA.c)[1]), err1=A["err"], ok1=bool(A["ok"]), seen1=int(sum(o["seen"] for o in A["obs"])), kfac=wA.kfac)
        Ms = [A]; used = [p0]; stop = None
        for m in range(2, npos + 1):
            free = np.array([all(np.linalg.norm(s_[:2] - u_[:2]) >= 0.6 * kk or abs(s_[2] - u_[2]) > 0.2 for u_ in used) and np.linalg.norm(s_ + hB - cam) >= Pl["min_cam"] * 0.8 for s_ in spots])
            C = spots[free]
            if len(C) == 0:
                break
            pn = None
            if m == 2:
                if A["d"] is not None:
                    cost = K.second_cost(A["ref"], A["d"], C, np.zeros(3), L, frames, near=0.5 * kk)
                    if np.isfinite(cost).any():
                        pn = C[int(np.argmin(cost))]
            else:
                post = post_of(grid, Ms, qflag, sig); c, idx = K.zone(grid, post)[:2]; _, axis = K.long_axis(grid, post, idx)
                sc = K.third_score([q["ref"] for q in Ms], c, axis, C, frames, near=0.5 * kk)
                if np.isfinite(sc).any():
                    pn = C[int(np.argmax(sc))]
            if pn is None:
                pn = C[int(rng.integers(len(C)))]
            Ms.append(measure(world_at(pn), fus, rng, frames)); used.append(pn)
            post = post_of(grid, Ms, qflag, sig)
            c, idx, length, vol, conf, second = K.zone(grid, post)
            e = float(np.linalg.norm(c - cam)); gap = K.zone_gap(grid, idx, cam)
            row["e%d" % m] = e; row["len%d" % m] = length; row["conf%d" % m] = conf; row["gap%d" % m] = gap
            if stop is None and length <= good_len and conf >= K.GOOD_CONF:
                stop = m; row["stop"] = m; row["e_stop"] = e; row["gap_stop"] = gap; row["len_stop"] = length
        if stop is None:
            row["stop"] = 0; last = max(k for k in range(2, npos + 1) if ("e%d" % k) in row) if any(("e%d" % k) in row for k in range(2, npos + 1)) else None
            if last:
                row["e_stop"] = row["e%d" % last]; row["gap_stop"] = row["gap%d" % last]; row["len_stop"] = row["len%d" % last]
        rows.append(row)
    return rows, good_len


def summ(rows, good_len, npos=NPOS):
    f = lambda k: np.array([r[k] for r in rows if k in r], dtype=float)
    pc = lambda a: round(float(100.0 * np.mean(a)), 1) if len(a) else None
    md = lambda a: round(float(np.median(a)), 2) if len(a) else None
    e1 = f("err1"); out = dict(n=len(rows), dist=md(f("dist")), dist_p90=round(float(np.percentile(f("dist"), 90)), 2), kfac=md(f("kfac")), good_len=good_len,
                               dir_med=md(e1), dir_in10=pc(e1 <= 10), dir_in20=pc(e1 <= 20), dir_over30=pc(e1 > 30),
                               dir_flagged=pc(np.array([not r["ok1"] for r in rows])), dir_bad_unflagged=pc(np.array([(r["err1"] > 30) and r["ok1"] for r in rows])))
    for m in range(2, npos + 1):
        e = f("e%d" % m); g = f("gap%d" % m)
        if len(e):
            out["pos%d" % m] = dict(med=md(e), in20=pc(e <= 0.2), in30=pc(e <= 0.3), in50=pc(e <= 0.5), zone_len=md(f("len%d" % m)), in_zone15=pc(g <= 0.15), in_zone25=pc(g <= 0.25), conf=md(f("conf%d" % m)))
    st = f("stop"); es = f("e_stop"); gs = f("gap_stop")
    out["stop"] = dict(at2=pc(st == 2), at3=pc(st == 3), at4=pc(st == 4), never=pc(st == 0), med=md(es), in30=pc(es <= 0.3), in50=pc(es <= 0.5), in_zone15=pc(gs <= 0.15), zone_len=md(f("len_stop")),
                       good_med=md(np.array([r["e_stop"] for r in rows if r.get("stop", 0) > 0 and "e_stop" in r])),
                       good_miss=pc(np.array([(r.get("stop", 0) > 0) and (r.get("gap_stop", 9) > 0.25) for r in rows])))
    lab = {}
    for r in rows:
        lab.setdefault(r["label"], []).append(r)
    out["by_label"] = {k: dict(n=len(v), dir_med=md(np.array([r["err1"] for r in v])), dir_over30=pc(np.array([r["err1"] > 30 for r in v])),
                               pos_med=md(np.array([r["e_stop"] for r in v if "e_stop" in r])), in30=pc(np.array([r["e_stop"] <= 0.3 for r in v if "e_stop" in r]))) for k, v in lab.items()}
    d = f("dist"); bins = [(0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 99.0)]
    out["by_dist"] = {"%g~%gm" % b: dict(n=int(((d >= b[0]) & (d < b[1])).sum()), dir_med=md(e1[(d >= b[0]) & (d < b[1])]), dir_over30=pc(e1[(d >= b[0]) & (d < b[1])] > 30)) for b in bins if ((d >= b[0]) & (d < b[1])).sum() >= 8}
    return out


def show(key, S):
    print("== %s  (카메라까지 가운데값 %.1fm, 열에 아홉은 %.1fm 안. K %.1fdB. '믿을 만' 기준 = 구간 %.1fm 이하)" % (key, S["dist"], S["dist_p90"], S["kfac"], S["good_len"]))
    print("   방향 (첫 자리): 가운데값 %.1f도, 10도 안 %.0f%%, 30도 넘음 %.0f%% | 경고 %.0f%%, 경고 없이 틀림 %.0f%%" % (S["dir_med"], S["dir_in10"], S["dir_over30"], S["dir_flagged"], S["dir_bad_unflagged"]))
    for m in range(2, NPOS + 1):
        if ("pos%d" % m) in S:
            t = S["pos%d" % m]
            print("   %d자리: 위치 오차 가운데값 %.2fm, 20cm 안 %.0f%%, 30cm 안 %.0f%%, 50cm 안 %.0f%% | 구간 길이 %.2fm, 참 위치가 구간(+15cm) 안 %.0f%%" % (m, t["med"], t["in20"], t["in30"], t["in50"], t["zone_len"], t["in_zone15"]))
    t = S["stop"]
    print("   멈춤 규칙대로: 2자리에서 끝 %.0f%%, 3자리 %.0f%%, 4자리 %.0f%%, 4자리로도 안 좁혀짐 %.0f%% | 끝났을 때 오차 가운데값 %s m, 30cm 안 %s%%, 구간(+15cm) 안 %s%%, 좁혀졌는데 틀림 %s%%" % (
        t["at2"], t["at3"], t["at4"], t["never"], t["med"], t["in30"], t["in_zone15"], t["good_miss"]))
    print("   숨긴 곳별 (방향 가운데값 / 30도 넘음 / 위치 가운데값): " + "  ".join("%s %.0f도/%.0f%%/%sm" % (k, v["dir_med"], v["dir_over30"], v["pos_med"]) for k, v in S["by_label"].items()))
    print("   거리별 방향: " + "  ".join("%s %.0f도 (30도 넘음 %.0f%%, %d번)" % (k, v["dir_med"], v["dir_over30"], v["n"]) for k, v in S["by_dist"].items()), flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names = sys.argv[2:] or BASE_PLACES
    if names == ["more"]:
        names = [k for k in PLACES if k not in BASE_PLACES]
    path = os.path.join(HERE, "places_result.json")
    out = json.load(open(path, encoding="utf-8"))["summary"] if os.path.exists(path) else {}
    for name in names:
        for shape in ("W", "V"):
            t0 = time.time(); rows, gl = run(name, shape, n); S = summ(rows, gl); key = "%s|%s" % (name, shape); out[key] = S
            S.update(settings(name))
            show(key, S); print("   (%.0f초)" % (time.time() - t0), flush=True)
            json.dump(dict(n=n, npos=NPOS, sig_scale=K.SIG_SCALE, q_bad=K.Q_BAD, sig_many=K.SIG_MANY, many_n=K.MANY_N, summary=out), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
