"""40명이 앉아 있는 교실에서, 시험지를 찍어 폰으로 보내는 카메라가 '몇 번 자리'인지 맞히나.   python3 classroom.py [횟수]   결과: classroom_result.json

교실: 9.0 x 7.2 x 2.8m, 콘크리트 벽, 앞벽 칠판(철판), 오른쪽 벽 유리창. 책상·의자 쇠다리는 흩어지는 약한 파로.
자리: 5줄 x 8칸 = 40자리 (옆 간격 1.0m, 앞뒤 1.1m). 사람 40명은 앉은 키 1.3m, 반지름 0.2m 기둥이고 움직이지 않는다.
카메라: 한 자리를 뽑아, 그 사람 책상 위(필통 등, 높이 0.75~0.9m) 또는 가슴(옷 단추, 몸 바로 앞 1.1~1.25m)에 둔다. 카메라 주인의 몸도 기둥이다.
수신기: 시험 중에 둘 수 있는 곳만 (모서리, 벽, 천장, 빔프로젝터 위). 대수와 높이를 바꿔 본다. 감독관이 들고 가까이 가는 것은 견줌용.
  - 앞 두 모서리, 교탁 높이 0.75m
  - 앞 두 모서리, 높은 곳 1.9m (칠판 위 선반·사물함 위: 앉은 사람 머리 위로 본다)
  - 앞 가운데 + 옆벽 가운데, 높이 1.9m (두 선이 직각에 가깝게 만난다)
방향은 장치의 코드(fusion.py)로 구하고, 자리 고르기는 장치의 위치 코드(locate.likelihood)를 그대로 쓴다:
자리마다 카메라가 있을 법한 점(책상 위·가슴)을 여럿 두고, 두 자리의 측정이 그 점을 얼마나 그럴듯하게 보는지를 곱해, 가장 그럴듯한 자리를 고른다.
넣지 않은 것: 사람 몸의 자세한 모양(기둥으로만), 사람이 조금씩 움직이는 것, 다른 학생의 폰이 같은 채널을 쓰는 것(패킷 수에만 영향).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import RoomWorld, ROOMS, fold, strip_factor, LAM_M
from fusion import Fusion
import locate as K
from locate3d import measure, PLANS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS["교실"] = dict(L=(9.0, 7.2, 2.8), wall="concrete", floor="tile", ceil="plaster", clutter=(-16.0, -8.0), absorb=(0.5, 0.9),
                    patches=[dict(wall="y0", rect=(1.5, 7.5, 0.9, 2.1), mat="metal"),            # 칠판 (앞벽, 철판)
                             dict(wall="x1", rect=(1.0, 6.6, 0.9, 2.4), mat="glass6"),           # 창 (오른쪽 벽)
                             dict(wall="y0", rect=(0.2, 1.1, 0.0, 2.0), mat="wood40")],          # 문
                    heights=((0.75, 1.0),))
COLS, ROWS = 8, 5
SEAT_X = 1.0 + np.arange(COLS) * 1.0                     # 0.5m 띄고 1.0m 간격
SEAT_Y = 2.2 + np.arange(ROWS) * 1.1                     # 앞에서 2.2m 부터 (교탁·통로)
BODY_R, BODY_H = 0.20, 1.30
SEATS = np.array([(x, y) for y in SEAT_Y for x in SEAT_X])    # 자리 번호 = 줄 * 8 + 칸
RX = {"앞 두 모서리 · 교탁 높이 0.75m": [(0.6, 0.6, 0.75), (8.4, 0.6, 0.75)],
      "앞 두 모서리 · 1.9m": [(0.6, 0.6, 1.9), (8.4, 0.6, 1.9)],
      "앞 두 모서리 · 천장 밑 2.6m": [(0.6, 0.6, 2.6), (8.4, 0.6, 2.6)],
      "앞 가운데 + 옆벽 · 2.6m": [(4.5, 0.5, 2.6), (8.6, 4.4, 2.6)],
      "옮겨 가며 3자리 · 2.0m": "adaptive",
      "모서리 2 + 가까이 2 (옮겨 가며)": "close"}       # 앞 두 모서리 2.6m 에서 잰 뒤, 감독관이 들고 가장 그럴듯한 자리의 뒤 1m, 옆뒤 대각 1.5m 에서 (머리 위 1.8m) 잰다         # 앞 가운데에서 한 번 -> 그 결과의 줄 옆 벽 쪽 통로 -> 그 자리 뒤 (감독관이 받침대에 올려 옮긴다)
C4 = [(0.6, 0.6), (8.4, 0.6), (0.6, 6.6), (8.4, 6.6)]
RX.update({"앞 둘 + 뒤 하나 모서리 · 2.6m": [(x, y, 2.6) for x, y in C4[:3]],
           "네 모서리 · 2.6m": [(x, y, 2.6) for x, y in C4],
           "네 모서리 · 1.9m": [(x, y, 1.9) for x, y in C4],
           "네 모서리 + 옆벽 가운데 · 2.6m": [(x, y, 2.6) for x, y in C4 + [(0.35, 3.6), (8.65, 3.6)]]})     # 시험 중에도 둘 수 있는 곳: 모서리와 벽 (높이 달아 둔다)
RX.update({"천장 가운데 한 곳 · 2.7m": [(4.5, 4.2, 2.7)],                                       # 위에서 내려다보면 방향만으로 책상 높이의 점이 정해진다
           "천장 가운데 두 곳 · 2.7m": [(4.5, 3.0, 2.7), (4.5, 5.5, 2.7)],
           "빔프로젝터 위 + 뒤 천장 · 2.3m": [(4.5, 2.6, 2.3), (4.5, 5.6, 2.3)]})
CEIL = ("천장 가운데 한 곳 · 2.7m", "천장 가운데 두 곳 · 2.7m", "빔프로젝터 위 + 뒤 천장 · 2.3m")
CORNER_ONLY = ("앞 두 모서리 · 천장 밑 2.6m", "앞 둘 + 뒤 하나 모서리 · 2.6m", "네 모서리 · 2.6m", "네 모서리 · 1.9m", "네 모서리 + 옆벽 가운데 · 2.6m")
AD_H = 2.0


class ClassWorld(RoomWorld):
    """RoomWorld 에 앉은 사람 여럿(기둥, 키가 저마다)을 넣는다. people: None 이면 아무도 없는 교실"""
    def __init__(self, rng, people, **kw):
        super().__init__(rng, "교실", **kw)
        self.people = people

    def paths(self, ant, person=None):
        amp, d, r, blk = super().paths(ant, None)
        if self.people is None:
            return amp, d, r, blk
        v = self.img - ant; b = self.block_many(ant, v, r, d)
        return amp * b, d, r, b

    def block_many(self, ant, v, r, d):
        Lx, Ly, Lz = self.L; o = 1                            # 사람의 거울상은 벽에 한 번 비친 것까지 (두 번 넘게 비친 길은 이미 약하다)
        ii = np.arange(-o, o + 1)
        P = self.people; out = np.ones(len(r), dtype=complex)
        px = (ii[:, None] * Lx + np.where(ii[:, None] % 2 != 0, Lx - P[None, :, 0], P[None, :, 0]))     # (거울, 사람)
        py = (ii[:, None] * Ly + np.where(ii[:, None] % 2 != 0, Ly - P[None, :, 1], P[None, :, 1]))
        PX = np.repeat(px, len(ii), axis=0).ravel(); PY = np.tile(py, (len(ii), 1)).ravel()
        H = np.tile(P[:, 2], len(ii) * len(ii)); R_ = np.tile(P[:, 3], len(ii) * len(ii))
        for p in range(len(r)):
            h = np.hypot(d[p, 0], d[p, 1])
            if h < 1e-6:
                continue
            ux, uy = d[p, 0] / h, d[p, 1] / h
            s = ((PX - ant[0]) * ux + (PY - ant[1]) * uy) / h
            ok = (s > 0.02) & (s < r[p] - 0.02)
            if not ok.any():
                continue
            b = -(PX - ant[0]) * uy + (PY - ant[1]) * ux
            z = fold(ant[2] + s * d[p, 2], Lz)
            ok &= (np.abs(b) < R_ + 0.6) & (z < H + 0.05)
            for q in np.nonzero(ok)[0]:
                d2 = s[q]; d1 = r[p] - s[q]
                kap = np.sqrt(2 * (d1 + d2) / (LAM_M * d1 * d2))
                out[p] *= strip_factor((b[q] - R_[q]) * kap, (b[q] + R_[q]) * kap)
        return out


def place(rng, seat, where):
    """카메라 자리와 주인의 몸 (x, y)"""
    sx, sy = SEATS[seat]; body = np.array([sx + rng.uniform(-0.1, 0.1), sy + 0.25])
    if where == "책상 위":
        cam = np.array([sx + rng.uniform(-0.3, 0.3), sy + rng.uniform(-0.3, -0.05), rng.uniform(0.75, 0.9)])
    else:                                                      # 가슴 (몸 앞 3~5cm)
        cam = np.array([body[0] + rng.uniform(-0.08, 0.08), body[1] - BODY_R - rng.uniform(0.03, 0.05), rng.uniform(1.1, 1.25)])
    return cam, body


class SeatGrid:
    """자리마다 카메라가 있을 법한 점들 (책상 위 여러 점 + 가슴 여러 점)"""
    def __init__(self):
        pts, owner = [], []
        for k, (sx, sy) in enumerate(SEATS):
            for dx in (-0.25, 0.0, 0.25):
                for dy in (-0.25, -0.1):
                    for z in (0.8,):
                        pts.append((sx + dx, sy + dy, z)); owner.append(k)
                pts.append((sx + dx * 0.3, sy + 0.25 - BODY_R - 0.04, 1.17)); owner.append(k)
        self.X = np.array(pts); self.owner = np.array(owner)


def pick(grid, ms):
    """자리마다의 확률 (자리 안 점들의 합)"""
    post = K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE)
    p = np.bincount(grid.owner, post, minlength=len(SEATS)); return p / p.sum()


def rc(k):
    return divmod(int(k), COLS)


def run(n=150, rxname="앞 두 모서리 · 천장 밑 2.6m", where="책상 위", people=True, seed=5, plan="tilt3"):
    fus = Fusion(); frames = PLANS[plan]; grid = SeatGrid(); rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(SEATS)))
        cam, body = place(rng, seat, where)
        P = None
        if people:
            P = []
            for k, (sx, sy) in enumerate(SEATS):
                bx, by = (body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25))
                P.append((bx, by, BODY_H + rng.uniform(-0.1, 0.1), BODY_R + rng.uniform(-0.03, 0.03)))
            P = np.array(P)
        ms = []; errs = []; w0 = None
        def at(base):
            nonlocal w0
            w = ClassWorld(rng, P, cam=cam, base=np.array(base, float))
            if w0 is None:
                w0 = w
            else:
                w.absorb = w0.absorb                           # 같은 교실
            m = measure(w, fus, rng, frames); ms.append(m); errs.append(m["err"])
        if RX[rxname] == "adaptive":
            at((4.5, 0.5, AD_H))
            b = int(np.argmax(pick(grid, ms))); bx, by = SEATS[b]
            at((0.35 if bx < 4.5 else 8.65, by, AD_H))                        # 그 줄의 옆 벽 통로 (가까운 쪽)
            b = int(np.argmax(pick(grid, ms))); bx, by = SEATS[b]
            at((bx, min(by + 1.6, 7.0), AD_H))                                # 그 자리 뒤 (뒤로 1.6m, 뒷벽 넘지 않게)
        elif RX[rxname] == "close":
            at((0.6, 0.6, 2.6)); at((8.4, 0.6, 2.6))
            b = int(np.argmax(pick(grid, ms))); bx, by = SEATS[b]
            at((bx, min(by + 1.0, 7.0), 1.8))                                 # 그 자리 뒤 1m
            b = int(np.argmax(pick(grid, ms))); bx, by = SEATS[b]
            sx = 1.5 if bx < 4.5 else -1.5
            at((bx + sx, min(by + 0.55, 7.0), 1.8))                           # 옆뒤 대각 (두 줄 사이 통로)
        else:
            for base in RX[rxname]:
                at(base)
        row = dict(seat=seat, err=errs, ok=[m["ok"] for m in ms])
        for tag in [str(k) for k in range(1, len(ms) + 1)]:
            use = ms[:int(tag)]
            if len(use) < int(tag):
                continue
            p = pick(grid, use); order = np.argsort(-p); best = int(order[0])
            (r0, c0), (r1, c1) = rc(seat), rc(best)
            row["best_" + tag] = best; row["p_" + tag] = float(p[best])
            row["exact_" + tag] = best == seat
            row["near_" + tag] = abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1          # 바로 옆·앞·뒤·대각 자리까지
            row["top3_" + tag] = seat in order[:3]
            row["dist_" + tag] = float(np.linalg.norm(SEATS[best] - SEATS[seat]))
            row["n50_" + tag] = int(np.searchsorted(np.cumsum(p[order]), 0.5) + 1)       # 확률 50% 를 채우는 데 필요한 자리 수 (후보가 몇 자리로 좁혀졌나)
        rows.append(row)
    return rows


def summ(rows):
    out = {}
    for tag in ("1", "2", "3", "4", "5", "6"):
        if not all(("exact_" + tag) in r for r in rows):
            continue
        out[tag] = dict(exact=round(100 * np.mean([r["exact_" + tag] for r in rows]), 1), near=round(100 * np.mean([r["near_" + tag] for r in rows]), 1),
                        top3=round(100 * np.mean([r["top3_" + tag] for r in rows]), 1), dist_med=round(float(np.median([r["dist_" + tag] for r in rows])), 2),
                        n50_med=float(np.median([r["n50_" + tag] for r in rows])))
    out["dir_med"] = [round(float(np.median([r["err"][j] for r in rows])), 1) for j in range(len(rows[0]["err"]))]
    last = max((k for k in out if k in ("1", "2", "3", "4", "5", "6")), key=int)
    out["n"] = len(rows)
    # 줄마다 (앞줄 -> 뒷줄): 두 자리로 정확히 맞힌 비율
    out["by_row"] = [round(100 * np.mean([r["near_" + last] for r in rows if rc(r["seat"])[0] == q]), 0) if any(rc(r["seat"])[0] == q for r in rows) else None for q in range(ROWS)]
    return out


def main(n=150):
    res = {}
    # 시험 중에는 자리 사이로 들고 다닐 수 없다: 모서리·벽·천장에 미리 두는 것만 (가까이 가서 재는 것은 견줌용)
    conds = [(rx, wh, True) for rx in ("앞 두 모서리 · 교탁 높이 0.75m",) + CORNER_ONLY + CEIL + ("모서리 2 + 가까이 2 (옮겨 가며)",) for wh in ("책상 위", "가슴")] + \
            [(rx, "책상 위", False) for rx in ("네 모서리 · 2.6m", "천장 가운데 한 곳 · 2.7m")]
    for rx, wh, ppl in conds:
        key = "%s | %s | %s" % (rx, wh, "40명" if ppl else "빈 교실")
        t0 = time.time(); s = summ(run(n, rx, wh, ppl)); res[key] = s
        last = max((k for k in s if k in ("1", "2", "3", "4", "5", "6")), key=int)
        f = s.get("2", s["1"])
        print("%-44s 두 자리: 정확 %4.1f%% 옆자리까지 %4.1f%% | 마지막(%s): 정확 %4.1f%% 옆자리까지 %4.1f%% 3후보 안 %4.1f%% 50%%후보 %2.0f자리 | 방향 %s | 줄별(옆자리까지) %s (%.0f초)" % (
            key, f["exact"], f["near"], last, s[last]["exact"], s[last]["near"], s[last]["top3"], s[last]["n50_med"], s["dir_med"], s["by_row"], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=res), open(os.path.join(HERE, "classroom_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
