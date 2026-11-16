"""40명 교실에서 수신기 1대(또는 2대)를 어디에 두면 카메라 자리를 가장 잘 맞히나: 네 모서리, 벽 가운데, 천장 가운데, 그리고 교실 밖 복도.
    python3 rx_places.py [횟수]      결과: rx_places_result.json

교실·자리·사람·카메라와 자리 고르기는 classroom.py 와 같다. 다른 점:
  - 복도 쪽 벽(오른쪽, x = 9m): 콘크리트 200mm 에 앞문·뒷문(나무 40mm, 높이 2.1m)과 문 사이 위쪽 복도 창(유리 6mm, 높이 1.8~2.4m). 바깥 창은 왼쪽 벽.
  - 교실 밖 수신기: 복도에서 벽 0.6m 앞. 교실 안의 반사(거울상)를 거친 파가 복도 벽을 뚫고(콘크리트·문·창마다 다른 투과) 나온다.
    복도 자체의 반사는 넣지 않았다 (넣으면 더 나빠진다).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import EPS, gamma, fold, LAM_M
import classroom as C

HERE = os.path.dirname(os.path.abspath(__file__))
CORR = [dict(wall="x1", rect=(0.3, 1.2, 0.0, 2.1), mat="wood40"),           # 앞문
        dict(wall="x1", rect=(6.0, 6.9, 0.0, 2.1), mat="wood40"),           # 뒷문
        dict(wall="x1", rect=(1.5, 5.7, 1.8, 2.4), mat="glass6")]           # 복도 창 (위쪽)
room.ROOMS["교실"] = dict(room.ROOMS["교실"], patches=[p for p in room.ROOMS["교실"]["patches"] if p["wall"] != "x1"] +
                         [dict(wall="x0", rect=(1.0, 6.6, 0.9, 2.4), mat="glass6")] + CORR)
EPS["concrete200"] = ("slab", 5.31 - 0.49j, 200.0)


def slab_t(eps, cos_t, chi):
    """두께 있는 판을 뚫고 나가는 전기장 비 (TE/TM 섞음). eps: ('slab', 유전율, 두께 mm) 또는 'open'"""
    if isinstance(eps, str):
        return 1.0 + 0j
    _, e, d = eps
    root = np.sqrt(e - (1 - cos_t * cos_t)); P = np.exp(-2j * np.pi * (d / 1000.0) / LAM_M * root)
    def t(r):
        return (1 - r * r) * P / (1 - r * r * P * P)
    te = (cos_t - root) / (cos_t + root); tm = (e * cos_t - root) / (e * cos_t + root)
    return (np.cos(chi) ** 2 * t(tm) + np.sin(chi) ** 2 * t(te)) * np.exp(2j * np.pi * (d / 1000.0) / LAM_M * cos_t)


class World(C.ClassWorld):
    def paths(self, ant, person=None):
        ant = np.asarray(ant, float)
        if ant[0] <= self.L[0]:
            return super().paths(ant, person)
        Lx, Ly, Lz = self.L
        v = self.img - ant; r = np.linalg.norm(v, axis=1); d = v / r[:, None]
        amp = np.zeros(len(r), dtype=complex)
        names = (("x0", "x1"), ("y0", "y1"), ("z0", "z1"))
        k0 = None
        for p in range(len(r)):
            if self.idx[p, 0] > 0 or v[p, 0] >= 0:
                continue
            te = (Lx - ant[0]) / v[p, 0]; Q = ant + te * v[p]
            if not (0 < Q[1] < Ly and 0 < Q[2] < Lz):
                continue                                                   # 복도에서 벽에 닿기 전에 다른 면을 지난다 -> 그런 길은 없다
            if k0 is None:
                k0 = p
            a = np.exp(-2j * np.pi * r[p] / LAM_M) / r[p] * self.gain[p]
            mat = EPS["concrete200"]
            for pt in self.R["patches"]:
                a0, a1, b0, b1 = pt["rect"]
                if pt["wall"] == "x1" and a0 <= Q[1] <= a1 and b0 <= Q[2] <= b1:
                    mat = EPS[pt["mat"]]
            a *= slab_t(mat, abs(d[p, 0]), self.chi[p])
            for ax in range(3):
                L = self.L[ax]; o1, o2 = [q for q in range(3) if q != ax]; i = int(self.idx[p, ax])
                if i == 0:
                    continue
                for n in (range(1, i + 1) if i > 0 else range(i + 1, 1)):
                    t = (n * L - ant[ax]) / v[p, ax]; q = ant + t * v[p]
                    u1 = float(fold(q[o1], self.L[o1])); u2 = float(fold(q[o2], self.L[o2]))
                    wall = names[ax][n % 2]; m = self.mat[wall]; ab = self.absorb[wall]
                    for pt in self.R["patches"]:
                        a0, a1, b0, b1 = pt["rect"]
                        if pt["wall"] == wall and a0 <= u1 <= a1 and b0 <= u2 <= b1:
                            m = pt["mat"]; ab = 1.0
                    a *= gamma(EPS[m], abs(d[p, ax]), self.chi[p]) * ab
            amp[p] = a
        # 정규화: 맨 앞 (가장 짧은) 길의 1/r 이 1 이 되게. 직접파 자리(0번)에 가장 센 길을 둔다
        r0 = r[0]; amp = amp * r0 * np.exp(2j * np.pi * r0 / LAM_M)
        if self.people is not None:
            amp = amp * self.block_many(ant, v, r, d)
        return amp, d, r, np.ones(len(r), dtype=complex)


C.ClassWorld = World
H = 2.6
SINGLE = {
    "앞 왼쪽 모서리": (0.6, 0.6, H), "앞 오른쪽 모서리 (복도 쪽)": (8.4, 0.6, H), "뒤 왼쪽 모서리": (0.6, 6.6, H), "뒤 오른쪽 모서리 (복도 쪽)": (8.4, 6.6, H),
    "왼쪽 벽 가운데": (0.35, 3.6, H), "오른쪽 벽 가운데 (복도 쪽)": (8.65, 3.6, H), "뒷벽 가운데": (4.5, 6.95, H), "칠판 위 가운데": (4.5, 0.3, H),
    "천장 가운데": (4.5, 4.2, 2.7), "교탁 위 (칠판 앞 0.5m, 1.0m)": (4.5, 0.8, 1.0),
    "복도 · 앞문 앞 1.9m": (9.8, 0.8, 1.9), "복도 · 가운데 창 앞 2.2m": (9.8, 3.6, 2.2), "복도 · 뒷문 앞 1.9m": (9.8, 6.4, 1.9),
}
PAIRS = {"복도 · 앞문 + 뒷문": [(9.8, 0.8, 1.9), (9.8, 6.4, 1.9)], "복도 · 앞문 + 가운데 창 + 뒷문": [(9.8, 0.8, 1.9), (9.8, 3.6, 2.2), (9.8, 6.4, 1.9)],
         "안 · 네 모서리": [(0.6, 0.6, H), (8.4, 0.6, H), (0.6, 6.6, H), (8.4, 6.6, H)]}


def main(n=100):
    res = {}
    for name, pos in list(SINGLE.items()) + list(PAIRS.items()):
        C.RX[name] = [pos] if isinstance(pos, tuple) else pos
        for wh in ("책상 위", "가슴"):
            t0 = time.time(); s = C.summ(C.run(n, name, wh, True)); key = "%s | %s" % (name, wh); res[key] = s
            last = max((k for k in s if k.isdigit()), key=int)
            print("%-32s %-4s %s대: 정확 %4.1f%% 옆자리까지 %4.1f%% 3후보 %4.1f%% | 방향 %s (%.0f초)" % (name, wh, last, s[last]["exact"], s[last]["near"], s[last]["top3"], s["dir_med"], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=res), open(os.path.join(HERE, "rx_places_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
