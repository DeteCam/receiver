"""세기만 재는 값싼 수신기 여러 개로 자리를 맞히면 얼마나 되나 (가장 싼 방법의 한계).
    python rt_rss.py [--sets kr40_t_s1 ...] [--n 560]

수신기(ESP32 한 개씩)는 카메라 신호의 세기만 안다 (20MHz 서브캐리어 평균, 세로 안테나, 수신기마다 1dB 의 보정 오차).
카메라가 얼마나 세게 내는지는 모르므로 수신기들 사이의 차이만 쓴다. 모형 두 가지를 견준다.
  (가) 거리만으로 셈한 세기 (사람이 가리는 것은 모른다)
  (나) 그 교실을 미리 재 둔 값이 있을 때: 다른 길 묶음(사람의 자세가 다른 날)에서 얻은 세기. 실제로는 얻기 어려운 위쪽 한계
"""
import os, sys, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np                                      # noqa: E402
import rt_study as S                                    # noqa: E402
from rt_common import KIND, seat_rc                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = {"2개 (두 옆벽 가운데)": [7, 11], "4개 (두 옆벽 앞·뒤)": [6, 8, 17, 18], "6개": [6, 8, 17, 18, 1, 16], "8개": [6, 7, 8, 17, 11, 18, 1, 16], "12개": [6, 7, 8, 17, 11, 18, 1, 16, 0, 2, 13, 15], "20개 (둘레 전부)": list(range(20))}
EZ = np.array([0, 0, 1.0])


def power_db(D, s, c, u):
    P = S.Paths(D, s, c, u)
    if not len(P.tau):
        return -200.0
    pol = P.cj[:, 0] * (P.th @ EZ) + P.cj[:, 1] * (P.ph @ EZ); H = (pol[:, None] * np.exp(-2j * np.pi * P.tau[:, None] * P.fk[None, :])).sum(0)
    return 10 * np.log10(np.mean(np.abs(H) ** 2) + 1e-30)


def _table(name):
    """한 길 묶음의 (재서 얻는 세기 [곳, 점] : 편파를 아무렇게나 한 번), (미리 재 둔 값 [곳, 점] : 편파 세 가지의 평균)"""
    d = np.load(os.path.join(HERE, "paths", name + ".npz")); D = {k: d[k] for k in d.files if k not in ("meta", "people")}
    ns, nc = len(D["spots"]), len(D["cams"]); rng = np.random.default_rng(abs(hash(name)) % (1 << 31)); M = np.zeros((ns, nc)); F = np.zeros((ns, nc))
    for c in range(nc):
        us = rng.normal(size=(4, 3)); us /= np.linalg.norm(us, axis=1, keepdims=True)
        for s in range(ns):
            M[s, c] = power_db(D, s, c, us[0]); F[s, c] = 10 * np.log10(np.mean([10 ** (power_db(D, s, c, u) / 10) for u in us[1:]]))
    return M, F, D["spots"].astype(float), D["cams"].astype(float), D["seats"]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", default=["kr40_t_s%d" % i for i in range(1, 9)]); ap.add_argument("--n", type=int, default=560); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    with mp.Pool(min(a.jobs, len(a.sets))) as pool:
        T = pool.map(_table, a.sets)
    ns = len(a.sets); spots, seats = T[0][2], T[0][4]; nseat = len(seats); rc = np.array([seat_rc(seats[k], seats) for k in range(nseat)])
    near = (np.abs(rc[:, None, 0] - rc[None, :, 0]) <= 1) & (np.abs(rc[:, None, 1] - rc[None, :, 1]) <= 1)
    FS = [-20 * np.log10(np.linalg.norm(T[j][3][None, :, :] - spots[:, None, :], axis=2)) for j in range(ns)]                 # (가) 거리만
    FP = [10 * np.log10(np.mean([10 ** (T[i][1] / 10) for i in range(ns) if i != j], 0)) for j in range(ns)]                   # (나) 다른 날들의 평균
    rng = np.random.default_rng(7); kinds = ["책상", "가슴", "안경"]
    print("세기만 재는 수신기 (높이 약 2m 의 벽·창), 교실 40석. 정확히 / 옆자리까지 / 후보 3자리 (%)")
    for mname, MODEL in (("(가) 거리만으로 셈한 세기", FS), ("(나) 미리 재 둔 값이 있을 때 (위쪽 한계)", FP)):
        for K in (1, 4):
            print("\n-- %s, 전송 %d번 평균" % (mname, K))
            for nn, nodes in NODES.items():
                row = []
                for kind in kinds:
                    hit = np.zeros(3); tot = 0
                    for t in range(a.n):
                        seat = int(rng.integers(nseat)); ci = int(rng.choice(KIND[kind])); js = rng.choice(ns, K, replace=False); goff = rng.normal(0, 1.0, len(nodes))
                        m = np.mean([T[j][0][nodes, seat * 4 + ci] for j in js], 0) + goff + rng.normal(0, 0.5, len(nodes))
                        cand = MODEL[js[0]][nodes][:, np.arange(nseat) * 4 + ci]                                        # [곳, 자리]: 같은 종류의 점
                        r = m[:, None] - cand; r = r - r.mean(0, keepdims=True); score = -(r ** 2).sum(0); order = np.argsort(-score)
                        hit += (order[0] == seat, near[seat, order[0]], seat in order[:3]); tot += 1
                    row.append("%s %3.0f/%3.0f/%3.0f" % (kind, *(100 * hit / tot)))
                print("%-22s " % nn + " | ".join(row))


if __name__ == "__main__":
    main()
