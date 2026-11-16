"""5GHz 에서도 같은 판(100 x 150mm, 축에서 80mm)으로 방향을 찾을 수 있나.   python3 band5.py [횟수]   결과: band5.json, 그림 37_5GHz_골.png

판의 회절 모형(firmware/core/model.py)은 파장 하나로 계산한다. 그 파장만 2.4GHz 6번(123mm)에서 5GHz 36번(58mm)·149번(52mm)으로 바꾸고,
montecarlo.py 와 같은 흉내(반사파, 누설, 패킷 수, 관절 오차)를 그대로 돌린다. 판 표는 주파수마다 따로 만든다 (firmware/core/cache/f<MHz>/).
넣지 않은 것: 5GHz 가 벽·사람에 더 많이 막히는 것(거리가 줄어든다), 5GHz 에서 반사가 달라지는 것. 그래서 '같은 방이면 골은 어떻게 되나'까지만 본다."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import os, sys, json
import numpy as np
import model, world
import montecarlo as mc

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = (("2.4GHz 6번", 2.437e9), ("5GHz 36번", 5.18e9), ("5GHz 149번", 5.745e9))
CORE_CACHE = model.HERE


def use(f):
    """회절 모형의 파장을 바꾸고 판 표를 다시 읽는다"""
    model.LAM = 299792458.0 / f * 1000.0; model.K = 2 * np.pi / model.LAM
    model.HERE = CORE_CACHE if f < 3e9 else os.path.join(CORE_CACHE, "cache", "f%d" % int(round(f / 1e6)))
    world.PLATES = [model.Plate(dist=d) for d in (1500.0, 3000.0, 5000.0)]
    return world.PLATES


def trough_width(d, L, lo=3.0):
    """가장 깊은 골의 폭: 바닥에서 lo dB 안쪽인 각도 범위 (도)"""
    i = int(np.argmin(L)); m = L <= L[i] + lo; a = i; b = i
    while m[(a - 1) % len(L)] and b - a < len(L):
        a -= 1
    while m[(b + 1) % len(L)] and b - a < len(L):
        b += 1
    return float((b - a + 1) * (d[1] - d[0]))


def main(n=300):
    res = {}; curves = {}
    for name, f in BANDS:
        P = use(f)
        cams, out = mc.run(n=n, strategies=["세워서 1번", "기울여 3번"], quiet=True)
        S = mc.summarize(out)
        d = np.arange(-180.0, 180.0, 0.5); L = P[1].level(0.0, d); curves[name] = (d, L)
        res[name] = dict(lam_mm=round(model.LAM, 1), strategies=dict((k, v["전체"]) for k, v in S.items()),
                         trough=dict(min_db=round(float(L.min()), 1), at=round(float(abs(d[np.argmin(L)])), 1), width3db=round(trough_width(d, L), 1)))
        r = res[name]
        print("%-11s 파장 %5.1fmm  골 %5.1fdB, ±%4.1f도, 폭 %4.1f도 | 기울여 3번: 가운데값 %4.1f도, 30도 넘음 %4.1f%% | 세워서 1번: %4.1f도" % (
            name, r["lam_mm"], r["trough"]["min_db"], r["trough"]["at"], r["trough"]["width3db"],
            r["strategies"]["기울여 3번"]["med"], r["strategies"]["기울여 3번"]["over30"], r["strategies"]["세워서 1번"]["med"]))
    use(BANDS[0][1])
    json.dump(dict(n=n, bands=res), open(os.path.join(HERE, "band5.json"), "w"), ensure_ascii=False, indent=1)
    figure(curves, res)


def figure(curves, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures import style, head, SURF, C1, C2, C3, SUB
    fig = plt.figure(figsize=(12, 6.2), facecolor=SURF)
    head(fig, "5GHz 에서도 같은 판으로 골이 생긴다", "카메라 3m · 수평, 판 100 × 150mm. 파장이 짧아 골이 더 깊고 좁고 카메라 쪽에 가깝다")
    ax = fig.add_axes([0.07, 0.12, 0.6, 0.66]); style(ax, "판이 카메라 쪽에서 돌아간 각도 (도)", "판이 없을 때 대비 세기 (dB)")
    for (name, _), c in zip(BANDS, (C1, C2, C3)):
        d, L = curves[name]; ax.plot(d, L, color=c, lw=2, label=name)
    ax.set_xlim(-90, 90); ax.set_xticks(range(-90, 91, 30)); ax.legend(loc="lower left", frameon=False, fontsize=11)
    ax2 = fig.add_axes([0.74, 0.12, 0.23, 0.66]); style(ax2, None, "방향 오차가 30도 넘는 비율 (%)")
    names = [b[0] for b in BANDS]; v = [res[k]["strategies"]["기울여 3번"]["over30"] for k in names]
    ax2.bar(range(3), v, color=(C1, C2, C3), width=0.6)
    for i, x in enumerate(v):
        ax2.text(i, x + 0.3, "%.1f%%" % x, ha="center", color=SUB, fontsize=11)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([k.replace(" ", "\n") for k in names], fontsize=10.5); ax2.set_title("기울여 3번 재기", color=SUB, fontsize=12)
    fig.savefig(os.path.join(paths.IMG, "37_5GHz_골.png"), facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
