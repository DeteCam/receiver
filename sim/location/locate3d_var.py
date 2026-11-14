"""locate3d 의 설정을 바꿔 가며 견준다: 시그마 배율, 통째로 틀릴 확률 q, 구간을 자르는 높이, 표면 정보를 얼마나 믿을지, 한 자리에서 3번/6번.
    python3 locate3d_var.py [횟수]      결과: locate3d_var.json"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import locate3d as L3

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = {
    "기본 (q 0.15, 시그마 1배, 자르는 높이 13.5%)": dict(),
    "시그마 1.5배": dict(sig=1.5),
    "시그마 2배": dict(sig=2.0),
    "q 0.30": dict(q=0.30),
    "자르는 높이 5%": dict(cut=0.05),
    "자르는 높이 2%": dict(cut=0.02),
    "시그마 1.5배 + 자르는 높이 5%": dict(sig=1.5, cut=0.05),
    "표면 칸만": dict(surf=1.0),
    "표면 아닌 칸 0.3배": dict(surf=0.3),
    "표면 아닌 칸 0.1배": dict(surf=0.1),
    "시그마 1.5배 + 표면 아닌 칸 0.3배": dict(sig=1.5, surf=0.3),
}
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    path = os.path.join(HERE, "locate3d_var.json"); out = {}
    for rm, where, shape, plan in (("보통 방", "inside", "W", "tilt3"), ("보통 방", "inside", "V", "tilt3"), ("보통 방", "air", "W", "tilt3"), ("보통 방", "wall", "W", "tilt3"),
                                   ("보통 방", "inside", "W", "tilt6"), ("보통 방", "inside", "V", "tilt6"), ("화장실", "inside", "W", "tilt3")):
        t0 = time.time(); key = "%s|%s|%s|%s" % (rm, where, shape, plan)
        S = L3.run_variants(VARS, n, rm, shape, where, plan); out[key] = S
        print("== %s  (%.0f초)" % (key, time.time() - t0))
        for name, v in S.items():
            print("   %-34s " % name + " | ".join("%d자리: 가운데값 %.2fm 50cm안 %4.1f%% 구간 %.2fm %3.0fL 구간안 %4.1f%% 믿을만 %4.1f%% (틀림 %.1f%%)" % (
                k, r["med"], r["in50"], r["zone_len"], r["zone_vol"], r["inside"], r["good_share"], r["good_miss"]) for k, r in v.items()), flush=True)
        json.dump(dict(n=n, variants={k: v for k, v in VARS.items()}, summary=out), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
