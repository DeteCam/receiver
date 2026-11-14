"""자리를 5번까지 늘리면 어디까지 좋아지는가, 구간에 여유를 두면 참 위치가 얼마나 들어오는가.   python3 locate3d_var2.py [횟수]   결과: locate3d_var2.json"""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import locate3d as L3

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = {"기본": dict(), "시그마 1.5배": dict(sig=1.5), "시그마 2배": dict(sig=2.0), "시그마 2배, 자르는 높이 5%": dict(sig=2.0, cut=0.05)}
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    path = os.path.join(HERE, "locate3d_var2.json"); out = {}
    for rm, where, shape in (("보통 방", "inside", "W"), ("보통 방", "inside", "V"), ("보통 방", "wall", "W"), ("화장실", "inside", "W")):
        t0 = time.time(); key = "%s|%s|%s" % (rm, where, shape)
        S = L3.run_variants(VARS, n, rm, shape, where, "tilt3", npos=5); out[key] = S
        print("== %s  (%.0f초)" % (key, time.time() - t0))
        for name, v in S.items():
            print("   " + name)
            for k, r in v.items():
                print("      %d자리: 가운데값 %.2fm, 50cm 안 %5.1f%%, 구간 길이 %.2fm %4.0fL, 참 위치가 구간 안 %5.1f%% / 15cm 여유 %5.1f%% / 25cm 여유 %5.1f%%, 믿을만 %5.1f%% (그중 틀림 = 전체의 %.1f%%)" % (
                    k, r["med"], r["in50"], r["zone_len"], r["zone_vol"], r["inside"], r["near15"], r["near25"], r["good_share"], r["good_miss"]), flush=True)
        json.dump(dict(n=n, variants=VARS, summary=out), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
