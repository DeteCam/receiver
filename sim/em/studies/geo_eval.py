"""판 모양·크기 x 안테나 모형 4가지: 무늬 맞추기(맞는 무늬를 알 때)로 방향 오차를 견준다. 결과는 geo_eval.json 에 쌓인다 (이미 한 것은 건너뛴다)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json, time
import numpy as np
import device as V, study as S, tmatch as T
import run_tables as RT
OUT = os.path.join(empaths.STUDIES, "geo_eval.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
N = int(os.environ.get("GEO_N", "150"))
names = sys.argv[1:] or [j[0] for j in RT.JOBS["geo"]]
for nm in names:
    if not os.path.exists(os.path.join(V.CACHE, nm + ".npz")):
        continue
    R = S.Resp(nm); Tm = None
    for env, mp, lab in (("보통", False, "반사없음"), ("보통", True, "보통"), ("나쁨", True, "나쁨")):
        key = "%s|%s" % (nm, lab)
        if key in res:
            continue
        Tm = Tm or T.Templates(R); t0 = time.time()
        s = T.run(R, Tm, n=N, env=env, multipath=mp); res[key] = {k: v for k, v in s.items() if k != "margin"}
        print(T.line(key, s), "(%.0f초)" % (time.time() - t0), flush=True)
        json.dump(res, open(OUT, "w"), ensure_ascii=False)
