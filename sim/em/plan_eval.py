"""자세 수의 효과: 세 자세(받침 -120, 0, 120) 와 여섯 자세(-150 ... 150). 요크·받침까지 넣은 전파 표, 맞는 무늬."""
import json, time
import numpy as np
import study as S, tmatch as T
out = {}
Y3 = (-120.0, 0.0, 120.0); Y6 = (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0)
f3 = {y: S.Resp("full_y%+04d" % y) for y in Y3}; t3 = {y: T.Templates(f3[y]) for y in Y3}
f6 = {y: S.Resp("full6_y%+04d" % y) for y in Y6}; t6 = {y: T.Templates(f6[y]) for y in Y6}
for lab, world, tm, plan in (("세 자세", f3, t3, S.PLAN3), ("여섯 자세", f6, t6, S.PLAN6)):
    for env in ("보통", "나쁨"):
        t0 = time.time(); s = T.run(world, tm, n=200, env=env, plan=plan); out["%s|%s" % (lab, env)] = {k: v for k, v in s.items() if k != "margin"}
        print(T.line("%s | %s 방" % (lab, env), s), "(%.0f초)" % (time.time() - t0), flush=True)
json.dump(out, open("plan_eval.json", "w"), ensure_ascii=False)
