"""모형이 틀렸을 때: 세상(실제 전파)은 A 표, 계산에 쓰는 무늬는 B 표.
    python3 mismatch.py 세상:무늬 [세상:무늬 ...]      ->  mismatch.json 에 쌓인다"""
import os, sys, time, json
import numpy as np
import study as S, tmatch as T
pairs = [a.split(":") for a in sys.argv[1:]]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mismatch.json")
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for wn, tn in pairs:
    R = S.Resp(wn); Tm = T.Templates(S.Resp(tn)); t0 = time.time()
    for mp, lab in ((False, "반사 없음"), (True, "보통 방")):
        s = T.run(R, Tm, n=100, multipath=mp); out["%s|%s|%s" % (wn, tn, lab)] = {k: v for k, v in s.items() if k not in ("err", "margin")}
        print(T.line("세상 %s, 무늬 %s | %s" % (wn, tn, lab), s), flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
