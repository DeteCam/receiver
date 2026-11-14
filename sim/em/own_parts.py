"""자기 부품의 영향: 세상 = 요크·받침·배터리·전선까지 다 있는 장치(받침 각마다 표가 다르다). 계산에 쓰는 무늬를 바꿔 가며 본다."""
import sys, json, time
import numpy as np
import study as S, tmatch as T
YAWS = (-120.0, 0.0, 120.0)
full = {y: S.Resp("full_y%+04d" % y) for y in YAWS}
tm_full = {y: T.Templates(full[y]) for y in YAWS}
cases = [("요크·받침까지 넣은 무늬 (받침 각마다)", tm_full)]
for nm, lab in (("head_cw", "머리만 넣은 무늬 (요크·받침 없음)"), ("head", "머리만, 균형추도 없음"), ("ant_conn", "기판과 단자만")):
    cases.append((lab, T.Templates(S.Resp(nm))))
out = {}
for lab, tm in cases:
    for mp, env in ((False, "반사 없음"), (True, "보통 방")):
        t0 = time.time(); s = T.run(full, tm, n=120, multipath=mp); out["%s|%s" % (lab, env)] = {k: v for k, v in s.items() if k not in ("err", "margin")}
        print(T.line("%s | %s" % (lab, env), s), flush=True)
json.dump(out, open("own_parts.json", "w"), ensure_ascii=False, indent=1)
