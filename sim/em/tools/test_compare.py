"""compare_real.py 가 제대로 고르는지: '세상'을 한 모형으로 꾸며 곡선을 만들고(반사·흔들림 포함), 그 곡선을 주면 그 모형을 1등으로 꼽는지 본다."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import json, os, sys, tempfile
import numpy as np
import study as S, compare_real as CR, world as W
from estimator import unit, frame

def fake_export(table, az, el, seed=3, env="보통", multipath=False):
    rng = np.random.default_rng(seed); c = unit(az, el); R = S.Resp(table); w = S.EMWorld(rng, c, R, env, multipath)
    curves = []
    for y, t in S.PLAN3:
        a, ex, ey = frame(y, t); phi = (np.arange(72) + 0.5) * 5.0; E = np.zeros(72, complex)
        for amp, d, pol in w.rays:
            E += amp * R.look(np.array([d @ ex, d @ ey, d @ a]), np.array([pol @ ex, pol @ ey, pol @ a]), phi)
        lv = 20 * np.log10(np.abs(E)) + rng.normal(0, 0.3, 72)
        curves.append(dict(yaw=y, pitch=t, bin_deg=5.0, lv=[round(float(v), 2) for v in lv]))
    return dict(positions=[dict(time="fake", az=None, el=None, curves=curves)])

for mp, title in ((False, "넓은 곳 (반사 없음): 실험을 이렇게 하라고 권한다"), (True, "보통 방 (반사 있음): 참고")):
    ok = 0; n = 0; print("==", title)
    for table, label in (("head", "기판 + 홈 안테나 + 머리 부품"), ("ideal_z", "이상적인 안테나 (전기)"), ("head_mono", "기판 + 막대 안테나 + 머리 부품")):
        for k, (az, el) in enumerate(((20.0, 10.0), (-100.0, 35.0), (150.0, -20.0))):
            ex = fake_export(table, az, el, seed=10 + k, multipath=mp)
            out = CR.compare(ex["positions"][0]["curves"], unit(az, el), quiet=True); n += 1
            good = out[0]["model"] == label and out[0]["err_deg"] < 10.0; ok += good
            print("세상 = %-10s 방향 (%4.0f, %3.0f): 1등 = %-30s 차이 %.2f dB, 방향 오차 %.1f도  %s" % (table, az, el, out[0]["model"], out[0]["resid_db"], out[0]["err_deg"], "" if good else "<== 다름"))
    print("맞게 고른 것 %d / %d" % (ok, n))
    if not mp:
        assert ok == n, "반사가 없는데도 모형을 못 골랐다"
d = tempfile.mkdtemp(); f = os.path.join(d, "export.json"); json.dump(fake_export("head", 20.0, 10.0), open(f, "w"))
os.system("%s %s %s --az 20 --el 10" % (sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_real.py"), f))
