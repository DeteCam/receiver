# -*- coding: utf-8 -*-
"""지금 판(2.4 GHz 용)으로 5 GHz 기기의 자리를 재면 어떻게 되나.   python3 sim/pre/band5_check.py  ->  sim/pre/band5_check.json

firmware/core/panel.py 의 계산(demod, bearing, score, Session)을 그대로 쓰고, 받는 전파만 5 GHz 로 바꾼다.
지금 panel.py 는 판의 대역 밖 주파수를 받지 않으므로(can_locate, score 의 band) 여기서는 그 막음을 끄고(band=None) 잰다.
안테나·스위치·합침기는 5 GHz 에서도 2.4 GHz 처럼 잘 받는다고 '좋게' 친다. 실물의 CHW1010-ANT2 는 패치와 16갈래 스위치가
2400~2483 MHz 용이라 5 GHz 에서는 더 나쁘다 (이 계산에 넣지 못한 것). 그래서 여기 숫자는 '가장 좋게 쳐도'의 숫자다.

견주는 것 (같은 교실, 같은 카메라 자리, 직접파 + 반사 둘):
  막음 없이   : 허브가 늘 하던 대로 잰다. 위상 단계는 2.4 GHz 에서 잰 값(0·90·180·270), 방향 계산은 2.437 GHz
  채널만 맞춤 : 방향 계산의 주파수는 그 채널로 맞추지만, 위상 단계는 2.4 GHz 에서 잰 값 그대로
  가장 좋게   : 위상 단계도 그 채널의 참값을 안다고 친다 (선 길이 0·21·42·63 mm, RG316)
결과 (band5_check.json 의 rows): bearing_med/p90 = 패널 하나의 방향 오차 가운데값/90% (도), over20 = 20도 넘게 틀린 비율 (%),
rho_med = 그 방향이 설명하는 세기의 비율 (믿음), seat_exact/seat_top3 = 패널 둘·전송 넷으로 자리를 정확히/후보 셋 안에 (%),
seat_far_med_m = 1등 자리와 참 자리의 거리 가운데값 (m), seat_p1_med = 앱이 1등 자리에 붙이는 확률의 가운데값 (%).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware", "core"))
import numpy as np
import panel as P

VF = 0.695                                   # RG316 의 속도 비율
DL = np.array([0.0, 21.0, 42.0, 63.0]) * 1e-3  # 지금 위상 선의 길이 차 (부품표: 21 mm 씩)
CH = [("2.4 GHz 6번", 2.437e9), ("5 GHz 36번", 5.18e9), ("5 GHz 100번", 5.50e9), ("5 GHz 149번", 5.745e9), ("5 GHz 161번", 5.805e9)]
F = 52; R = P.grid(); rng = np.random.RandomState(11)


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def steps_at(f):
    """선 길이가 주는 위상 단계 (도)"""
    return (360.0 * DL * f / (VF * P.C0)) % 360.0


def field(fk, spot, bore, paths):
    ex, ey, ez = P.axes(bore); Rw = R[:, :1] * ex + R[:, 1:2] * ey + R[:, 2:3] * ez; ref = np.array([0.0, 0.0, 0.09])
    h = np.zeros((len(R), F), complex); r = np.zeros(F, complex)
    for amp, d, tau in paths:
        g = max(d.dot(ey), 0.0); base = amp * g * np.exp(-2j * np.pi * fk * tau)
        h += base[None, :] * np.exp(2j * np.pi * Rw.dot(d)[:, None] * fk[None, :] / P.C0); r += base * np.exp(2j * np.pi * (ref[0] * ex + ref[2] * ez).dot(d) * fk / P.C0)
    return h, r


def measure(h, r, fk, snr=30.0, jit=0.5):
    """세기만 잰다. 위상 단계는 선 길이로 정해지고, 서브캐리어마다 조금씩 다르다"""
    na = np.sqrt(np.median(np.abs(r) ** 2) * 10 ** (-snr / 10.0) / 2.0); I = np.zeros((len(h), 4, F))
    ph = 2 * np.pi * DL[:, None] * fk[None, :] / (VF * P.C0)          # [단계, 서브캐리어]
    for m in range(len(h)):
        for i in range(4):
            y = r + np.exp(1j * ph[i]) * h[m] + na * (rng.randn(F) + 1j * rng.randn(F)); I[m, i] = np.abs(y) ** 2 * 10 ** (rng.normal(0, jit) / 10.0)
    return I


def paths_for(c, sp, blocked=False):
    d = unit(c - sp)
    return [(0.15 if blocked else 1.0, d, 12e-9), (0.45 * np.exp(1j * rng.uniform(0, 6.28)), unit(d * [1, 1, -1] + [0, 0, 0.9]), 16e-9),
            (0.35 * np.exp(1j * rng.uniform(0, 6.28)), unit(d * [1, -1, 1] + [0, 0.4, 0]), 21e-9)]


WAYS = [("막음 없이", lambda f: (P.STEPS, 2.437e9)), ("채널만 맞춤", lambda f: (P.STEPS, f)), ("가장 좋게", lambda f: (steps_at(f), f))]

# 교실 (test_panel.py 와 같은 40석 + 패널 둘)
xs = 0.625 + 1.1 * np.arange(6); ys = 2.0 + 0.95 * np.arange(7); seats = [dict(x=float(x), y=float(y) + 0.225) for y in ys for x in xs]; S = P.Seats(seats)
SPOTS = [(np.array([0.08, 4.85, 2.0]), unit([1.0, 0.0, -0.27])), (np.array([7.42, 2.5, 1.95]), unit([-1.0, 0.55, -0.27]))]
SXY = np.array([[s["x"], s["y"]] for s in seats])

out = dict(made="2026-11-18", note=__doc__.split("\n\n")[1].replace("\n", " "), steps={}, rows=[])
for name, f in CH:
    fk = f + (np.arange(F) - F / 2.0) * 312.5e3; out["steps"][name] = [round(float(x), 1) for x in steps_at(f)]
    for way, pick in WAYS:
        if f < 3e9 and way != "막음 없이":
            continue                                                        # 2.4 GHz 에서는 셋이 같다
        st, fc = pick(f)
        # 1) 방향 (패널 하나, 카메라 120곳)
        errs, rhos = [], []
        sp, bo = SPOTS[0]
        for k in range(120):
            c = np.array([rng.uniform(0.6, 6.2), rng.uniform(2.0, 7.8), 0.8]); h, r = field(fk, sp, bo, paths_for(c, sp))
            b, s = P.bearing(P.demod(measure(h, r, fk), steps=st), R, bo, sp, fc=fc, band=None); errs.append(ang(b, unit(c - sp))); rhos.append(1 - np.exp(-s / 12.0))
        # 2) 자리 (패널 둘, 전송 넷, 한 번은 한쪽이 가려짐)
        hit1 = hit3 = 0; far = []; p1s = []; N = 80
        for t in range(N):
            kk = rng.randint(len(seats)); c = np.array([seats[kk]["x"] - 0.15, seats[kk]["y"] - 0.27, 0.8]); ses = P.Session(S)
            for shot in range(4):
                for j, (sp2, bo2) in enumerate(SPOTS):
                    h, r = field(fk, sp2, bo2, paths_for(c, sp2, blocked=(shot == 1 and j == 1)))
                    ses.add(P.score(P.demod(measure(h, r, fk), steps=st), R, bo2, sp2, S.X, fc=fc, band=None))
            top = ses.top(3); hit1 += top[0][0] == kk; hit3 += kk in [i for i, _ in top]; far.append(float(np.linalg.norm(SXY[top[0][0]] - SXY[kk]))); p1s.append(top[0][1])
        row = dict(ch=name, way=way, steps_used=[round(float(x), 1) for x in st], fc_used=fc,
                   bearing_med=round(float(np.median(errs)), 1), bearing_p90=round(float(np.percentile(errs, 90)), 1), over20=round(100.0 * np.mean(np.array(errs) > 20), 1),
                   rho_med=round(float(np.median(rhos)), 3), seat_exact=round(100.0 * hit1 / N, 1), seat_top3=round(100.0 * hit3 / N, 1), seat_far_med_m=round(float(np.median(far)), 2), seat_p1_med=round(100.0 * float(np.median(p1s)), 1))
        out["rows"].append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "band5_check.json"), "w"), ensure_ascii=False, indent=1)
