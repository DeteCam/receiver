# -*- coding: utf-8 -*-
"""패널 계산의 검사 (장치 없이).  python3 tests/test_panel.py
1) 세기 네 번에서 '기준의 켤레 x 안테나'가 나오는가  2) 위상 단계가 틀어져 있어도 스스로 재서 맞추는가
3) 직접파 + 반사파에서 방향 오차  4) 패널 둘 + 전송 여러 번으로 자리를 맞히는가"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import numpy as np
import panel as P

rng = np.random.RandomState(7); F = 52; fk = 2.437e9 + (np.arange(F) - F / 2.0) * 312.5e3; R = P.grid()


def field(src, spot, bore, paths):
    """안테나마다의 채널 [M,F] 과 기준 안테나의 채널 [F]. paths = [(세기, 방향 단위 벡터, 늦는 시간)]"""
    ex, ey, ez = P.axes(bore); Rw = R[:, :1] * ex + R[:, 1:2] * ey + R[:, 2:3] * ez; ref = np.array([0.0, 0.0, 0.09])
    h = np.zeros((len(R), F), complex); r = np.zeros(F, complex)
    for amp, d, tau in paths:
        g = max(d.dot(ey), 0.0); base = amp * g * np.exp(-2j * np.pi * fk * tau)
        h += base[None, :] * np.exp(2j * np.pi * Rw.dot(d)[:, None] * fk[None, :] / P.C0); r += base * np.exp(2j * np.pi * (ref[0] * ex + ref[2] * ez).dot(d) * fk / P.C0)
    return h, r


def measure(h, r, steps=P.STEPS, snr=30.0, jit=0.5):
    """세기만 잰다: 안테나마다 위상 단계 네 번. 패킷마다 0.5dB 흔들림과 잡음"""
    na = np.sqrt(np.median(np.abs(r) ** 2) * 10 ** (-snr / 10.0) / 2.0); I = np.zeros((len(h), len(steps), F))
    for m in range(len(h)):
        for i, p in enumerate(np.radians(steps)):
            y = r + np.exp(1j * p) * h[m] + na * (rng.randn(F) + 1j * rng.randn(F)); I[m, i] = np.abs(y) ** 2 * 10 ** (rng.normal(0, jit) / 10.0)
    return I


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a.dot(b), -1, 1))))


# 1) 복원
spot = np.array([0.08, 4.85, 2.0]); bore = unit([1.0, 0.0, -0.27]); cam = np.array([4.0, 5.6, 0.8]); d0 = unit(cam - spot)
h, r = field(cam, spot, bore, [(1.0, d0, 13e-9)]); Z = P.demod(measure(h, r, snr=60.0, jit=0.0)); Zt = np.conj(r)[None, :] * h
e1 = float(np.abs(Z - Zt).max() / np.abs(Zt).max()); assert e1 < 0.02, e1
# 2) 위상 단계가 틀어진 위상기 (0, 97, 171, 262도) 를 스스로 잰다
true = np.array([0.0, 97.0, 171.0, 262.0]); Ic = np.concatenate([measure(*field(c, spot, bore, [(1.0, unit(c - spot), 10e-9)]), steps=true, snr=40.0, jit=0.1) for c in (cam, np.array([2.0, 2.5, 1.0]), np.array([6.0, 7.5, 0.9]))])
est = P.calibrate_steps(Ic); e2 = float(np.abs((est - true + 180) % 360 - 180).max()); assert e2 < 5.0, (est, e2)
# 3) 방향: 직접파만 / 직접파 + 약한 반사 둘
errs = []
for k in range(40):
    c = np.array([rng.uniform(0.6, 6.2), rng.uniform(2.0, 7.8), 0.8]); d = unit(c - spot)
    paths = [(1.0, d, 12e-9), (0.45 * np.exp(1j * rng.uniform(0, 6.28)), unit(d * [1, 1, -1] + [0, 0, 0.9]), 16e-9), (0.35 * np.exp(1j * rng.uniform(0, 6.28)), unit(d * [1, -1, 1] + [0, 0.4, 0]), 21e-9)]
    hh, rr = field(c, spot, bore, paths); b, _ = P.bearing(P.demod(measure(hh, rr)), R, bore, spot); errs.append(ang(b, d))
e3 = float(np.median(errs)); assert e3 < 6.0, e3
# 4) 자리: 패널 둘 (마주 보는 옆벽), 전송 세 번. 한 번은 한쪽 패널이 가려져 엉뚱한 방향을 본다
xs = 0.625 + 1.1 * np.arange(6); ys = 2.0 + 0.95 * np.arange(7); seats = [dict(x=float(x), y=float(y) + 0.225) for y in ys for x in xs]; S = P.Seats(seats)
spots = [(np.array([0.08, 4.85, 2.0]), unit([1.0, 0.0, -0.27])), (np.array([7.42, 2.5, 1.95]), unit([-1.0, 0.55, -0.27]))]; ok = 0; N = 30
for t in range(N):
    k = rng.randint(len(seats)); c = np.array([seats[k]["x"] - 0.15, seats[k]["y"] - 0.27, 0.8]); ses = P.Session(S)
    for shot in range(3):
        for j, (sp, bo) in enumerate(spots):
            d = unit(c - sp); blocked = (shot == 1 and j == 1)
            paths = [(0.15 if blocked else 1.0, d, 12e-9), (0.5 * np.exp(1j * rng.uniform(0, 6.28)), unit(d * [1, 1, -1] + [0, 0, 1.2]), 17e-9)]
            hh, rr = field(c, sp, bo, paths); ses.add(P.score(P.demod(measure(hh, rr)), R, bo, sp, S.X))
    ok += ses.top(1)[0][0] == k
assert ok >= 0.8 * N, ok
print("패널 계산: 복원 오차 %.3f, 위상 단계 오차 %.1f도, 방향 오차(가운데값) %.1f도, 자리 %d/%d" % (e1, e2, e3, ok, N))
print("OK")
