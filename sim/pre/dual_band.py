# -*- coding: utf-8 -*-
"""두 대역(2.4 · 5 GHz)을 한 판으로 쓸 때의 위상 선 계산. 다른 방법 검토 5.5 의 숫자.

    python3 dual_band.py            -> dual_band.json

1) 위상 선(기준 길의 길이 차) 고르기
   세기 네 단계에서 Z = conj(r) h 를 푸는 식은 I = a + 2 Re(Z) cos p - 2 Im(Z) sin p 다 (firmware/core/panel.py 의 demod).
   단계 p 가 고르게 퍼질수록 잡음이 덜 커진다. 0·90·180·270 이면 Re·Im 의 분산이 1/8 (가장 좋다).
   선 길이 차로 정한 단계를 2.4 GHz 13채널과 5 GHz 24채널에 넣어, 가장 나쁜 채널의 분산이 1/8 의 몇 배인지 본다.
   RG316 의 속도 계수 0.695.
2) 위상 단계 재기
   한 채널의 세기 넷만으로는 단계가 하나로 정해지지 않는다. 네 점 (cos p, sin p) 을 평면에서 일차 변환(+옮김)해도
   네 점이 다시 단위원 위에 오는 변환이 한 줄로 이어지고, 그 변환은 세기를 똑같이 설명한다 (0·90·180·270 처럼
   마주 보는 두 쌍일 때만 점이 움직이지 않는다). 그래서 '선 길이 셋 + 길마다 고정 위상 셋'을 여러 채널에서 한꺼번에
   맞춘다. 두 대역을 섞어야 길이(주파수에 비례)와 고정 위상(주파수와 무관)이 갈린다.
"""
import itertools
import json
import os

import numpy as np
from scipy.optimize import fsolve, least_squares

VF, C0 = 0.695, 299792458.0
F24 = np.array([2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467, 2472]) * 1e6
F5 = np.array(list(range(5180, 5321, 20)) + list(range(5500, 5721, 20)) + list(range(5745, 5826, 20))) * 1e6
FALL = np.concatenate([F24, F5])
HERE = os.path.dirname(os.path.abspath(__file__))


def steps_of(f, dl_mm, off_deg=0.0):
    """선 길이 차 (mm) 와 길마다 고정 위상 (도) -> 주파수 f 에서 단계 (도)"""
    return (360.0 * f * np.asarray(dl_mm, float) / 1000.0 / (VF * C0) + np.asarray(off_deg, float)) % 360.0


def noise_gain(steps_deg):
    """Re(Z), Im(Z) 분산의 1/8 대비 배수 (큰 쪽)"""
    p = np.radians(steps_deg); A = np.stack([np.ones_like(p), 2 * np.cos(p), -2 * np.sin(p)], 1)
    try:
        cov = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(max(cov[1, 1], cov[2, 2]) / 0.125)


def worst_batch(D, freqs):
    """길이 차 [N, 4] (mm) -> 채널들 가운데 가장 나쁜 배수 [N]"""
    D = np.asarray(D, float); out = np.zeros(len(D))
    for f in freqs:
        p = 2 * np.pi * f * D / 1000.0 / (VF * C0); c, s = np.cos(p), np.sin(p)
        G = np.empty((len(D), 3, 3))
        G[:, 0, 0] = 4; G[:, 0, 1] = G[:, 1, 0] = 2 * c.sum(1); G[:, 0, 2] = G[:, 2, 0] = -2 * s.sum(1)
        G[:, 1, 1] = 4 * (c * c).sum(1); G[:, 2, 2] = 4 * (s * s).sum(1); G[:, 1, 2] = G[:, 2, 1] = -4 * (c * s).sum(1)
        det = np.linalg.det(G); ok = np.abs(det) > 1e-9
        g = np.full(len(D), np.inf)
        if ok.any():
            cov = np.linalg.inv(G[ok]); g[ok] = np.maximum(cov[:, 1, 1], cov[:, 2, 2]) / 0.125
        out = np.maximum(out, g)
    return out


def worst(dl, freqs):
    return float(worst_batch(np.asarray(dl, float)[None, :], freqs)[0])


def search(step_mm=1, max_mm=100):
    """길이 차 셋 (0 < a < b < c <= max_mm) 을 step_mm 간격으로 모두 본다"""
    C = np.array(list(itertools.combinations(range(step_mm, max_mm + 1, step_mm), 3)), float)
    D = np.hstack([np.zeros((len(C), 1)), C]); best_w, best_d = np.inf, None
    for i in range(0, len(D), 20000):
        w = worst_batch(D[i:i + 20000], FALL); k = int(np.argmin(w))
        if w[k] < best_w:
            best_w, best_d = float(w[k]), tuple(int(v) for v in D[i + k])
    return best_w, best_d


def tolerance(dl, sd_mm=2.0, n=20000, seed=1):
    r = np.random.default_rng(seed)
    D = np.asarray(dl, float)[None, :] + np.hstack([np.zeros((n, 1)), r.uniform(-sd_mm, sd_mm, (n, 3))])
    out = worst_batch(D, FALL)
    return dict(median=round(float(np.median(out)), 2), p95=round(float(np.percentile(out, 95)), 2), max=round(float(out.max()), 2))


def same_power_steps(p_deg, m11):
    """세기를 똑같이 설명하는 다른 단계: 네 점에 대칭 일차 변환 M (M11 = m11) 과 옮김 t 를 걸어 단위원 위로"""
    p = np.radians(p_deg); X = np.stack([np.cos(p), np.sin(p)], 1)

    def eq(v):
        m12, m22, t1, t2 = v; M = np.array([[m11, m12], [m12, m22]]); t = np.array([t1, t2])
        return [np.sum((M @ x + t) ** 2) - 1 for x in X]
    v, info, ok, _ = fsolve(eq, [0, 1, 0, 0], full_output=True)
    m12, m22, t1, t2 = v; Q = X @ np.array([[m11, m12], [m12, m22]]).T + np.array([t1, t2])
    if ok != 1 or np.abs(np.linalg.norm(Q, axis=1) - 1).max() > 1e-6:
        return None
    q = np.degrees(np.arctan2(Q[:, 1], Q[:, 0])); return ((q - q[0]) % 360).round(1).tolist()


def fake_power(steps, rng, M=16, F=52, snr=30.0):
    """기준 r 과 안테나 h 를 무작위로 두고 단계마다 세기 [4, M*F]"""
    r = rng.normal(size=F) + 1j * rng.normal(size=F); h = (rng.normal(size=(M, F)) + 1j * rng.normal(size=(M, F))) * 0.7
    Y = []
    for pk in np.radians(steps):
        v = r[None] + np.exp(1j * pk) * h
        n = (rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)) * np.sqrt(np.mean(np.abs(r) ** 2) * 10 ** (-snr / 10) / 2)
        Y.append((np.abs(v + n) ** 2).ravel())
    return np.array(Y)


def resid(Y, steps):
    p = np.radians(steps); A = np.stack([np.ones(4), 2 * np.cos(p), -2 * np.sin(p)], 1)
    return ((Y - A @ np.linalg.pinv(A) @ Y).ravel()) / np.sqrt(np.mean(Y ** 2))


def joint_fit(chans, true_dl, true_off, nominal, rng, starts=8):
    """채널 여럿의 세기로 선 길이 셋 + 고정 위상 셋을 맞춘다 -> 모든 채널에서 단계 오차 최대 (도)"""
    Ys = [fake_power(steps_of(f, true_dl, true_off), rng) for f in chans]

    def fun(x):
        dl = np.r_[0, x[:3]]; off = np.r_[0, x[3:]]
        return np.concatenate([resid(Y, steps_of(f, dl, off)) for Y, f in zip(Ys, chans)])
    best = None
    for _ in range(starts):
        x0 = np.r_[np.asarray(nominal[1:], float) + rng.uniform(-2, 2, 3), rng.uniform(-20, 20, 3)]
        s = least_squares(fun, x0, x_scale=[1, 1, 1, 10, 10, 10])
        if best is None or s.cost < best.cost:
            best = s
    dl = np.r_[0, best.x[:3]]; off = np.r_[0, best.x[3:]]
    check = [2412e6, 2472e6, 5180e6, 5500e6, 5825e6]
    return [round(float(np.abs((steps_of(f, dl, off) - steps_of(f, true_dl, true_off) + 180) % 360 - 180).max()), 1) for f in check]


def main():
    out = dict(vf=VF, channels_24=(F24 / 1e6).astype(int).tolist(), channels_5=(F5 / 1e6).astype(int).tolist())
    now = [0, 21, 42, 63]
    out["now"] = dict(dl_mm=now, worst_24=round(worst(now, F24), 2), worst_5=round(worst(now, F5), 2), tol_2mm=tolerance(now))
    w, dl = search(1)
    out["best_1mm"] = dict(dl_mm=list(dl), worst=round(w, 2), worst_24=round(worst(dl, F24), 2), worst_5=round(worst(dl, F5), 2), tol_2mm=tolerance(dl))
    out["best_1mm"]["steps_deg"] = {str(int(f / 1e6)): steps_of(f, dl).round().astype(int).tolist() for f in (2412e6, 2442e6, 2472e6, 5180e6, 5500e6, 5825e6)}
    w5, dl5 = search(5)
    w10, dl10 = search(10)
    out["best_5mm"] = dict(dl_mm=list(dl5), worst=round(w5, 2))
    out["best_10mm"] = dict(dl_mm=list(dl10), worst=round(w10, 2))
    # 세기만으로는 단계가 하나로 정해지지 않는다
    out["same_power"] = {str(p): {str(m): same_power_steps(p, m) for m in (0.9, 0.95, 1.05, 1.1)}
                         for p in ([0, 46, 127, 245], [0, 98, 269, 159], [0, 88, 175, 263], [0, 90, 180, 270])}
    # 여러 채널을 묶어 맞추기 (실제 길이가 이름값에서 1~2 mm 틀리고, 스위치 길마다 고정 위상이 붙었다고 둔다)
    rng = np.random.default_rng(7); true_dl = [0, 11.6, 29.1, 59.3]; true_off = [0, 12.0, -9.0, 17.0]; nominal = list(dl)
    fits = {}
    for chans in ([2412e6, 2437e6, 2462e6], [5180e6, 5500e6, 5785e6], [2437e6, 5180e6, 5785e6], [2412e6, 2462e6, 5180e6, 5500e6, 5785e6]):
        fits["+".join(str(int(f / 1e6)) for f in chans)] = joint_fit(chans, true_dl, true_off, nominal, rng)
    nominal_only = [round(float(np.abs((steps_of(f, nominal) - steps_of(f, true_dl, true_off) + 180) % 360 - 180).max()), 1)
                    for f in [2412e6, 2472e6, 5180e6, 5500e6, 5825e6]]
    out["joint_fit"] = dict(true_dl=true_dl, true_off=true_off, check_mhz=[2412, 2472, 5180, 5500, 5825], fits=fits, nominal_only=nominal_only)
    with open(os.path.join(HERE, "dual_band.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("now", "best_1mm", "best_5mm", "best_10mm")}, ensure_ascii=False))
    print("joint_fit", json.dumps(out["joint_fit"], ensure_ascii=False))


if __name__ == "__main__":
    main()
