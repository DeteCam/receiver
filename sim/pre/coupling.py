# -*- coding: utf-8 -*-
"""안테나끼리의 간섭(상호 결합)이 얼마나 커도 방향을 맞히나. 빠른 시험의 통과 기준을 정한다.

    python3 coupling.py            -> coupling.json

모델
  안테나 16개 (4 x 4, 33 mm, 두 대역 판). 받은 값 h = C (a(방향) + 잡음).
  C 는 결합 행렬: 대각 1, 이웃 i, j 사이 c_ij = k * (33mm / d_ij) * exp(-j 2pi (d_ij - 33mm) / lambda).
  k (가장 가까운 이웃 사이 결합, dB) 를 -30 ~ -6 dB 로 바꿔 가며, 결합을 모르고(이상적인 안테나로 보고) 방향을 찾는다.
  비교로 '회전 받침으로 잰 실측표'를 쓰는 경우(결합을 알고 있다 = 보정)도 본다.

빠른 시험에서 잴 수 있는 것은 안테나 하나의 세기뿐이다. 그래서 결합 k 가 만드는
  (1) 이웃이 있을 때와 없을 때 정면에서 받는 세기의 차 (시험 2)
  (2) 돌렸을 때 세기 무늬의 출렁임 (시험 3, 이웃이 없을 때 무늬 대비, 봉우리-골 dB)
를 같이 셈해서, 시험에서 본 숫자로 k 를 거꾸로 짐작할 수 있게 한다.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware", "core"))
import panel as P  # noqa: E402

PITCH = 0.033
FREQS = {"2.4 GHz (6번)": (2.437e9, 60.0), "5.2 GHz (40번)": (5.200e9, 45.0), "5.8 GHz (157번)": (5.785e9, 30.0)}   # (주파수, 헷갈림 없이 볼 수 있는 정면에서의 각)
KAPPA_DB = [-30, -20, -15, -12, -10, -8, -6]
SNR_DB = 20.0
NF = 8                       # 서브캐리어 수 (20 MHz 안에 고르게)


def coupling_matrix(R, lam, kappa_db):
    k = 10 ** (kappa_db / 20.0)
    d = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
    C = np.eye(len(R), dtype=complex)
    m = d > 0
    C[m] = k * (PITCH / d[m]) * np.exp(-2j * np.pi * (d[m] - PITCH) / lam)
    return C


def ideal(R, dirs, lam):
    """패널 좌표의 방향 dirs [N,3] (y = 앞) 에서 안테나마다의 값 [N,M] (앞쪽 cos 무늬)"""
    return np.exp(2j * np.pi * dirs.dot(R.T) / lam) * np.clip(dirs[:, 1], 0.05, None)[:, None]


def dir_grid(step=2.0):
    az = np.radians(np.arange(-89, 90, step)); el = np.radians(np.arange(-60, 61, step)); A, L = np.meshgrid(az, el)
    G = np.stack([np.cos(L) * np.sin(A), np.cos(L) * np.cos(A), np.sin(L)], -1).reshape(-1, 3)
    return G[G[:, 1] > 0.1]


def estimate(Z, cand, man):
    """Z [M,F], 후보 방향 cand [N,3] 의 값 man [N,M] -> 가장 잘 맞는 방향"""
    Rm = Z.dot(Z.conj().T)
    p = np.einsum("nm,mk,nk->n", man.conj(), Rm, man).real / np.maximum((np.abs(man) ** 2).sum(1), 1e-30)
    return cand[int(np.argmax(p))]


def run(fc, kappa_db, n=300, seed=0, fov_deg=60.0):
    """정면에서 fov_deg 안의 원뿔 속 방향만 (5 GHz 의 헷갈림은 coupling 과 따로 본다)"""
    rng = np.random.default_rng(seed)
    R = P.grid(4, 4, PITCH)                                  # x, y(=0), z
    lam = P.C0 / fc
    C = coupling_matrix(R, lam, kappa_db)
    cand = dir_grid(2.0)
    man_ideal = ideal(R, cand, lam)
    man_cal = man_ideal.dot(C.T)                              # 실측표 (결합을 안다)
    err_u, err_c = [], []
    for _ in range(n):
        while True:
            az = np.radians(rng.uniform(-fov_deg, fov_deg)); el = np.radians(rng.uniform(-min(fov_deg, 40.0), min(fov_deg, 40.0)))
            d = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            if d[1] >= np.cos(np.radians(fov_deg)):
                break
        f = fc + np.linspace(-8.75e6, 8.75e6, NF)
        a = np.stack([ideal(R, d[None], P.C0 / ff)[0] for ff in f], 1)          # [M,F]
        h = C.dot(a)
        s = np.sqrt(np.mean(np.abs(h) ** 2) / 10 ** (SNR_DB / 10) / 2)
        Z = h + s * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
        for man, out in ((man_ideal, err_u), (man_cal, err_c)):
            g = estimate(Z, cand, man)
            out.append(np.degrees(np.arccos(np.clip(g.dot(d), -1, 1))))
    # 시험에서 보이는 숫자: 안테나 6번(가운데 쪽)의 세기, 이웃 있음 / 없음
    az = np.radians(np.arange(-80, 81, 5)); D = np.stack([np.sin(az), np.cos(az), np.zeros_like(az)], 1)
    iso = ideal(R, D, lam); emb = iso.dot(C.T)
    k = 5
    ratio = 20 * np.log10(np.abs(emb[:, k]) / np.abs(iso[:, k]))
    front = float(abs(ratio[len(az) // 2]))
    ripple = float(ratio.max() - ratio.min())
    return dict(err_uncal_med=float(np.median(err_u)), err_uncal_p90=float(np.percentile(err_u, 90)),
                err_cal_med=float(np.median(err_c)), err_cal_p90=float(np.percentile(err_c, 90)),
                front_change_db=round(front, 2), ripple_db=round(ripple, 2))


def main():
    out = {"model": "4x4, 33 mm, SNR %g dB, 서브캐리어 %d, 대역마다 헷갈림 없는 정면 원뿔 안, 300번" % (SNR_DB, NF), "rows": []}
    for name, (fc, fov) in FREQS.items():
        base = run(fc, -60, fov_deg=fov)
        print("%-14s 결합 없음: %4.1f / %4.1f도 (정면 %d도 안)" % (name, base["err_uncal_med"], base["err_uncal_p90"], fov))
        for kd in KAPPA_DB:
            r = run(fc, kd, fov_deg=fov); r.update(band=name, fov_deg=fov, kappa_db=kd, err_none_med=base["err_uncal_med"], err_none_p90=base["err_uncal_p90"])
            out["rows"].append(r)
            print("%-14s k=%4d dB  보정 없이 %5.1f / %5.1f도  보정하면 %4.1f / %4.1f도  정면 세기 차 %4.1f dB  무늬 출렁임 %4.1f dB"
                  % (name, kd, r["err_uncal_med"], r["err_uncal_p90"], r["err_cal_med"], r["err_cal_p90"], r["front_change_db"], r["ripple_db"]))
    with open(os.path.join(HERE, "coupling.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
