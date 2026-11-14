"""방 전체 계산(room2d.py)의 결과로 '골 찾기'를 바꿔 보는 실험. 새 전파 계산 없이 room2d_data/ (없으면 room2d_cache/) 의 값만 다시 읽는다.
    python3 room2d_exp.py          ->  room2d_exp.json

같은 곡선에 대해
  곡선 만들기   지금 장치(골이 뚜렷한 주파수만 골라 가운데값) / 모든 주파수의 가운데값 / 평균 세기
  골 찾기       지금 장치(10° 로 고르게 한 뒤 fusion.trough) / 20° / 30° 로 고르게 / 60° 창으로 모으기 / 기준보다 12dB 넘게 꺼진 값 깎기
를 바꿔 가며 방향 오차를 견준다. 조건을 하나씩 바꾸는 물리 실험(거울을 뺀 화장실, 파이 기판을 뺀 화장실)은 room2d.py 의 ROOMS 에 있다."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, json
import numpy as np
import room2d as R2
from fusion import trough, circ_smooth, fill_nan

HERE = os.path.dirname(os.path.abspath(__file__))


def c_now(A):
    return R2.curve(A)


def c_all(A):
    db = 20 * np.log10(A + 1e-30); db = db - np.median(db, axis=0); return np.median(db, axis=1)


def c_pow(A):
    p = (A ** 2).mean(axis=1); db = 10 * np.log10(p + 1e-30); return db - np.median(db)


def t_smooth(width):
    def f(phi, lv):
        return trough(phi, circ_smooth(fill_nan(lv), width / (phi[1] - phi[0])))
    return f


def t_window(width=60.0):
    def f(phi, lv):
        step = phi[1] - phi[0]; x = fill_nan(lv); att = np.clip(np.median(x) - x, 0, None)
        k = int(round(width / step)); sc = np.convolve(np.r_[att[-k:], att, att[:k]], np.ones(k) / k, mode="same")[k:-k]
        i = int(np.argmax(sc)); return float((phi[i] + 180.0) % 360.0 - 180.0), float(sc[i]), 0.0
    return f


def t_clip(depth=12.0):
    def f(phi, lv):
        x = fill_nan(lv); return trough(phi, np.maximum(x, np.median(x) - depth))
    return f


CURVES = {"지금 장치": c_now, "모든 주파수 가운데값": c_all, "평균 세기": c_pow}
TROUGHS = {"지금 장치 (10° 고르게)": trough, "20° 로 고르게": t_smooth(20.0), "30° 로 고르게": t_smooth(30.0), "60° 창으로 모으기": t_window(60.0), "12dB 넘게 꺼진 값 깎기": t_clip(12.0)}


def cached(name, ri):
    """그 방·자리의 판 각 72개가 다 계산되어 있으면 (판 각, 카메라, 주파수), 아니면 None"""
    data = os.path.join(R2.DATA, "%s_%d.npz" % (R2.TAG[name], ri))
    if os.path.exists(data):
        return np.load(data)["S"].astype(float)
    fs = [os.path.join(R2.CACHE, "%s_%d_%03d.npz" % (R2.TAG[name], ri, int(p))) for p in R2.PHI]
    return np.stack([np.load(f)["S"] for f in fs]) if all(os.path.exists(f) for f in fs) else None


def errors(name, ri, S, cf, tf):
    R = R2.ROOMS[name]; cam = R2.cams(R); ax, ay, yaw = R["rx"][ri]; e = []
    for j, (cx, cy) in enumerate(cam):
        psi = tf(R2.PHI, cf(S[:, j, :]))[0]
        az = (psi + 90.0 + yaw) % 360.0; at = np.degrees(np.arctan2(cy - ay, cx - ax)) % 360.0
        e.append(abs((az - at + 180.0) % 360.0 - 180.0))
    return np.array(e)


if __name__ == "__main__":
    out = {}
    for name, R in R2.ROOMS.items():
        for ri in range(len(R["rx"])):
            S = cached(name, ri)
            if S is None:
                continue
            key = "%s · 수신기 %d" % (name, ri + 1); res = {}
            for cn, cf in CURVES.items():
                for tn, tf in TROUGHS.items():
                    if cn != "지금 장치" and tn != "지금 장치 (10° 고르게)":
                        continue                                  # 하나씩만 바꾼다
                    e = errors(name, ri, S, cf, tf)
                    res["%s | %s" % (cn, tn)] = dict(med=round(float(np.median(e)), 1), over30=round(float((e > 30).mean() * 100), 1), n=int(len(e)))
            out[key] = res
            print(key, {k: (v["med"], v["over30"]) for k, v in res.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "room2d_exp.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
