"""전파 표에서 '골이 얼마나 뚜렷한가'를 재는 숫자 (방향 구하는 방법과 상관없는 물리적인 값).

방향(면 안의 각 psi, 면에서 벗어난 각 beta)과 편파마다 판을 한 바퀴 돌린 곡선을 꺼내서
  front  카메라 쪽 +-25도 구간의 평균 세기 (곡선 가운데값 기준 dB. 깊을수록 좋다)
  spur   그 밖(70도 넘게 떨어진 곳)에서 가장 깊은 50도 구간의 평균 (가짜 골)
  margin spur - front (+ 면 진짜 골이 더 깊다)
  err    '가장 깊은 50도 구간'을 골이라고 했을 때의 각도 오차
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json
import numpy as np
import device as V
import study as S

PSI = np.arange(0.0, 360.0, 15.0); BETA = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0); CHI = (0.0, 45.0, 90.0, 135.0)
SP = np.arange(0.0, 360.0, 5.0)


def boxmean(c, half=5):
    k = 2 * half + 1; return np.convolve(np.r_[c[-half:], c, c[:half]], np.ones(k) / k, mode="valid")


def one(R, psi, beta, chi):
    b, p = np.radians(beta), np.radians(psi); dh = np.array([np.cos(b) * np.cos(p), np.cos(b) * np.sin(p), np.sin(b)])
    tu = np.array([np.sin(b) * np.cos(p), np.sin(b) * np.sin(p), -np.cos(b)]); pu = np.array([-np.sin(p), np.cos(p), 0.0])
    eh = np.cos(np.radians(chi)) * tu + np.sin(np.radians(chi)) * pu
    c = 20 * np.log10(np.abs(R.look(dh, eh, SP)) + 1e-12); c = c - np.median(c)
    m = boxmean(c); x = (SP - psi + 180.0) % 360.0 - 180.0
    front = float(m[int(np.argmin(np.abs(x)))]); far = np.abs(x) >= 70.0; spur = float(m[far].min())
    err = float(abs(x[int(np.argmin(m))]))
    return front, spur, err


def table_metrics(name, betas=BETA):
    R = S.Resp(name); rows = [one(R, psi, beta, chi) for beta in betas for psi in PSI for chi in CHI]
    A = np.array(rows); mar = A[:, 1] - A[:, 0]
    return dict(front=float(np.median(A[:, 0])), spur=float(np.median(A[:, 1])), margin=float(np.median(mar)), win=float(np.mean(mar > 0) * 100),
                err=float(np.median(A[:, 2])), bad20=float(np.mean(A[:, 2] > 20) * 100), n=len(A))


def line(name, m):
    return "%-16s 진짜 골 %5.1f dB, 가짜 골 %5.1f dB, 진짜가 더 깊은 경우 %3.0f%% | 가장 깊은 구간으로 찾으면 오차 가운데값 %4.1f도, 20도 넘게 틀림 %3.0f%%" % (name, m["front"], m["spur"], m["win"], m["err"], m["bad20"])


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(V.CACHE) if f.endswith(".npz") and not f.endswith("_0.npz"))
    path = os.path.join(empaths.STUDIES, "metrics.json")
    out = json.load(open(path, encoding="utf-8")) if (sys.argv[1:] and os.path.exists(path)) else {}          # 이름을 주면 그 표만 다시 재서 결과 파일에 더한다
    for n in names:
        out[n] = table_metrics(n); print(line(n, out[n]), flush=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
