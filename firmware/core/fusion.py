"""곡선의 자세한 모양에 기대지 않는 방향 추정.

스캔마다 '세기가 꺼진 구간의 가운데'만 뽑고(논문 방식과 같다), 여러 스캔의 결과를 기하로 합친다.
  - 골이 보인 스캔: 카메라는 그 스캔의 회전축과 골 가운데 방향이 만드는 반쪽 평면 근처에 있고, 판이 가릴 수 있는 범위 안에 있다.
  - 골이 안 보인 스캔: 카메라는 그 스캔에서 판이 가릴 수 없는 쪽(회전축 방향에 가까운 쪽)에 있다.
판이 가릴 수 있는 범위는 판 치수에서 나온다: 판이 안테나 기준 -58 ~ +92mm, 반지름 80mm 이므로 면에서 -36도 ~ +49도.
실제 측정값에도 그대로 쓴다: curves = [(판 각(도) 배열, 세기(dB) 배열), ...], frames = [(yaw, tilt), ...]
"""
import numpy as np
from estimator import unit, azel, frame, to_scan, sphere

B_LO, B_HI = -36.0, 49.0          # 판이 직선으로 가리는 범위 (면에서 벗어난 각)
ANT_BACK, ANT_UP, AXIS_UP = 0.020, 0.073, 0.120     # 안테나 자리 (m): 끄덕임 축에서 뒤로 2cm, 위로 7.3cm. 끄덕임 축은 바닥에서 12cm


def antenna_pos(yaw, tilt):
    """안테나가 있는 곳 (m). 원점 = 받침 바닥 한가운데, 받침 0도 쪽이 +X. 머리를 숙이고 받침을 돌리면 안테나가 받침 축 둘레를 돈다."""
    t, y = np.radians(tilt), np.radians(yaw)
    x = -ANT_BACK * np.cos(t) + ANT_UP * np.sin(t)
    return np.array([x * np.cos(y), x * np.sin(y), AXIS_UP + ANT_BACK * np.sin(t) + ANT_UP * np.cos(t)])


def origin(frames):
    """방향을 재는 기준점 (m): 스캔들의 안테나 자리 한가운데. 세 번 재기(tilt3)면 받침 축 위, 바닥에서 17.9cm"""
    return np.mean([antenna_pos(y, t) for y, t in frames], axis=0)


def seen_from(C, frames, k, dist):
    """기준점에서 본 방향 C (N,3) 가 k 번째 스캔의 안테나 자리에서는 어느 방향인가. dist = 카메라까지 거리(m). None 이면 아주 멀다고 본다."""
    if dist is None:
        return C
    o = antenna_pos(*frames[k]) - origin(frames)
    v = np.asarray(C) * float(dist) - o
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def circ_smooth(x, n):
    n = max(1, int(n) | 1)
    k = np.ones(n) / n
    return np.convolve(np.r_[x[-n:], x, x[:n]], k, mode="same")[n:-n]


def fill_nan(x):
    x = np.array(x, dtype=float); ok = np.isfinite(x)
    if ok.all():
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], np.r_[idx[ok] - len(x), idx[ok], idx[ok] + len(x)], np.r_[x[ok], x[ok], x[ok]])
    return x


def trough(phi, lv, gap=36.0, max_width=150.0):
    """세기가 꺼진 구간의 가운데 각, 깊이(dB), 흔들림(dB).
    기준(가운데값)보다 일정 값 넘게 낮은 칸들을 찾고, 가장 깊은 곳이 든 구간에서 시작해 가까운(36도 안쪽) 이웃 구간을 이어 붙인 뒤
    그 구간의 한가운데를 돌려준다. 가운데가 볼록한 W 모양이든 하나짜리 V 모양이든, 한쪽이 더 깊어도 같은 값이 나온다."""
    phi = np.asarray(phi, dtype=float); step = phi[1] - phi[0]; n = len(phi)
    x = fill_nan(lv)
    s1 = circ_smooth(x, 10.0 / step)
    base = np.median(s1)
    att = base - s1
    i0 = int(np.argmax(att)); depth = float(att[i0])
    thr = float(np.clip(0.3 * depth, 2.0, 4.0))
    low = att >= thr
    if not low[i0]:
        return float((phi[i0] + 180.0) % 360.0 - 180.0), depth, 0.0
    # 원형으로 이어진 구간들 (시작 칸, 길이)
    runs = []; k = 0
    start = next((j for j in range(n) if not low[j]), None)
    if start is None:
        return float((phi[i0] + 180.0) % 360.0 - 180.0), depth, 0.0
    j = start
    while k < n:
        idx = (j + k) % n
        if low[idx]:
            a0 = k
            while k < n and low[(j + k) % n]:
                k += 1
            runs.append((a0, k - a0))
        else:
            k += 1
    pos0 = (i0 - j) % n
    m = next(r for r in range(len(runs)) if runs[r][0] <= pos0 < runs[r][0] + runs[r][1])
    lo, hi = runs[m][0], runs[m][0] + runs[m][1]
    g = int(round(gap / step)); wmax = int(round(max_width / step))
    grown = True
    while grown:
        grown = False
        for r, (a0, ln) in enumerate(runs):
            dmax = att[[(j + q) % n for q in range(a0, a0 + ln)]].max()
            if dmax < 0.35 * depth:
                continue
            if a0 >= hi and a0 - hi <= g and (a0 + ln - lo) <= wmax:
                hi = a0 + ln; grown = True
            elif a0 + ln <= lo and lo - (a0 + ln) <= g and (hi - a0) <= wmax:
                lo = a0; grown = True
    mid = (lo + hi - 1) / 2.0
    centre = phi[0] + ((j + mid) % n) * step
    inside = np.zeros(n, dtype=bool); inside[[(j + q) % n for q in range(lo, hi)]] = True
    rest = (s1 - base)[~inside]
    noise = float(1.4826 * np.median(np.abs(rest - np.median(rest)))) if rest.size else 0.0
    return float((centre + 180.0) % 360.0 - 180.0), depth, noise


def p_seen(beta):
    """그 스캔에서 골이 보일 가능성 (면에서 벗어난 각으로)"""
    return 1.0 / (1.0 + np.exp(-(beta - B_LO - 4.0) / 5.0)) / (1.0 + np.exp((beta - B_HI + 4.0) / 5.0))


class Fusion:
    def __init__(self, n_grid=8000, min_depth=4.5, k_noise=3.0):
        self.G = sphere(n_grid); self.min_depth = min_depth; self.k_noise = k_noise

    def read(self, curves):
        out = []
        for phi, lv in curves:
            psi, depth, noise = trough(phi, lv)
            out.append(dict(psi=psi, depth=depth, noise=noise, seen=bool(depth >= max(self.min_depth, self.k_noise * noise))))
        return out

    def score(self, C, obs, frames, dist=None):
        s = np.zeros(len(C))
        for k, (o, (yaw, tilt)) in enumerate(zip(obs, frames)):
            psi, beta = to_scan(seen_from(C, frames, k, dist), frame(yaw, tilt))
            ps = p_seen(beta)
            if o["seen"]:
                sig = 6.0 * np.sqrt(8.0 / max(o["depth"], 4.0))
                d = (psi - o["psi"] + 180.0) % 360.0 - 180.0
                s += -0.5 * (d / sig) ** 2 + np.log(ps + 1e-6)
            else:
                s += np.log(1.0 - 0.95 * ps)
        return s

    def solve(self, curves, frames, dist=None):
        """dist(m)를 주면 카메라가 그 거리에 있다고 보고, 스캔마다 안테나 자리가 다른 것(가까운 카메라에서 방향이 몇 도 달라 보인다)을 계산에 넣는다."""
        obs = self.read(curves)
        n_seen = sum(o["seen"] for o in obs)
        if n_seen == 0:
            return None, obs
        sc = self.score(self.G, obs, frames, dist)
        c0 = self.G[int(np.argmax(sc))]
        for sp, st in ((6.0, 1.0), (1.5, 0.25)):
            az0, el0 = azel(c0)
            d = np.arange(-sp, sp + 1e-9, st)
            A, E = np.meshgrid(az0 + d / max(np.cos(np.radians(el0)), 0.15), np.clip(el0 + d, -89.5, 89.5))
            C = unit(A.ravel(), E.ravel())
            c0 = C[int(np.argmax(self.score(C, obs, frames, dist)))]
        return c0, obs

    def spread(self, obs, frames, c, drop=3.0):
        """가장 좋은 방향 둘레에서 점수가 drop 안쪽인 후보들이 퍼진 각 (도). 크면 한 스캔으로만 정해져 불확실하다는 뜻."""
        sc = self.score(self.G, obs, frames)
        m = sc >= sc.max() - drop
        return float(np.degrees(np.arccos(np.clip(self.G[m] @ c, -1, 1))).max())


def consistency(est, obs, frames, tol=15.0, margin=8.0, dist=None):
    """찾은 방향이 스캔 하나하나와 맞는지 본다. (모두 맞는지, 스캔별 내용) 을 돌려준다.
    - 골이 보인 스캔: 잰 골 가운데가 그 방향에서 예상되는 각과 tol 안으로 맞고, 방향이 판이 가릴 수 있는 범위 안에 있어야 한다.
    - 골이 안 보인 스캔: 그 방향이 판이 뚜렷이 가리는 범위(가장자리에서 margin 안쪽) 한가운데면 안 맞는 것이다."""
    out = []; ok = True
    for k, (o, (yaw, tilt)) in enumerate(zip(obs, frames)):
        psi, beta = to_scan(seen_from(est, frames, k, dist), frame(yaw, tilt))
        if o["seen"]:
            d = abs((o["psi"] - psi + 180.0) % 360.0 - 180.0)
            good = d <= tol and (B_LO - margin <= beta <= B_HI + margin)
            out.append(dict(seen=True, resid=float(d), beta=float(beta), ok=bool(good)))
        else:
            clear = B_LO + margin < beta < B_HI - margin
            out.append(dict(seen=False, resid=None, beta=float(beta), ok=bool(not clear)))
        ok = ok and out[-1]["ok"]
    return bool(ok), out


def next_scan(obs, frames, tilt):
    """골이 한 스캔에서만 보였을 때, 그 반쪽 평면을 가로질러 재는 다음 스캔의 받침 각을 고른다."""
    k = max(range(len(obs)), key=lambda i: obs[i]["depth"] if obs[i]["seen"] else -1)
    a, ex, ey = frame(*frames[k])
    p = np.radians(obs[k]["psi"]); u = np.cos(p) * ex + np.sin(p) * ey
    b = np.radians(8.0); c = np.cos(b) * u + np.sin(b) * a          # 반쪽 평면 위에서 가장 그럴듯한 점 (면에서 +8도)
    az, el = azel(c)
    x = -np.tan(np.radians(el)) / np.tan(np.radians(tilt))
    d = np.degrees(np.arccos(np.clip(x, -1, 1)))
    cands = [(float(az) + d + 180) % 360 - 180, (float(az) - d + 180) % 360 - 180]
    used = [f[0] for f in frames]
    far = lambda y: min(abs((y - u_ + 180) % 360 - 180) for u_ in used)
    return max(cands, key=far), tilt
