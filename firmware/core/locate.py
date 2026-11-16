# -*- coding: utf-8 -*-
"""여러 자리에서 잰 결과를 합쳐 카메라가 있을 만한 '추정 구간'을 구한다.
수신기(파이)와 시뮬레이션이 같은 파일을 쓴다. numpy 만 있으면 된다 (파이썬 3.7, numpy 1.16 에서도 돈다).

생각
  한 자리에서 나오는 것은 '방향 하나'가 아니라 '어느 방향이 얼마나 그럴듯한가' 이다 (스캔마다 골이 보인 각, 깊이, 안 보인 스캔).
  방 안을 10cm 칸으로 나누고, 칸마다 "여기에 카메라가 있다면 이 자리의 측정이 얼마나 그럴듯한가"를 구해 자리끼리 곱한다.
  가장 그럴듯한 칸에 이어 붙은 덩어리가 추정 구간이다. 카메라가 벽에 붙어 있다고 가정하지 않는다 (방 한가운데 화분 속이어도 된다).
  한 자리의 측정이 통째로 틀렸을 때(반사가 만든 가짜 골)를 대비해 칸마다 바닥값을 깔아 둔다: 그러면 구간이 커지거나 믿음이 낮아져서 티가 난다.

쓰는 법
  S = Session(lo=(0, 0, 0), hi=(3.5, 4.5, 2.4))            # 방(또는 찾을 범위)의 두 구석 (m). 폰 AR 의 좌표 그대로
  S.add(pose=((x, y, z), heading_deg), obs=obs, frames=frames)   # 자리마다 한 번. pose = 받침 바닥 한가운데의 자리와 받침 0도(앞 표시)가 향한 각
  r = S.result()                                           # 추정 구간, 믿음, 다음에 놓을 자리
좌표: z 가 위. heading 은 위에서 볼 때 방의 +x 에서 반시계로 잰 각(도).

시험장 모드 (ExamSession, 2026-11-17)
  몰카 찾기(Session)는 '방 안 어디든' 칸으로 나눠 구간을 구하고, 수신기는 바닥·탁자 어디에나 놓는다.
  시험장은 카메라가 '어느 자리'에 있는지가 답이고, 시험 중에는 자리 사이로 들어갈 수 없다. 그래서 다른 것은 셋뿐이다:
    - 후보: 방의 칸 대신 자리마다 카메라가 있을 법한 점 (책상 위 여러 점 + 가슴). 답은 자리 번호 후보 3개.
    - 수신기를 놓는 곳: 사용자가 정한 곳만 (모서리·벽·교탁 자리, 받침대 높이 2.4m).
    - 다음 자리 고르기: 지금 확률이 높은 자리들이 그곳에서 가장 벌어져 보이는 곳 (sim/location/perimeter2.py 에서 가장 좋았던 것).
  점수 계산(score_points, likelihood)은 몰카 찾기와 같은 것을 쓴다.
  E = ExamSession(seats=[{"id": "1-1", "x": .., "y": ..}, ...], spots=[{"name": "앞 왼쪽", "x": .., "y": .., "z": 2.4}, ...], front=(0, -1))
  E.add(pose, obs, frames); r = E.result()                 # 자리 후보 3개, 다음에 놓을 곳
"""
import numpy as np
from estimator import frame, to_scan
from fusion import antenna_pos, origin, p_seen

STEP = 0.10          # 칸 크기 (m)
Q_BAD = 0.15         # 한 자리의 측정이 통째로 틀렸을 확률 (방 시뮬레이션에서 10~25%)
SIG_SCALE = 1.5      # 골 가운데 각의 흔들림을 모형보다 이만큼 넓게 본다 (좁게 잡으면 구간이 너무 작아져 참 위치를 놓친다: sim/ 의 locate3d_var2.py)
MANY_N, SIG_MANY = 6, 2 ** 0.5      # 한 자리에서 여섯 번 넘게 잰 측정은 흔들림을 루트 2배 더 넓게 본다. 한 자리의 스캔들은 같은 반사를 겪어 함께 틀리므로,
                                    # 따로따로 믿으면 구간이 너무 빨리 줄어 참 위치를 놓친다 (sim/ 의 places.py: 욕실에서 '좁혀졌는데 틀림' 27% -> 17%)
ZONE_CUT = 0.135     # 구간 = 가장 그럴듯한 칸에 이어 붙은 칸들 가운데, 그럴듯함이 최고의 13.5% (= 2시그마) 를 넘는 것
ZONE_PAD = 0.15      # 화면에 그릴 때 구간 둘레에 두는 여유 (m)
GOOD_LEN, GOOD_CONF = 1.0, 0.5      # '믿을 만한 결과': 구간의 긴 쪽이 1m 이하이고 구간 안의 확률 합이 50% 이상
GOOD_FRAC = 0.4     # 좁은 곳에서는 1m 가 너무 느슨하다: 방의 긴 변의 40% 와 1m 가운데 작은 쪽을 쓴다 (변기 칸 1.5m -> 0.6m)
MAX_POS = 5          # 이만큼 재도 안 좁혀지면 그만 잰다


# ---------------------------------------------------------------- 한 자리의 측정 -> 방 안 점들의 점수
def score_points(Xdev, obs, frames, sig_scale=1.0):
    """Xdev (N,3): 수신기 좌표의 점 (m. 원점 = 받침 바닥 한가운데, 받침 0도 쪽이 +X). 점마다 '여기에 카메라가 있다면'의 로그 점수.
    스캔마다 안테나 자리가 다른 것(받침 축에서 4.8cm)을 그대로 넣었다: 가까운 점일수록 스캔마다 방향이 달라 보인다."""
    s = np.zeros(len(Xdev))
    for o, (yaw, tilt) in zip(obs, frames):
        v = Xdev - antenna_pos(yaw, tilt)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        psi, beta = to_scan(v, frame(yaw, tilt)); ps = p_seen(beta)
        if o["seen"]:
            sig = sig_scale * 6.0 * np.sqrt(8.0 / max(o["depth"], 4.0))
            d = (psi - o["psi"] + 180.0) % 360.0 - 180.0
            s += -0.5 * (d / sig) ** 2 + np.log(ps + 1e-6)
        else:
            s += np.log(1.0 - 0.95 * ps)
    return s


def to_dev(Xroom, pose):
    """방 좌표의 점 -> 수신기 좌표. pose = (받침 바닥 한가운데의 자리, 받침 0도가 방에서 향하는 각)"""
    base, head = pose
    a = np.radians(-head); c, s = np.cos(a), np.sin(a); v = Xroom - np.asarray(base, dtype=float)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


def to_room(d, head):
    """수신기 좌표의 방향 -> 방 좌표의 방향"""
    a = np.radians(head); c, s = np.cos(a), np.sin(a)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


class Grid(object):
    """방(또는 찾을 범위)을 칸으로 나눈 것. L 만 주면 (0,0,0) ~ L, lo 를 주면 lo ~ lo + L"""
    def __init__(self, L, step=STEP, lo=(0.0, 0.0, 0.0)):
        self.L = np.asarray(L, dtype=float); self.step = step; self.lo = np.asarray(lo, dtype=float)
        ax = [np.arange(step / 2, l, step) for l in self.L]
        self.shape = tuple(len(a) for a in ax)
        self.X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) + self.lo


def likelihood(grid, meas, q=Q_BAD, sig_scale=1.0):
    """한 자리의 측정이 칸마다 주는 그럴듯함. '측정이 맞다(1-q)' 와 '통째로 틀렸다(q: 그러면 어느 칸이나 같다)' 를 섞는다.
    그래서 한 자리가 틀려도 참 위치의 칸이 0 이 되지 않고, 대신 구간이 커지거나 믿음이 낮아져서 티가 난다."""
    if len(meas["frames"]) >= MANY_N:
        sig_scale = sig_scale * SIG_MANY
    s = score_points(to_dev(grid.X, meas["pose"]), meas["obs"], meas["frames"], sig_scale)
    l = np.exp(s - s.max())
    return l + q / (1.0 - q) * l.mean()


def posterior(grid, measures, prior=None, q=Q_BAD, sig_scale=1.0):
    """칸마다의 확률 (합 = 1). prior: 칸마다의 사전 가중 (표면 가까이만 1 등)"""
    p = np.ones(len(grid.X)) if prior is None else np.asarray(prior, dtype=float).copy()
    for m in measures:
        p *= likelihood(grid, m, q, sig_scale)
    tot = p.sum()
    return p / tot if tot > 0 else p


def grow(mask3, seed):
    """seed 칸에서 시작해 mask3 안에서 맞닿은 칸들로 번져 나간 덩어리 (3차원, 면으로 맞닿은 칸)"""
    reg = np.zeros_like(mask3); reg[seed] = True
    while True:
        g = reg.copy()
        g[1:, :, :] |= reg[:-1, :, :]; g[:-1, :, :] |= reg[1:, :, :]
        g[:, 1:, :] |= reg[:, :-1, :]; g[:, :-1, :] |= reg[:, 1:, :]
        g[:, :, 1:] |= reg[:, :, :-1]; g[:, :, :-1] |= reg[:, :, 1:]
        g &= mask3
        if g.sum() == reg.sum():
            return reg
        reg = g


def zone(grid, post, cut=ZONE_CUT):
    """추정 구간: 가장 그럴듯한 칸에 이어 붙은 덩어리. 돌려주는 것:
    (가운데 점, 구간에 든 칸 번호들, 긴 쪽 길이 m, 부피 리터, 믿음 = 구간 안의 확률 합, 둘째 후보의 세기 = 구간 밖에서 가장 큰 값 / 최고값)"""
    P3 = post.reshape(grid.shape); i0 = int(np.argmax(post)); seed = np.unravel_index(i0, grid.shape)
    reg = grow(P3 >= cut * post[i0], seed); idx = np.nonzero(reg.ravel())[0]
    w = post[idx]; X = grid.X[idx]
    centre = (X * w[:, None]).sum(0) / w.sum()
    length = float(np.linalg.norm(X.max(0) - X.min(0)) + grid.step)
    out = post.copy(); out[idx] = 0.0
    return centre, idx, length, len(idx) * grid.step ** 3 * 1000.0, float(w.sum()), float(out.max() / post[i0])


def in_zone(grid, idx, p, tol=None):
    """점 p 가 구간 안(가장 가까운 구간 칸까지 칸 하나 거리 안)에 있는가"""
    tol = grid.step * 0.9 if tol is None else tol
    return bool(np.min(np.linalg.norm(grid.X[idx] - p, axis=1)) <= tol)


def zone_gap(grid, idx, p):
    """점 p 에서 구간까지의 거리 (m). 구간 안이면 0 에 가깝다"""
    return float(np.min(np.linalg.norm(grid.X[idx] - p, axis=1)))


def long_axis(grid, post, idx):
    """구간의 가운데와 긴 쪽 방향"""
    Xz = grid.X[idx]; wz = post[idx]; mu = (Xz * wz[:, None]).sum(0) / wz.sum()
    cov = ((Xz - mu) * wz[:, None]).T @ (Xz - mu) / wz.sum() + np.eye(3) * 1e-6
    return mu, np.linalg.eigh(cov)[1][:, -1]


# ---------------------------------------------------------------- 다음에 놓을 자리
def ray_exit(o, d, lo, hi):
    """점 o 에서 d 쪽으로 가는 선이 상자(lo ~ hi)를 빠져나가는 거리"""
    t = np.inf
    for ax in range(3):
        if abs(d[ax]) > 1e-9:
            tt = ((hi[ax] if d[ax] > 0 else lo[ax]) - o[ax]) / d[ax]
            if tt > 0:
                t = min(t, tt)
    return float(t)


def spots(lo, hi, z, avoid, min_sep=0.6, margin=0.3, step=0.25):
    """수신기를 놓을 수 있는 자리 후보 (지금과 같은 높이의 바닥이나 탁자 위라고 본다). avoid: 이미 잰 자리들"""
    xs = np.arange(lo[0] + margin, hi[0] - margin + 1e-9, step); ys = np.arange(lo[1] + margin, hi[1] - margin + 1e-9, step)
    P = np.array([(x, y, z) for x in xs for y in ys]).reshape(-1, 3)
    keep = np.ones(len(P), dtype=bool)
    for a in avoid:
        keep &= np.linalg.norm(P[:, :2] - np.asarray(a)[:2], axis=1) >= min_sep
    return P[keep]


def second_cost(o, d, P, lo, hi, frames, sig_deg=8.0, near=0.5):
    """첫 측정의 선(o 에서 d 쪽)을 보고, 후보 자리 P 마다 '선 위의 깊이를 얼마나 잘 가르는가' (작을수록 좋다. m 단위의 어림 오차).
    선 위의 점 X 를 자리 B 에서 보면, B 의 방향 오차 sig 가 선을 따라 sig x (B 에서 X 까지 거리) / sin(두 선이 이루는 각) 만큼의 깊이 오차가 된다."""
    T = min(ray_exit(o, d, lo, hi), 8.0)
    ts = np.linspace(0.45, max(T, 0.6), 14); X = o[None, :] + ts[:, None] * d[None, :]
    hB = origin(frames); cost = np.full(len(P), np.inf)
    for i, p in enumerate(P):
        u = X - (p + hB); rb = np.linalg.norm(u, axis=1)
        if rb.min() < near:                                          # 선에 너무 붙으면 카메라 코앞일 수 있다 (좁은 곳에서는 near 를 줄인다)
            continue
        sin_t = np.linalg.norm(np.cross(np.broadcast_to(d, u.shape), u), axis=1) / np.maximum(rb, 1e-6)
        el = np.degrees(np.arcsin(np.clip(u[:, 2] / rb, -1, 1)))
        pen = np.where(el < -65.0, 3.0, 1.0)                           # 수신기 바로 아래쪽은 잘 못 본다
        cost[i] = np.mean(np.minimum(np.radians(sig_deg) * rb / np.maximum(sin_t, 0.05) * pen, 3.0))
    return cost


def third_score(refs, centre, axis, P, frames, near=0.5):
    """셋째 자리부터: 지금 구간의 긴 쪽(axis)을 옆에서 보는 자리가 좋다. 구간 가운데에서 0.8~2m, 앞의 자리들과 다른 쪽. (클수록 좋다)"""
    hB = origin(frames); score = np.full(len(P), -np.inf)
    for i, p in enumerate(P):
        u = centre - (p + hB); r = np.linalg.norm(u)
        if r < near:
            continue
        side = np.linalg.norm(np.cross(u / r, axis))                  # 1 이면 긴 쪽을 정면으로 가른다
        new = min(np.linalg.norm(np.cross(u / r, (centre - m) / np.linalg.norm(centre - m))) for m in refs)   # 앞의 자리들과 다른 쪽에서 보는가
        score[i] = side + 0.5 * new - 0.25 * max(r - 2.0, 0.0) - 0.5 * max(0.8 - r, 0.0)
    return score


def pick(P, value, n=3, sep=0.5, largest=False):
    """값이 좋은 순서로, 서로 sep 넘게 떨어진 자리 n 개"""
    order = np.argsort(-value if largest else value); out = []
    for i in order:
        if not np.isfinite(value[i]):
            continue
        if all(np.linalg.norm(P[i] - P[j]) >= sep for j in out):
            out.append(int(i))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 장치에서 쓰는 묶음
class Session(object):
    """한 방에서 여러 자리를 재는 한 번의 탐색"""
    def __init__(self, lo=None, hi=None, step=STEP):
        self.lo = None if lo is None else np.asarray(lo, dtype=float); self.hi = None if hi is None else np.asarray(hi, dtype=float)
        self.step = step; self.measures = []; self.grid = None; self.prior = None

    def set_room(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float); self.hi = np.asarray(hi, dtype=float); self.grid = None; self.prior = None

    def small(self):
        """좁은 곳일수록 작아지는 배율 (짧은 변 2m 이상 = 1, 0.8m 이하 = 0.4). 자리 사이 거리, 벽에서 띄우는 거리, 선에서 떨어지는 거리에 곱한다."""
        g = self._grid()
        return float(min(1.0, max(0.4, min(g.L[0], g.L[1]) / 2.0)))

    def good_len(self):
        g = self._grid()
        return float(min(GOOD_LEN, GOOD_FRAC * max(g.L[0], g.L[1])))

    def add(self, pose, obs, frames, est=None):
        """pose = ((x, y, z), heading_deg). obs = Fusion.read() 가 준 스캔별 관측, frames = [(yaw, tilt), ...]. est = 그 자리에서 구한 방향(수신기 좌표, 없으면 None)"""
        self.measures.append(make_measure(pose, obs, frames, est)); self.grid = None
        return len(self.measures)

    def _grid(self):
        if self.grid is None:
            if self.lo is None or self.hi is None:                    # 방 크기를 모르면 잰 자리들 둘레 4m, 높이는 바닥에서 2.6m 까지
                B = np.array([m["pose"][0] for m in self.measures])
                self.lo = np.array([B[:, 0].min() - 4.0, B[:, 1].min() - 4.0, min(B[:, 2].min(), 0.0)])
                self.hi = np.array([B[:, 0].max() + 4.0, B[:, 1].max() + 4.0, min(B[:, 2].min(), 0.0) + 2.6])
            self.grid = Grid(self.hi - self.lo, self.step, self.lo)
        return self.grid

    def result(self, max_cells=1500):
        """지금까지 잰 것으로 구한 추정 구간. 폰에 그대로 보낼 수 있는 dict (숫자와 리스트만)"""
        n = len(self.measures)
        if n == 0:
            return dict(positions=0, verdict="none", message="아직 잰 자리가 없습니다.")
        g = self._grid(); post = posterior(g, self.measures, self.prior, Q_BAD, SIG_SCALE)
        c, idx, length, vol, conf, second = zone(g, post)
        mu, axis = long_axis(g, post, idx)
        seen = sum(1 for m in self.measures if any(o["seen"] for o in m["obs"]))
        good = bool(n >= 2 and length <= self.good_len() and conf >= GOOD_CONF)
        if seen == 0:
            verdict, msg = "nothing", "어느 자리에서도 신호가 가려지는 곳(골)이 보이지 않았습니다. 카메라가 영상을 보내는 중인지, 채널과 주소가 맞는지 확인하세요."
        elif good:
            verdict, msg = "good", "구간이 %.0fcm 로 좁혀졌습니다. 구간 안의 물건을 살펴보세요." % (100 * length)
        elif n >= MAX_POS:
            verdict, msg = "wide", "%d 자리를 재도 구간이 %.1fm 입니다. 반사가 심한 방일 수 있습니다. 구간 안과 둘레를 넓게 살펴보세요." % (n, length)
        else:
            verdict, msg = "more", ("한 자리로는 방향만 압니다. 권하는 자리로 옮겨 한 번 더 재세요." if n == 1 else "구간이 아직 %.1fm 입니다. 권하는 자리에서 한 번 더 재면 줄어듭니다." % length)
        cells = g.X[idx]; w = post[idx] / post[idx].max()
        if len(cells) > max_cells:                                       # 너무 많으면 그럴듯한 칸부터
            keep = np.argsort(-w)[:max_cells]; cells = cells[keep]; w = w[keep]
        out = dict(positions=n, verdict=verdict, message=msg, centre=[round(float(v), 3) for v in c], length_m=round(length, 2), volume_l=round(vol, 1),
                   confidence=round(conf, 2), second_peak=round(second, 2), axis=[round(float(v), 3) for v in axis],
                   box=dict(lo=[round(float(v), 2) for v in cells.min(0) - g.step / 2 - ZONE_PAD], hi=[round(float(v), 2) for v in cells.max(0) + g.step / 2 + ZONE_PAD]),
                   cell_m=g.step, pad_m=ZONE_PAD, cells=[[round(float(a), 2), round(float(b), 2), round(float(cc), 2), round(float(ww), 2)] for (a, b, cc), ww in zip(cells, w)],
                   rays=[dict(origin=[round(float(v), 3) for v in m["ref"]], dir=None if m["d"] is None else [round(float(v), 3) for v in m["d"]]) for m in self.measures],
                   room=dict(lo=[round(float(v), 2) for v in g.lo], hi=[round(float(v), 2) for v in g.lo + g.L]))
        out["next"] = [] if verdict in ("good", "wide", "nothing") else self.suggest(post, c, idx, axis)
        return out

    def suggest(self, post=None, c=None, idx=None, axis=None, n=3):
        """다음에 수신기를 놓으면 좋은 자리 n 곳 (좋은 순서). 가구가 있어 못 놓으면 다음 것을 쓴다."""
        g = self._grid(); ms = self.measures; z = float(ms[-1]["pose"][0][2]); frames = ms[-1]["frames"]
        k = self.small()
        P = spots(g.lo, g.lo + g.L, z, [m["pose"][0] for m in ms], min_sep=0.6 * k, margin=0.3 * k, step=0.25 if k >= 1.0 else 0.15)
        if len(P) == 0:
            return []
        if len(ms) == 1:
            if ms[0]["d"] is None:
                return []
            val = second_cost(ms[0]["ref"], ms[0]["d"], P, g.lo, g.lo + g.L, frames, near=0.5 * k); best = pick(P, val, n, sep=0.5 * k)
        else:
            if post is None:
                post = posterior(g, ms, self.prior, Q_BAD, SIG_SCALE); c, idx = zone(g, post)[:2]; _, axis = long_axis(g, post, idx)
            val = third_score([m["ref"] for m in ms], np.asarray(c), np.asarray(axis), P, frames, near=0.5 * k); best = pick(P, val, n, sep=0.5 * k, largest=True)
        return [dict(x=round(float(P[i][0]), 2), y=round(float(P[i][1]), 2), z=round(float(P[i][2]), 2)) for i in best]


# ---------------------------------------------------------------- 시험장 모드
EXAM_DESK_Z, EXAM_CHEST_Z = 0.80, 1.17      # 책상 위 (필통 등), 가슴 (단추) 높이 (m). sim/location/classroom.py 의 SeatGrid 와 같다
EXAM_BODY_BACK = 0.25                       # 의자에 앉은 몸은 자리 점에서 칠판 반대쪽으로 0.25m (몸 반지름 0.2 + 0.04 앞이 가슴 카메라)
EXAM_PLAN = "tilt18"                        # 시험장에서 한 곳에서 재는 계획 (숙임 30·54.7·78도 x 받침 6방향). 3번보다 옆자리까지 맞히는 비율이 크게 오른다
EXAM_MAX_POS = 6                            # 이만큼 재면 그만 (6곳에서 옆자리까지 96~97%: sim/location/perimeter2.py)
EXAM_GOOD_P = 0.6                           # 후보 3자리의 확률 합이 이만큼 넘고 EXAM_GOOD_N 곳을 쟀으면 '좁혀짐'
EXAM_GOOD_N = 6                             # 4곳에서 멈추면 3후보 안 88% , 6곳까지 재면 92~96% (sim/location/exam_mode_check.py). 확률 합은 늘 자신이 지나쳐서 곳 수로 정한다


class SeatPoints(object):
    """자리마다 카메라가 있을 법한 점들. X (N,3), owner (N,) = 자리 번호(0부터)"""
    def __init__(self, seats, front=(0.0, -1.0)):
        f = np.asarray(front, dtype=float); f = f / np.linalg.norm(f); side = np.array([-f[1], f[0]])
        pts, owner = [], []
        for k, s_ in enumerate(seats):
            c = np.array([float(s_["x"]), float(s_["y"])])
            for dx in (-0.25, 0.0, 0.25):
                for df in (0.25, 0.10):                                   # 책상 위: 자리 점에서 칠판 쪽으로 0.10·0.25m, 옆으로 -0.25·0·0.25m
                    q = c + df * f + dx * side; pts.append((q[0], q[1], EXAM_DESK_Z)); owner.append(k)
                q = c - (EXAM_BODY_BACK - 0.24) * f + 0.3 * dx * side; pts.append((q[0], q[1], EXAM_CHEST_Z)); owner.append(k)   # 가슴
        self.X = np.array(pts); self.owner = np.array(owner); self.n = len(seats)


def make_measure(pose, obs, frames, est=None):
    base = np.asarray(pose[0], dtype=float); head = float(pose[1])
    m = dict(pose=(base, head), obs=[dict(seen=bool(o["seen"]), psi=float(o["psi"]), depth=float(o["depth"])) for o in obs], frames=[(float(y), float(t)) for y, t in frames],
             ref=base + origin(frames), d=None)
    if est is not None:
        d = to_room(np.asarray(est, dtype=float), head); m["d"] = d / np.linalg.norm(d)
    return m


def spread_value(p, seats_xy, spot_xy, top_mass=0.9):
    """지금 자리 확률 p 일 때, 수신기를 spot 에 두면 후보 자리들이 얼마나 갈리는가 (클수록 좋다).
    확률 높은 자리들(합 top_mass 까지)끼리 그 자리에서 본 방향 차이를, 그 거리에서의 방향 오차(6도 + 2.5도/m)로 나눠 확률로 가중해 더한다 (3 에서 자른다)."""
    o = np.argsort(-p); k = int(np.searchsorted(np.cumsum(p[o]), top_mass) + 1); top = o[:max(k, 2)]; w = p[top]
    v = seats_xy[top] - np.asarray(spot_xy, dtype=float)[None, :2]
    az = np.arctan2(v[:, 1], v[:, 0]); d = np.hypot(v[:, 0], v[:, 1]); sig = np.radians(6.0 + 2.5 * d)
    da = np.abs((az[:, None] - az[None, :] + np.pi) % (2 * np.pi) - np.pi) / ((sig[:, None] + sig[None, :]) / 2)
    return float(np.sum(w[:, None] * w[None, :] * np.minimum(da, 3.0)))


class ExamSession(object):
    """시험장: 정해진 곳(모서리·벽·교탁 자리)에서만 재고, 답은 자리 번호 후보.
    seats: [{"id": "3-2", "x": m, "y": m}, ...]  자리 점 = 의자 앞 끝, 책상 뒤 끝쯤 (폰 AR 이나 좌석표로)
    spots: [{"name": "앞 왼쪽 모서리", "x": m, "y": m, "z": m}, ...]  수신기 받침 바닥을 놓을 수 있는 곳 (받침대 위, z 는 받침 바닥 높이)
    front: 학생이 보는 쪽 (칠판 쪽) 단위 벡터 (방 좌표 x, y)"""
    def __init__(self, seats, spots, front=(0.0, -1.0)):
        if not seats:
            raise ValueError("자리가 없습니다")
        if not spots:
            raise ValueError("수신기를 놓을 곳이 없습니다")
        self.seats = [dict(id=str(s_.get("id", i + 1)), x=float(s_["x"]), y=float(s_["y"])) for i, s_ in enumerate(seats)]
        self.spots = [dict(name=str(q.get("name", "곳 %d" % (i + 1))), x=float(q["x"]), y=float(q["y"]), z=float(q.get("z", 2.4))) for i, q in enumerate(spots)]
        self.front = (float(front[0]), float(front[1])); self.pts = SeatPoints(self.seats, self.front)
        self.xy = np.array([[s_["x"], s_["y"]] for s_ in self.seats]); self.measures = []; self.used = []

    def add(self, pose, obs, frames, est=None, spot=None):
        """pose = ((x, y, z), heading_deg). spot = 놓은 곳의 번호 (spots 안의 순서). 모르면 pose 에서 가장 가까운 곳으로 본다"""
        m = make_measure(pose, obs, frames, est); self.measures.append(m)
        if spot is None:
            b = m["pose"][0]; spot = int(np.argmin([np.hypot(q["x"] - b[0], q["y"] - b[1]) for q in self.spots]))
        self.used.append(int(spot))
        return len(self.measures)

    def seat_probs(self):
        if not self.measures:
            return np.full(len(self.seats), 1.0 / len(self.seats))
        post = posterior(self.pts, self.measures, None, Q_BAD, SIG_SCALE)
        p = np.bincount(self.pts.owner, post, minlength=len(self.seats)); return p / p.sum()

    def next_spot(self, p=None):
        """다음에 수신기를 놓을 곳 (spots 안의 번호). 아직 안 잰 곳 가운데 후보 자리들이 가장 벌어져 보이는 곳. 처음에는 0번"""
        free = [i for i in range(len(self.spots)) if i not in self.used]
        if not free:
            return None
        if not self.measures:
            return free[0]
        p = self.seat_probs() if p is None else p
        val = [spread_value(p, self.xy, (self.spots[i]["x"], self.spots[i]["y"])) for i in free]
        return free[int(np.argmax(val))]

    def result(self):
        n = len(self.measures); p = self.seat_probs(); o = np.argsort(-p)
        top = [dict(id=self.seats[i]["id"], x=round(self.seats[i]["x"], 2), y=round(self.seats[i]["y"], 2), p=round(float(p[i]), 3)) for i in o[:3]]
        mass = float(sum(t["p"] for t in top)); seen = sum(1 for m in self.measures if any(o_["seen"] for o_ in m["obs"]))
        nxt = self.next_spot(p)
        if n == 0:
            verdict, msg = "none", "아직 잰 곳이 없습니다. '%s'에 수신기를 놓고 재세요." % self.spots[nxt]["name"]
        elif seen == 0:
            verdict, msg = "nothing", "어느 곳에서도 골이 보이지 않았습니다. 카메라가 신호를 내는 중인지(ESP32 찌르기), 채널과 주소가 맞는지 확인하세요."
        elif n >= EXAM_GOOD_N and mass >= EXAM_GOOD_P:
            verdict, msg = "good", "후보 3자리: %s (합 %.0f%%). 이 자리들을 확인하세요." % (", ".join(t["id"] for t in top), 100 * mass)
        elif n >= EXAM_MAX_POS or nxt is None:
            verdict, msg = "wide", "%d곳을 재도 후보가 넓습니다 (후보 3자리 합 %.0f%%). %s 둘레를 넓게 확인하세요." % (n, 100 * mass, ", ".join(t["id"] for t in top))
        elif n >= 4:
            verdict, msg = "more", "지금 후보 3자리: %s. '%s' 쪽으로 옮겨 %d곳 더 재면 더 확실해집니다." % (", ".join(t["id"] for t in top), self.spots[nxt]["name"], EXAM_GOOD_N - n)
        else:
            verdict, msg = "more", "후보 3자리 합 %.0f%%. '%s' 쪽으로 옮겨 한 번 더 재세요." % (100 * mass, self.spots[nxt]["name"])
        out = dict(mode="exam", positions=n, verdict=verdict, message=msg, top=top, top_mass=round(mass, 3), plan=EXAM_PLAN,
                   seats=[dict(id=s_["id"], p=round(float(p[i]), 3)) for i, s_ in enumerate(self.seats)],
                   used=[self.spots[i]["name"] for i in self.used])
        out["next"] = None if (nxt is None or verdict in ("good", "wide", "nothing")) else dict(index=int(nxt), **self.spots[nxt])
        return out
