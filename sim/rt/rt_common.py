"""패널 연구(rt_study, room_study, rt_rss)가 함께 쓰는 상수와 작은 함수.

돌판 수신기의 계산(rt_eval.py, firmware/core 의 locate·fusion·estimator)에서 패널 쪽이 쓰던 것만 옮겨 왔다.
값은 그대로라서 옮기기 전과 같은 결과가 나온다. 돌판 쪽 코드는 receiver-legacy 저장소에 있다.
"""
import numpy as np

C0 = 299792458.0
REV_S = 4096 * 0.0015                                   # 사람 몸이 흔들리는 빠르기를 정하는 기준 시간 6.14초 (돌판 한 바퀴의 시간을 그대로 둔다. 바꾸면 결과가 달라진다)
SUB = np.r_[np.arange(-26, 0), np.arange(1, 27)] * 312.5e3           # 20 MHz 의 서브캐리어 52개 (Hz)
KIND = {"책상": (0, 1), "가슴": (2,), "안경": (3,)}                    # 자리마다 카메라 점 넷 가운데 어느 것이 그 종류인가
XS = 0.6 + np.arange(6) * 1.1 + 0.025; YS = 2.0 + np.arange(7) * 0.95  # 교실 40석의 칸(가로)과 줄(세로)

Q_BAD = 0.15                                            # 한 곳의 측정이 통째로 틀렸을 확률 (곳마다의 점수를 곱할 때 깔아 둔다)
EXAM_DESK_Z, EXAM_CHEST_Z = 0.80, 1.17                  # 책상 위 (필통 등), 가슴 (단추) 높이 (m)
EXAM_BODY_BACK = 0.25                                   # 의자에 앉은 몸은 자리 점에서 칠판 반대쪽으로 0.25 m (몸 반지름 0.2 + 0.04 앞이 가슴 카메라)


def rotz(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a); v = np.asarray(v, float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def sph(th, ph):
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    return np.stack([st * cp, st * sp, ct], -1), np.stack([ct * cp, ct * sp, -st], -1), np.stack([-sp, cp, np.zeros_like(sp)], -1)


def sph_v(v):
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return sph(np.arccos(np.clip(v[..., 2], -1, 1)), np.arctan2(v[..., 1], v[..., 0]))


def seat_rc(xy, seats_xy=None):
    """자리 점 -> (줄, 칸). 줄은 칠판에서 가까운 쪽부터"""
    xs, ys = (XS, YS) if seats_xy is None else (np.unique(np.round(seats_xy[:, 0], 3)), np.unique(np.round(seats_xy[:, 1], 3)))
    return int(np.argmin(np.abs(ys - xy[1]))), int(np.argmin(np.abs(xs - xy[0])))


def post_seat(pts, logs, q=Q_BAD, temp=1.0):
    """곳마다의 점수(로그)를 곱한다 (한 곳이 통째로 틀릴 가능성 q 를 깔아 둔다). 자리마다의 확률을 낸다"""
    p = np.ones(len(pts.X))
    for s in logs:
        l = np.exp((s - s.max()) / temp); p *= l + q / (1.0 - q) * l.mean(); p /= p.max()
    p = np.bincount(pts.owner, p, minlength=pts.n); return p / p.sum()


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
