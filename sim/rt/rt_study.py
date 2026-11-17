"""수신기 구조와 읽는 법을 같은 전파 길 위에서 견준다. '스탠드 없이, 두 곳 이하'라는 조건에서 무엇이 정확도를 막는지 본다.

    python rt_study.py measure --recv plate uca8 ura4x4 ... [--n 160] [--jobs 11] [--sets kr40_real_s1 ...]
    python rt_study.py table   [--recv ...]                       곳 조합마다의 자리 맞힘 표
    python rt_study.py bearing [--recv ...]                       한 곳에서 잰 방향의 오차

받는 쪽 (recv)
  plate       지금 수신기: 안테나 하나 + 도는 알루미늄 판, 세기 곡선 맞추기 (rt_eval 의 tmpw 그대로). plate6 = 한 곳에서 6번만
  uca8        위상이 맞는 안테나 8개를 반지름 8cm 원에 (수직 쌍극자). cyl16 = 그 원 두 층
  ura4x4 ...  위상이 맞는 패치 안테나 배열을 벽에 걸어 자리 쪽을 보게 (간격 반 파장). ura4x8 = 가로 8 x 세로 4, ura8x8. ura4x4p37 = 간격 37mm
              뒤에 h 를 붙이면 (ura4x4h) 수신 체인 하나로: 기준 안테나와 합쳐 세기만 재서 위상을 되찾는다 (안테나마다 패킷 4개)
  sar20       안테나 하나를 반지름 20cm 로 돌리고 가운데 기준 안테나로 위상을 맞춘다 (48점, 차례로)
  oracle<s>   직접파 방향을 표준편차 s 도로 안다고 칠 때 (기하만의 한계). oracleaz<s> = 방위각만

모든 받는 쪽은 '후보 점마다의 로그 점수'를 내고, 자리 고르기는 장치 코드와 같은 방법(post_seat)으로 한다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse                        # noqa: E402
import numpy as np                                      # noqa: E402
import rt_eval as E                                     # noqa: E402
from rt_eval import sph, sph_v, rotz, K, SUB, C0, KIND, PLANS, post_seat, seat_rc   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "study"); LAM = C0 / 2.437e9
AIM = np.array([3.75, 4.85, 0.9])                       # 배열이 바라보는 곳: 자리들의 한가운데


class Points(K.SeatPoints):
    """장치의 후보 점(책상 위 6 + 가슴 1)에 머리 높이 점 하나를 더한다 (안경 카메라)"""
    def __init__(self, seats, front=(0.0, -1.0)):
        super().__init__(seats, front); f = np.asarray(front, float); f = f / np.linalg.norm(f)
        ex = np.array([[s_["x"] - 0.2 * f[0], s_["y"] - 0.2 * f[1], 1.22] for s_ in seats])
        self.X = np.vstack([self.X, ex]); self.owner = np.r_[self.owner, np.arange(len(seats))]


def geom(name):
    """(안테나 자리 [M,3] (배열 좌표: x 오른쪽, y 보는 쪽, z 위), 패치인가, 차례로 재는가)"""
    h = LAM / 2
    if name.startswith("uca"):
        m = int(name[3:]); r = h / (2 * np.sin(np.pi / m)); a = 2 * np.pi * np.arange(m) / m
        return np.stack([r * np.cos(a), r * np.sin(a), 0 * a], -1), False, False
    if name.startswith("cyl"):
        m = int(name[3:]) // 2; r = h / (2 * np.sin(np.pi / m)); a = 2 * np.pi * np.arange(m) / m; ring = np.stack([r * np.cos(a), r * np.sin(a), 0 * a], -1)
        return np.vstack([ring + [0, 0, h / 2], ring - [0, 0, h / 2]]), False, False
    if name.startswith("ura"):
        body, _, pm = name[3:].partition("p"); nr, nc = [int(v) for v in body.split("x")]     # 세로 줄 수 x 가로 칸 수, p<mm> = 안테나 간격 (없으면 반 파장)
        h = float(pm) / 1000.0 if pm else h
        x = (np.arange(nc) - (nc - 1) / 2) * h; z = (np.arange(nr) - (nr - 1) / 2) * h
        return np.array([[xx, 0.0, zz] for zz in z for xx in x]), True, False
    if name.startswith("sar"):
        r = float(name[3:]) / 100.0; m = 48; a = 2 * np.pi * np.arange(m) / m
        return np.stack([r * np.cos(a), r * np.sin(a), 0 * a], -1), False, True
    raise ValueError(name)


def face(spot, mount="aim", tilt=0.0):
    """배열 좌표축 (ex, ey, ez) 을 방 좌표로.
    aim   ey = 자리 한가운데를 똑바로 보는 쪽 (좌우·위아래로 꺾이는 받침)
    hinge 좌우로만 자리 쪽을 보고, 아래로는 tilt 도만 숙인다
    flush 가까운 벽에 등을 붙인다 (벽에서 0.3m 넘게 떨어진 곳은 hinge 와 같다). 아래로 tilt 도"""
    if mount == "aim":
        ey = AIM - spot; ey = ey / np.linalg.norm(ey)
    else:
        W, Dp, _ = ROOM; x, y = float(spot[0]), float(spot[1])
        dmin, nrm = min([(x, (1.0, 0, 0)), (W - x, (-1.0, 0, 0)), (y, (0, 1.0, 0)), (Dp - y, (0, -1.0, 0))], key=lambda t_: t_[0])
        if mount == "hinge" or dmin > 0.3:
            v = AIM - spot; v[2] = 0.0; nrm = v / np.linalg.norm(v)
        t = np.radians(tilt); ey = np.cos(t) * np.asarray(nrm, float) + np.sin(t) * np.array([0, 0, -1.0])
    ex = np.cross(ey, [0, 0, 1.0])
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([1.0, 0, 0])
    ex = ex / np.linalg.norm(ex); return ex, ey, np.cross(ex, ey)


def _rotv(w):
    """회전 벡터 (라디안) -> 회전 행렬"""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def pose_err(rng, spot, Rw, bore, pos_sd=0.0, yaw_sd=0.0, tilt_sd=0.0):
    """앱이 아는 수신기 자세 = 참 자세 + 오차. (자리, 안테나 자리, 보는 쪽) 을 오차가 든 것으로 돌려준다"""
    if not (pos_sd or yaw_sd or tilt_sd):
        return spot, Rw, bore
    b = bore if bore is not None else np.array([0, 1.0, 0]); ex = np.cross(b, [0, 0, 1.0]); ex = ex / max(np.linalg.norm(ex), 1e-9)
    w = np.radians(rng.normal(0, yaw_sd)) * np.array([0, 0, 1.0]) + np.radians(rng.normal(0, tilt_sd)) * ex + np.radians(rng.normal(0, tilt_sd)) * b
    Rm = _rotv(w); return spot + rng.normal(0, pos_sd, 3) * np.array([1, 1, 0.5]), Rw @ Rm.T, (Rm @ bore if bore is not None else None)


class Paths:
    """한 곳 - 한 카메라 점의 길. cj[P,2] = 수신기 쪽 편파(theta, phi) 계수"""
    def __init__(self, D, s, c, u, los_only=False):
        ok = D["ok"][s, c].copy()
        if los_only:
            ok &= D["nint"][s, c] == 0
        A = D["A"][s, c][ok].astype(complex); self.tau = D["tau"][s, c][ok].astype(float); self.body = D["body"][s, c][ok]; nint = D["nint"][s, c][ok]
        self.d, self.th, self.ph = sph(D["tht"][s, c][ok].astype(float), D["pht"][s, c][ok].astype(float))
        _, tr, pr = sph(D["thr"][s, c][ok].astype(float), D["phr"][s, c][ok].astype(float))
        self.cj = np.einsum("pi,pij->pj", np.stack([tr @ u, pr @ u], -1), A); self.fk = float(D["fc"]) + SUB
        self.spot = D["spots"][s].astype(float); self.cam = D["cams"][c].astype(float); v = self.cam - self.spot; self.c_true = v / np.linalg.norm(v)
        pw = (np.abs(self.cj) ** 2).sum(1); self.los = bool((nint == 0).any())
        self.k_db = float(10 * np.log10(pw[nint == 0].sum() / max(pw.sum(), 1e-30) + 1e-9)) if self.los else -99.0


def array_snap(rng, P, name, npk=20, dur=0.5, tv=1.0, snr=30.0, cal=(5.0, 0.5), holo=False, mount="aim", tilt=0.0):
    """배열이 받은 CSI. 돌려준다: (H [패킷, 안테나, 서브캐리어] 또는 SAR 이면 Z [안테나, 서브캐리어], 안테나 자리(방 좌표), 패치 보는 쪽 또는 None)"""
    R, patch, seq = geom(name); ex, ey, ez = face(P.spot, mount, tilt) if patch else (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    Rw = R[:, :1] * ex + R[:, 1:2] * ey + R[:, 2:3] * ez; M = len(R)
    pol = P.cj[:, 0] * (P.th @ ez) + P.cj[:, 1] * (P.ph @ ez)                   # 안테나는 배열의 세로축 편파
    g = np.clip(P.d @ ey, 0.0, None) if patch else np.ones(len(P.d))           # 패치: 앞쪽 반구, 세기 무늬 cos^2
    base = (pol * g)[:, None] * np.exp(-2j * np.pi * P.tau[:, None] * P.fk[None, :])              # [P, F]
    B = base[:, None, :] * np.exp(2j * np.pi * (P.d @ Rw.T)[:, :, None] * P.fk[None, None, :] / C0)   # [P, M, F] 길마다 안테나·서브캐리어의 응답
    n = 4 * M if holo else M if seq else npk; tt = np.sort(rng.uniform(0, dur, n)) if not (seq or holo) else np.linspace(0, dur, n)
    fb = np.ones((n, len(P.d)), complex)
    if tv > 0 and P.body.any():                                                 # rt_eval 과 같은 흔들림 (시간 축은 6.14초를 기준으로)
        nb = int(P.body.sum()); ph0 = rng.normal(0, 0.6 * tv, nb); amp = np.abs(rng.normal(0, 0.3 * tv, nb)); m = rng.integers(1, 4, nb); ps = rng.uniform(0, 2 * np.pi, nb)
        fb[:, P.body] = (1 + rng.normal(0, 0.15 * tv, nb))[None, :] * np.exp(1j * (ph0[None, :] + amp[None, :] * np.sin(2 * np.pi * m[None, :] * tt[:, None] / E.REV_S + ps[None, :])))
    ce = np.exp(1j * np.radians(rng.normal(0, cal[0], M))) * 10 ** (rng.normal(0, cal[1], M) / 20.0)       # 보정하고 남은 안테나마다의 오차
    def add_noise(H):
        sig = np.median(np.abs(H) ** 2); na = np.sqrt(max(sig, 1e-24) * 10 ** (-snr / 10) / 2)
        return H + na * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
    if holo:                                                                    # 수신 체인 하나: 기준 안테나 + (위상을 0·90·180·270도 돌린) 안테나 m 을 합쳐 세기만 잰다
        fbk = fb.reshape(M, 4, -1); Z = np.zeros((M, B.shape[2]), complex); ph = np.exp(1j * np.pi / 2 * np.arange(4))
        sig = np.median(np.abs(fb @ base) ** 2); na = np.sqrt(max(sig, 1e-24) * 10 ** (-snr / 10) / 2)
        for k in range(M):
            hk = np.einsum("jp,pf->jf", fbk[k], B[:, k, :]) * ce[k]; hr = fbk[k] @ base                    # [4, F]
            y = hr + ph[:, None] * hk + na * (rng.normal(size=hk.shape) + 1j * rng.normal(size=hk.shape))
            I = np.abs(y) ** 2 * 10 ** ((rng.normal(0, 0.5, (4, 1)) + rng.normal(0, 0.5, y.shape)) / 10.0)   # 패킷마다, 서브캐리어마다 0.5dB 흔들림
            Z[k] = ((I[0] - I[2]) - 1j * (I[1] - I[3])) / 4.0                                               # = 기준의 켤레 x 안테나 m
        return Z, Rw, (ey if patch else None)
    if seq:                                                                     # 차례로: 안테나 m 은 t_m 에, 기준 안테나(가운데)는 그때마다
        Hm = np.einsum("mp,pmf->mf", fb, B); Href = fb @ base
        Hm = add_noise(Hm) * ce[:, None]; Href = add_noise(Href)
        return Hm * np.exp(-1j * np.angle(Href)), Rw, (ey if patch else None)
    H = (fb @ B.reshape(len(P.d), -1)).reshape(n, M, -1)
    return add_noise(H) * ce[None, :, None], Rw, (ey if patch else None)


ROOM = (7.5, 9.0, 2.7)                                  # 교실 가로, 세로, 높이 (거울상에 쓴다)


def images(X, which):
    """후보 점의 거울상: c 천장, f 바닥, w 네 벽"""
    out = []; W, Dp, Hh = ROOM
    if "c" in which:
        out.append(X * [1, 1, -1] + [0, 0, 2 * Hh])
    if "f" in which:
        out.append(X * [1, 1, -1])
    if "w" in which:
        out += [X * [-1, 1, 1], X * [-1, 1, 1] + [2 * W, 0, 0], X * [1, -1, 1], X * [1, -1, 1] + [0, 2 * Dp, 0]]
    return out


def _steer(X, spot, Rw, bore):
    d = X - spot; d = d / np.linalg.norm(d, axis=1, keepdims=True); a = np.exp(2j * np.pi * (d @ Rw.T) / LAM)
    return a * np.clip(d @ bore, 0.05, None)[:, None] if bore is not None else a


def array_score(H, Rw, bore, spot, X, neff=12.0, music=False, img=""):
    """후보 점 X [N,3] 마다의 로그 점수. 설명되는 세기의 비율 rho 로 -neff*log(1-rho). img: 거울상 방향까지 한 묶음(부분공간)으로 본다"""
    a = _steer(X, spot, Rw, bore)                                               # [N, M]
    an = (np.abs(a) ** 2).sum(1)
    if img.startswith("x") and H.ndim == 3:                                     # 거울상 방향도 따로 본다: 반사로만 보일 때 그 자리에 점수를 준다 (0.7배)
        Rm = np.einsum("nmf,nkf->mk", H, H.conj()) / (H.shape[0] * H.shape[2]); tr = max(np.trace(Rm).real, 1e-30)
        rho = np.einsum("nm,mk,nk->n", a.conj(), Rm, a).real / np.maximum(an * tr, 1e-30)
        for Xi in images(X, img[1:]):
            ai = _steer(Xi, spot, Rw, bore); rho = np.maximum(rho, 0.7 * np.einsum("nm,mk,nk->n", ai.conj(), Rm, ai).real / np.maximum((np.abs(ai) ** 2).sum(1) * tr, 1e-30))
        return -neff * np.log(np.maximum(1 - rho, 1e-4))
    if img and H.ndim == 3:
        Rm = np.einsum("nmf,nkf->mk", H, H.conj()) / (H.shape[0] * H.shape[2]); tr = np.trace(Rm).real
        A = np.stack([a] + [_steer(Xi, spot, Rw, bore) for Xi in images(X, img)], -1)            # [N, M, K]
        G = np.einsum("nmk,nml->nkl", A.conj(), A) + 1e-3 * np.eye(A.shape[-1])[None] * an[:, None, None]
        B = np.einsum("nmk,mj,njl->nkl", A.conj(), Rm, A)                                          # A^H R A
        rho = np.einsum("nkl,nlk->n", np.linalg.inv(G), B).real / max(tr, 1e-30)
        return -neff * np.log(np.maximum(1 - rho, 1e-4))
    if H.ndim == 2:                                                             # SAR: 한 장
        y = a.conj() @ H; p = (np.abs(y) ** 2).mean(1); tot = (np.abs(H) ** 2).sum(0).mean()
        rho = p / np.maximum(an * tot, 1e-30); return -neff * np.log(np.maximum(1 - rho, 1e-4))
    Rm = np.einsum("nmf,nkf->mk", H, H.conj()) / (H.shape[0] * H.shape[2])
    if music:
        w, V = np.linalg.eigh(Rm); ns = max(1, len(w) - min(4, len(w) - 1)); En = V[:, :ns]
        q = (np.abs(a.conj() @ En) ** 2).sum(1) / an; return -neff * np.log(np.maximum(q, 1e-4))
    p = np.einsum("nm,mk,nk->n", a.conj(), Rm, a).real; rho = p / np.maximum(an * np.trace(Rm).real, 1e-30)
    return -neff * np.log(np.maximum(1 - rho, 1e-4))


APER = {"top12": list(range(12)), "top6": list(range(0, 12, 2)), "top4": [0, 4, 7, 11], "up12": list(range(12, 24)), "bot12": list(range(24, 36)),
        "top12up12": list(range(24)), "top12bot12": list(range(12)) + list(range(24, 36)), "top6up6": list(range(0, 12, 2)) + list(range(12, 24, 2)), "all36": list(range(36)),
        "mid6": list(range(3, 9)), "left6": list(range(6)), "top8": [0, 2, 3, 5, 6, 8, 9, 11]}


def aperture_trial(rng, D, c, u, elems, X, opt):
    """칠판 테두리에 늘어놓은 안테나들을 한 대의 큰 배열로 쓴다 (가까운 거리 초점 맞추기). 안테나는 차례로 재고, 기준 안테나(가운데 위)와의 위상 차를 쓴다.
    돌려준다: 후보 점마다의 로그 점수"""
    tv = opt.get("tv", 1.0); snr = opt.get("snr", 30.0); dur = opt.get("dur", 1.0); rep_n = int(opt.get("rep", 2)); ref = int(opt.get("ref", 5))
    ey = np.array([0.0, 1.0, 0.0]); ez = np.array([0.0, 0.0, 1.0]); fk = float(D["fc"]) + SUB
    def chan(s, tt):
        P = Paths(D, s, c, u); pol = P.cj[:, 0] * (P.th @ ez) + P.cj[:, 1] * (P.ph @ ez); g = np.clip(P.d @ ey, 0.0, None)
        base = (pol * g)[:, None] * np.exp(-2j * np.pi * P.tau[:, None] * fk[None, :]); fb = np.ones((len(tt), len(P.d)), complex)
        if tv > 0 and P.body.any():
            r3 = np.random.default_rng([int(rng.integers(1 << 30)), s]); nb = int(P.body.sum())
            ph0 = r3.normal(0, 0.6 * tv, nb); amp = np.abs(r3.normal(0, 0.3 * tv, nb)); m = r3.integers(1, 4, nb); ps = r3.uniform(0, 2 * np.pi, nb)
            fb[:, P.body] = (1 + r3.normal(0, 0.15 * tv, nb))[None, :] * np.exp(1j * (ph0[None, :] + amp[None, :] * np.sin(2 * np.pi * m[None, :] * tt[:, None] / E.REV_S + ps[None, :])))
        return fb @ base, P.los                                                  # [시각, 서브캐리어]
    M = len(elems); tt = np.linspace(0, dur, M * rep_n).reshape(rep_n, M)       # 안테나마다 rep_n 번, 돌아가며
    Z = np.zeros((rep_n, M, len(fk)), complex); nlos = 0
    href, _ = chan(ref, tt.reshape(-1)); href = href.reshape(rep_n, M, -1)
    for k, e in enumerate(elems):
        h, los = chan(e, tt[:, k]); nlos += los
        Z[:, k] = h
    sig = np.median(np.abs(Z) ** 2); na = np.sqrt(max(sig, 1e-24) * 10 ** (-snr / 10) / 2)
    Z = Z + na * (rng.normal(size=Z.shape) + 1j * rng.normal(size=Z.shape)); href = href + na * (rng.normal(size=href.shape) + 1j * rng.normal(size=href.shape))
    Z = Z * np.exp(-1j * np.angle(href)) * np.exp(1j * np.radians(rng.normal(0, opt.get("cal", 8.0), M)))[None, :, None]
    R = D["spots"][elems].astype(float) + rng.normal(0, opt.get("perr", 0.004), (M, 3))                 # 안다고 여기는 안테나 자리 (오차 포함)
    dist = np.linalg.norm(X[:, None, :] - R[None, :, :], axis=2)                                           # [N, M]
    w = np.clip((X[:, None, 1] - R[None, :, 1]) / dist, 0.05, None) / dist                               # 패치 무늬와 거리
    rref = np.linalg.norm(X - D["spots"][ref].astype(float), axis=1)                                      # 기준 안테나까지 (기준 위상을 뺐으므로 그 거리를 뺀다)
    kf = 2 * np.pi * fk / C0; score = np.zeros(len(X)); tot = (np.abs(Z) ** 2).sum((1, 2)).mean()
    for j in range(0, len(fk), 4):                                               # 서브캐리어 13개만 써도 같다
        a = w * np.exp(-1j * kf[j] * (dist - rref[:, None]))                    # [N, M]
        y = np.einsum("nm,rm->rn", a.conj(), Z[:, :, j]); score += (np.abs(y) ** 2).mean(0) / (np.abs(a) ** 2).sum(1)
    rho = score / max(tot / len(fk) * len(range(0, len(fk), 4)), 1e-30)
    return -opt.get("neff", 12.0) * np.log(np.maximum(1 - rho, 1e-4)), nlos / M


def grid_dirs(step=3.0):
    az = np.radians(np.arange(0, 360, step)); el = np.radians(np.arange(-66, 31, step)); A, L = np.meshgrid(az, el)
    return np.stack([np.cos(L) * np.cos(A), np.cos(L) * np.sin(A), np.sin(L)], -1).reshape(-1, 3)


def ang_err(d_est, d_true):
    """(방위각 오차, 높이각 오차, 3차원 각 오차) 도"""
    az = lambda v: np.degrees(np.arctan2(v[1], v[0])); el = lambda v: np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    da = (az(d_est) - az(d_true) + 180) % 360 - 180
    return float(da), float(el(d_est) - el(d_true)), float(np.degrees(np.arccos(np.clip(d_est @ d_true, -1, 1))))


_G = {}


def _init():
    _G["pat"] = E.Pattern(); _G["fus"] = E.Fusion(); _G["tm"] = E.TM(_G["pat"], 160.0); _G["D"] = {}; _G["G"] = grid_dirs()


def _load(name):
    if name not in _G["D"]:
        d = np.load(os.path.join(HERE, "paths", name + ".npz")); _G["D"][name] = {k: d[k] for k in d.files if k not in ("meta", "people")}
    return _G["D"][name]


def _trial(args):
    """한 카메라에 대해 모든 곳에서 잰 점수 [S, N] 와 방향 오차"""
    seed, name, kind, recv, opt = args
    D = _load(name); rng = np.random.default_rng(seed); seats_xy = D["seats"]; empty = set(D["empty"].tolist()) if "empty" in D else set()
    if len(seats_xy) != 40:                                                     # 교실(kr40)이 아니면 자리들의 한가운데를 바라보게 한다
        global AIM
        AIM = np.array([seats_xy[:, 0].mean(), seats_xy[:, 1].mean() + 0.2, 0.9])
    if len(empty) >= len(seats_xy):
        empty = set()
    while True:
        seat = int(rng.integers(len(seats_xy)))
        if seat not in empty:
            break
    c = seat * 4 + int(rng.choice(KIND[kind])); u = rng.normal(size=3); u /= np.linalg.norm(u)
    seats = [dict(id=str(k), x=float(x), y=float(y) + 0.225) for k, (x, y) in enumerate(seats_xy)]; pts = Points(seats); S = len(D["spots"])
    if recv.startswith("ap:"):                                                  # 칠판 테두리 배열: 설치 한 번이 '한 곳'
        r2 = np.random.default_rng([seed, 99]); sc1, flos = aperture_trial(r2, D, c, u, APER[recv[3:]], pts.X, opt)
        return seat, sc1[None].astype(np.float32), np.array([flos > 0.5]), np.array([flos], np.float32), np.full((1, 3), np.nan, np.float32)
    sc = np.zeros((S, len(pts.X)), np.float32); los = np.zeros(S, bool); kdb = np.zeros(S, np.float32); berr = np.full((S, 3), np.nan, np.float32)
    los_only = bool(opt.get("los_only")); want_b = seed % 1000 < opt.get("nbear", 0)
    for s in (opt.get("spots") or range(S)):                                    # opt["spots"]: 이 곳들만 잰다 (배울 자료를 많이 만들 때)
        r2 = np.random.default_rng([seed, s]); spot = D["spots"][s].astype(float)
        if recv.startswith("plate"):
            plan = "tilt6" if recv == "plate6" else "tilt18"; frames = PLANS[plan]
            w = E.RTWorld(r2, D, s, c, frames, _G["pat"], u, rate=opt.get("rate", 100.0), tv=opt.get("tv", 1.0), snr=opt.get("snr", 30.0), los_only=los_only)
            both = [w.scan(y, t) for y, t in frames]; pc = [b[1] for b in both]
            pose = (w.base + r2.normal(0, 0.05, 3) * np.array([1, 1, 0.5]), w.head + r2.normal(0, 3.0))
            _, b = _G["tm"].score(K.to_dev(pts.X, pose), [pc], frames); sc[s] = b[0]; los[s] = w.los; kdb[s] = w.k_db
            if want_b:
                Xg = spot + 6.0 * _G["G"]; _, bg = _G["tm"].score(K.to_dev(Xg, pose), [pc], frames); berr[s] = ang_err(_G["G"][int(np.argmax(bg[0]))], w.c_true)
        else:
            P = Paths(D, s, c, u, los_only); los[s] = P.los; kdb[s] = P.k_db
            if recv.startswith("oracle"):
                az_only = recv.startswith("oracleaz"); sig = float(recv[8:] if az_only else recv[6:])
                d = pts.X - spot; d = d / np.linalg.norm(d, axis=1, keepdims=True); t = P.c_true
                az = lambda v: np.degrees(np.arctan2(v[..., 1], v[..., 0])); el = lambda v: np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
                ea = r2.normal(0, sig) / max(np.cos(np.radians(el(t))), 0.2); ee = r2.normal(0, sig)
                da = (az(d) - az(t) - ea + 180) % 360 - 180; de = el(d) - el(t) - ee
                sc[s] = -0.5 * (da * np.cos(np.radians(el(t))) / sig) ** 2 - (0 if az_only else 0.5 * (de / sig) ** 2)
            else:
                nm, _, img = recv.partition("+"); music = nm.endswith("m"); nm = nm[:-1] if music else nm; holo = nm.endswith("h"); nm = nm[:-1] if holo else nm
                H, Rw, bore = array_snap(r2, P, nm, npk=opt.get("npk", 20), dur=opt.get("dur", 0.5), tv=opt.get("tv", 1.0), snr=opt.get("snr", 30.0), holo=holo,
                                         mount=opt.get("mount", "aim"), tilt=opt.get("tilt", 0.0))
                spot, Rw, bore = pose_err(r2, spot, Rw, bore, opt.get("pos_sd", 0.0), opt.get("yaw_sd", 0.0), opt.get("tilt_sd", 0.0))
                sc[s] = array_score(H, Rw, bore, spot, pts.X, opt.get("neff", 12.0), music, img)
                if want_b:
                    bg = array_score(H, Rw, bore, spot, spot + 6.0 * _G["G"], opt.get("neff", 12.0), music)
                    if bore is not None:
                        bg = np.where(_G["G"] @ bore > 0.1, bg, -1e9)                # 패치 배열은 뒤쪽(벽 속)을 보지 않는다
                    berr[s] = ang_err(_G["G"][int(np.argmax(bg))], P.c_true)
    return seat, sc, los, kdb, berr


def measure(a):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True); opt = json.loads(a.opt)
    with mp.Pool(a.jobs, initializer=_init) as pool:
        for recv in a.recv:
            t0 = time.time(); jobs = []; meta = []
            for kind in a.kinds:
                for i in range(a.n):
                    nm = a.sets[i % len(a.sets)]; g = i // len(a.sets) if a.same else i
                    jobs.append((1000 + g, nm, kind, recv, opt)); meta.append((kind, i % len(a.sets), g))
            rows = pool.map(_trial, jobs, chunksize=2)
            tag = recv + (a.tag and "_" + a.tag)
            np.savez_compressed(os.path.join(OUT, "sc_%s.npz" % tag), seat=np.array([r[0] for r in rows]), sc=np.stack([r[1] for r in rows]), los=np.stack([r[2] for r in rows]),
                                kdb=np.stack([r[3] for r in rows]), berr=np.stack([r[4] for r in rows]), kind=np.array([m[0] for m in meta]), setk=np.array([m[1] for m in meta]), grp=np.array([m[2] for m in meta]),
                                sets=np.array(a.sets), opt=a.opt)
            print("%-12s %d번  %.0f초" % (tag, len(rows), time.time() - t0), flush=True)


COMBOS = [("칠판 위 두 끝", (0, 1)), ("칠판 위 가운데 한 곳", (2,)), ("교탁 위 한 곳", (3,)), ("앞 두 모서리 1.5m", (4, 5)), ("분필받이 두 끝", (6, 7)), ("앞 두 모서리 2.2m", (17, 18)),
          ("칠판 위 왼쪽 + 뒷벽 오른쪽 2.2m", (0, 12)), ("칠판 위 가운데 + 뒷벽 가운데 1.8m", (2, 10)), ("옆벽 두 가운데 1.5m", (13, 14)), ("옆벽 두 가운데 2.2m", (15, 16)),
          ("앞 왼쪽 + 뒤 오른쪽 모서리 1.5m", (4, 9)), ("칠판 위 두 끝 + 뒷벽 두 곳 2.2m", (0, 1, 11, 12)), ("네 모서리 1.5m", (4, 5, 8, 9)), ("천장 가운데 한 곳", (19,))]


SPOT_NAMES = ["칠판 위 왼쪽 끝", "칠판 위 오른쪽 끝", "칠판 위 가운데", "교탁 위", "앞 왼쪽 모서리 1.5m", "앞 오른쪽 모서리 1.5m", "분필받이 왼쪽", "분필받이 오른쪽",
              "뒤 왼쪽 모서리 1.5m", "뒤 오른쪽 모서리 1.5m", "뒷벽 가운데 1.8m", "뒷벽 왼쪽 2.2m", "뒷벽 오른쪽 2.2m", "왼쪽 벽 가운데 1.5m", "오른쪽 벽 가운데 1.5m",
              "왼쪽 벽 가운데 2.2m", "오른쪽 벽 가운데 2.2m", "앞 왼쪽 모서리 2.2m", "앞 오른쪽 모서리 2.2m", "천장 가운데 2.6m"]


def _pts(name):
    D = np.load(os.path.join(HERE, "paths", name + ".npz")); seats_xy = D["seats"]
    return Points([dict(id=str(k), x=float(x), y=float(y) + 0.225) for k, (x, y) in enumerate(seats_xy)]), seats_xy


def score_combo(Z, pts, seats_xy, combo, temp=1.0):
    ok = near = top3 = 0; T = len(Z["seat"]); res = np.zeros((T, 3), bool)
    for t in range(T):
        p = post_seat(pts, [Z["sc"][t, s].astype(float) for s in combo], temp=temp); o = np.argsort(-p); seat = int(Z["seat"][t]); best = int(o[0])
        (r0, c0), (r1, c1) = seat_rc(seats_xy[seat], seats_xy), seat_rc(seats_xy[best], seats_xy)
        res[t] = (best == seat, abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1, seat in o[:3].tolist())
    return res


SPOT_NAMES2 = ["칠판 윗변 왼쪽 끝", "칠판 윗변 가운데", "칠판 윗변 오른쪽 끝", "교탁 위", "앞 창 쪽 모서리 삼각대 1.6m", "앞 복도 쪽 모서리 삼각대 1.6m",
               "창 유리 앞쪽 2.0m", "창 유리 가운데 2.0m", "창 유리 뒤쪽 2.0m", "창턱 가운데 1.05m", "앞문 위 2.0m", "복도 창턱 가운데 1.95m", "뒷문 위 2.0m",
               "사물함 위 왼쪽", "사물함 위 가운데", "사물함 위 오른쪽", "뒷벽 가운데 2.0m", "복도 창턱 앞쪽 1.95m", "복도 창턱 뒤쪽 1.95m", "창 유리 맨 앞 2.0m"]


SPOT_NAMES_L = ["왼쪽 벽 가운데 2.0m", "오른쪽 벽 가운데 2.0m", "왼쪽 벽 앞쪽 2.0m", "오른쪽 벽 앞쪽 2.0m", "왼쪽 벽 뒤쪽 2.0m", "오른쪽 벽 뒤쪽 2.0m",
                "칠판 윗변 가운데", "칠판 윗변 왼쪽 끝", "칠판 윗변 오른쪽 끝"]


def _names(Z):
    n = str(Z["sets"][0])
    return SPOT_NAMES2 if n.startswith("kr40_t") else SPOT_NAMES_L if n.startswith("lec72_t") else SPOT_NAMES


class Fast:
    """post_seat 와 같은 계산을 모든 시도에 한꺼번에. 곳마다의 로그 항을 미리 만들어 두고 조합은 더하기만 한다"""
    def __init__(self, Z, pts, seats_xy, temp=1.0):
        sc = Z["sc"].astype(np.float64); l = np.exp((sc - sc.max(-1, keepdims=True)) / temp); q = K.Q_BAD
        self.L = np.log(l + q / (1.0 - q) * l.mean(-1, keepdims=True)).astype(np.float32)                  # [T, S, N]
        self.O = np.zeros((len(pts.X), pts.n), np.float32); self.O[np.arange(len(pts.X)), pts.owner] = 1.0
        self.rc = np.array([seat_rc(xy, seats_xy) for xy in seats_xy]); self.seat = Z["seat"].astype(int)

    def judge(self, Ls, seat):
        P = np.exp(Ls - Ls.max(1, keepdims=True)) @ self.O; o = np.argsort(-P, 1)[:, :3]; b = o[:, 0]; rc = self.rc
        near = (np.abs(rc[b, 0] - rc[seat, 0]) <= 1) & (np.abs(rc[b, 1] - rc[seat, 1]) <= 1)
        return np.stack([b == seat, near, (o == seat[:, None]).any(1)], 1)

    def combo(self, combo):
        return self.judge(self.L[:, list(combo)].sum(1), self.seat)

    def tdiv(self, combo, idx, k):
        """idx [G, ns] = 같은 카메라를 사람 자세가 다른 ns 순간에 잰 시도 번호. 이어진 k 순간을 곱한다"""
        Lc = self.L[:, list(combo)].sum(1); G, ns = idx.shape; out = []
        for j in range(ns):
            Ls = sum(Lc[idx[:, (j + i) % ns]] for i in range(k)); out.append(self.judge(Ls, self.seat[idx[:, 0]]))
        return np.concatenate(out)


def table(a):
    import itertools
    for recv in a.recv:
        f = os.path.join(OUT, "sc_%s.npz" % (recv + (a.tag and "_" + a.tag)))
        if not os.path.exists(f):
            print("없음:", f); continue
        Z = np.load(f); Z = {k: Z[k] for k in Z.files}; pts, seats_xy = _pts(str(Z["sets"][0])); S = Z["sc"].shape[1]; kinds = list(dict.fromkeys(Z["kind"].tolist()))
        F = Fast(Z, pts, seats_xy, a.temp); NM = _names(Z); kd = Z["kind"]
        fmt = lambda res, kk: " | ".join("%s %3.0f/%3.0f/%3.0f" % (k, *(100 * res[kk == k].mean(0))) for k in kinds)
        print("\n== %s  (n=%d, 종류마다 %d)  정확히 / 옆자리까지 / 후보 3자리" % (recv, len(Z["seat"]), (kd == kinds[0]).sum()), flush=True)
        combos = [("칠판 테두리 배열 한 대", (0,))] if S == 1 else COMBOS if S >= 20 and NM is SPOT_NAMES else []
        if a.combo:
            combos = [(" + ".join(NM[int(i)] if S == len(NM) else str(i) for i in c.split(",")), tuple(int(i) for i in c.split(","))) for c in a.combo]
        for nm, combo in combos:
            print("%-44s " % nm[:44] + fmt(F.combo(combo), kd))
        if a.best:                                                              # 곳 a.best 개의 모든 조합을 견줘 좋은 차례로
            cand = [int(i) for i in a.cand] if a.cand else list(range(S)); rows = []
            for combo in itertools.combinations(cand, a.best):
                res = F.combo(combo); m = np.array([res[kd == k].mean(0) for k in kinds]); rows.append((m[:, 0].mean(), m[:, 0].min(), combo, res))
            rows.sort(key=lambda r: -r[0])
            print("-- %d곳 조합 %d가지 가운데 (세 종류의 '정확히' 평균이 높은 차례)" % (a.best, len(rows)))
            for avg, mn, combo, res in rows[:a.top]:
                print("%-44s " % " + ".join(NM[i] for i in combo)[:44] + fmt(res, kd) + "   평균 %.0f" % (100 * avg))
            rows.sort(key=lambda r: -r[1])
            print("-- 가장 약한 종류의 '정확히'가 높은 차례")
            for avg, mn, combo, res in rows[:max(3, a.top // 2)]:
                print("%-44s " % " + ".join(NM[i] for i in combo)[:44] + fmt(res, kd) + "   최소 %.0f" % (100 * mn))
        if a.tdiv and "grp" in Z:                                               # 같은 카메라를 여러 순간에 재서 곱한다 (걸어 두고 계속 잴 때)
            ns = len(Z["sets"]); key = np.array([kinds.index(k) for k in kd]) * 100000 + Z["grp"]; ug = np.unique(key)
            idx = np.array([[np.where((key == g) & (Z["setk"] == j))[0][0] for j in range(ns)] for g in ug if all(((key == g) & (Z["setk"] == j)).any() for j in range(ns))])
            kk = kd[idx[:, 0]]
            for c in (a.combo or ["0,1"]):
                combo = tuple(int(i) for i in c.split(","))
                for k in a.tdiv:
                    res = F.tdiv(combo, idx, int(k)); print("%-36s %2d순간  " % (" + ".join(NM[i] for i in combo)[:36], int(k)) + fmt(res, np.tile(kk, ns)))


def _mlp_fit(X, y, nc, hid=256, ep=80, lr=2e-3, wd=2e-3, seed=0):
    """작은 신경망 (은닉층 하나) 을 numpy 로. 돌려준다: 예측 함수 (로그 확률)"""
    rng = np.random.default_rng(seed); n, d = X.shape
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hid)); b1 = np.zeros(hid); W2 = rng.normal(0, np.sqrt(1.0 / hid), (hid, nc)); b2 = np.zeros(nc)
    P = [W1, b1, W2, b2]; M = [np.zeros_like(q) for q in P]; V = [np.zeros_like(q) for q in P]; t = 0
    for e in range(ep):
        o = rng.permutation(n)
        for i in range(0, n, 64):
            j = o[i:i + 64]; x = X[j] * (rng.random((len(j), d)) > 0.1)                     # 입력 일부를 가려 넘치게 외우지 않게
            h = np.maximum(x @ P[0] + P[1], 0); z = h @ P[2] + P[3]; z -= z.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            g = pr; g[np.arange(len(j)), y[j]] -= 1; g /= len(j)
            gW2 = h.T @ g; gb2 = g.sum(0); gh = (g @ P[2].T) * (h > 0); gW1 = x.T @ gh; gb1 = gh.sum(0)
            G = [gW1 + wd * P[0], gb1, gW2 + wd * P[2], gb2]; t += 1
            for k in range(4):
                M[k] = 0.9 * M[k] + 0.1 * G[k]; V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                P[k] = P[k] - lr * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
    def pred(Xq):
        z = np.maximum(Xq @ P[0] + P[1], 0) @ P[2] + P[3]; z -= z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
    return pred


def learn(a):
    """배운 모델: 사람 배치가 다른 교실들(디지털 트윈)에서 잰 '후보 점마다의 점수'로 자리를 맞히게 가르치고, 처음 보는 사람 배치에서 시험한다.
    합친 모델 = 물리 모델의 자리 확률(로그) + 배운 모델의 자리 확률(로그)"""
    for recv in a.recv:
        f = os.path.join(OUT, "sc_%s.npz" % (recv + (a.tag and "_" + a.tag)))
        if not os.path.exists(f):
            print("없음:", f); continue
        Z = np.load(f); Z = {k: Z[k] for k in Z.files}; pts, seats_xy = _pts(str(Z["sets"][0])); ns = len(Z["sets"]); te = Z["setk"] >= ns - a.ntest; tr = ~te
        kinds = list(dict.fromkeys(Z["kind"].tolist())); nseat = len(seats_xy); F = Fast(Z, pts, seats_xy); y = Z["seat"].astype(int); kt = Z["kind"][te]
        print("\n== %s  배운 모델 (가르친 것 %d, 시험 %d = 처음 보는 사람 배치 %d가지)  정확히 / 옆자리까지 / 후보 3자리" % (recv, tr.sum(), te.sum(), a.ntest), flush=True)
        def acc(L):
            o = np.argsort(-L, 1)[:, :3]; b = o[:, 0]; rc = F.rc; st = y[te]
            return np.stack([b == st, (np.abs(rc[b, 0] - rc[st, 0]) <= 1) & (np.abs(rc[b, 1] - rc[st, 1]) <= 1), (o == st[:, None]).any(1)], 1)
        for c in (a.combo or ["0,1", "2,15"]):
            combo = tuple(int(i) for i in c.split(",")); nm = " + ".join(_names(Z)[i] for i in combo)
            Ls = F.L[:, list(combo)].sum(1); Pp = np.exp(Ls - Ls.max(1, keepdims=True)) @ F.O; lpp = np.log(Pp / Pp.sum(1, keepdims=True) + 1e-9)       # 물리 모델의 자리 확률
            X = np.concatenate([1 - np.exp(-Z["sc"][:, s_, :].astype(np.float32) / 12.0) for s_ in combo] + [np.clip(lpp, -12, 0).astype(np.float32) / 6.0], 1)
            pred = _mlp_fit(X[tr], y[tr], nseat, ep=a.epochs); lm = pred(X[te])
            for lab, L in (("물리 모델", lpp[te]), ("배운 모델", lm), ("합친 모델", lm + lpp[te])):
                res = acc(L); print("%-30s %s  " % (nm[:30], lab) + " | ".join("%s %3.0f/%3.0f/%3.0f" % (k, *(100 * res[kt == k].mean(0))) for k in kinds), flush=True)


def bearing(a):
    for recv in a.recv:
        f = os.path.join(OUT, "sc_%s.npz" % (recv + (a.tag and "_" + a.tag)))
        if not os.path.exists(f):
            continue
        Z = np.load(f); Z = {k: Z[k] for k in Z.files}; B = Z["berr"]; ok = np.isfinite(B[:, :, 0]); los = Z["los"]
        print("\n== %s 한 곳에서 잰 방향의 오차 (도, 가운데값 | 90%%)   잰 것 %d" % (recv, ok.sum()))
        for nm, m in (("직접파 있음", ok & los), ("직접파 없음", ok & ~los), ("모두", ok)):
            if m.sum() < 5:
                continue
            e = B[m]; q = lambda x, p: np.percentile(np.abs(x), p)
            print("  %-10s n=%5d  방위각 %5.1f | %5.1f   높이각 %5.1f | %5.1f   3차원 %5.1f | %5.1f" % (nm, m.sum(), q(e[:, 0], 50), q(e[:, 0], 90), q(e[:, 1], 50), q(e[:, 1], 90), q(e[:, 2], 50), q(e[:, 2], 90)))
        if Z["sc"].shape[1] >= 20:
            for s in (range(Z["sc"].shape[1]) if _names(Z) is SPOT_NAMES2 else (0, 2, 3, 4, 6, 13, 15, 19)):
                m = ok[:, s]
                if m.sum() >= 5:
                    e = B[m, s]; print("  곳 %2d  직접파 %3.0f%%  방위각 %5.1f  높이각 %5.1f" % (s, 100 * los[m, s].mean(), np.median(np.abs(e[:, 0])), np.median(np.abs(e[:, 1]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--recv", nargs="+", default=["plate"]); ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--sets", nargs="+", default=["kr40_real_s%d" % i for i in range(1, 9)])
    ap.add_argument("--kinds", nargs="+", default=["책상", "가슴", "안경"]); ap.add_argument("--opt", default="{}"); ap.add_argument("--tag", default=""); ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--combo", nargs="*"); ap.add_argument("--ntest", type=int, default=2); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--same", action="store_true"); ap.add_argument("--tdiv", nargs="*"); ap.add_argument("--best", type=int, default=0); ap.add_argument("--cand", nargs="*"); ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(); {"measure": measure, "table": table, "bearing": bearing, "learn": learn}[a.cmd](a)
