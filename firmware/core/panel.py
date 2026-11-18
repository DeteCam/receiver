# -*- coding: utf-8 -*-
"""패널(안테나 배열)로 방향과 자리를 재는 계산. 모터가 없다.

수신 체인이 하나(ESP32-C5)라 패킷마다 위상의 기준이 달라진다. 그래서 안테나 사이의 위상을 직접 잴 수 없다.
대신 기준 안테나의 신호와 고른 안테나의 신호를 합쳐서 '세기'만 잰다. 기준 길의 위상을 0·90·180·270도로 바꿔 네 번 재면

    I_p = |r + e^{j p} h|^2        (r = 기준 안테나, h = 고른 안테나,  p = 0, 90, 180, 270도)
    (I_0 - I_180) - j (I_90 - I_270) = 4 conj(r) h

로 '기준의 켤레 x 안테나'가 나온다 (위상 이동 간섭계와 같은 식). 안테나마다 같은 r 이 곱해지므로 안테나 사이의 위상 차이가 그대로 남는다.
이 파일은 파이썬 3.7 + numpy 1.16 에서 돈다.
"""
import numpy as np

C0 = 299792458.0
STEPS = np.array([0.0, 90.0, 180.0, 270.0])      # 기준 길의 위상 단계 (도). 실물의 값은 calibrate_steps 로 잰다
Q_BAD = 0.15                                     # 한 번 잰 것이 통째로 틀릴 가능성 (가려서 돌아온 전파)
BAND = (2.400e9, 2.4835e9)                       # 안테나 판(CHW1010-ANT2)의 패치와 16갈래 스위치가 받는 대역 (Hz)


def ch_freq(ch):
    """Wi-Fi 채널 번호 -> 가운데 주파수 (Hz). 1~13: 2.4 GHz, 14: 2484 MHz, 32~177: 5 GHz"""
    ch = int(ch)
    if 1 <= ch <= 13:
        return (2407 + 5 * ch) * 1e6
    if ch == 14:
        return 2484e6
    if 32 <= ch <= 177:
        return (5000 + 5 * ch) * 1e6
    raise ValueError("모르는 채널: %r" % ch)


def can_locate(ch, band=BAND):
    """이 채널의 기기를 지금 판으로 잴 수 있나. 판이 받지 못하는 대역(5 GHz)의 기기는 재지 않는다:
    재면 틀린 자리가 그럴듯한 점수와 함께 나온다 (sim/pre/band5_check.py). 채널을 모르면(0, None) 재지 않는다"""
    try:
        f = ch_freq(ch)
    except (TypeError, ValueError):
        return False
    return band[0] <= f - 10e6 and f + 10e6 <= band[1] + 1e6


def grid(rows=4, cols=4, pitch=0.0375):
    """안테나 자리 [M,3] (패널 좌표: x 오른쪽, y 보는 쪽, z 위). 스위치가 고르는 차례 = 아래 줄 왼쪽부터"""
    x = (np.arange(cols) - (cols - 1) / 2.0) * pitch; z = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    return np.array([[xx, 0.0, zz] for zz in z for xx in x])


def axes(bore, up=(0.0, 0.0, 1.0)):
    """패널이 보는 쪽(bore, 방 좌표)에서 패널 좌표축 (ex, ey, ez)"""
    ey = np.asarray(bore, float); ey = ey / np.linalg.norm(ey); ex = np.cross(ey, up); n = np.linalg.norm(ex)
    ex = ex / n if n > 1e-6 else np.array([1.0, 0.0, 0.0]); return ex, ey, np.cross(ex, ey)


def demod(I, steps=STEPS, gain=None):
    """세기 I [안테나, 4단계, 서브캐리어] -> Z [안테나, 서브캐리어] = conj(기준) x 안테나.
    gain [안테나, 4]: 패킷마다의 수신 이득(선형 세기 배율, ESP32 의 agc/fft 이득에서 구한 값). 주면 나눠서 맞춘다.
    steps 가 0·90·180·270 이 아니면 최소제곱으로 푼다 (실물 위상기의 오차를 받아 준다)."""
    I = np.asarray(I, float)
    if gain is not None:
        I = I / np.asarray(gain, float)[:, :, None]
    p = np.radians(np.asarray(steps, float))
    A = np.stack([np.ones_like(p), 2 * np.cos(p), -2 * np.sin(p)], 1)        # I = a + 2 Re(Z) cos p - 2 Im(Z) sin p,  Z = conj(r) h
    sol = np.linalg.pinv(A).dot(I.transpose(1, 0, 2).reshape(len(p), -1))    # [3, M*F]
    return (sol[1] + 1j * sol[2]).reshape(I.shape[0], I.shape[2])


def calibrate_steps(I, guess=STEPS, iters=20):
    """위상 단계의 실제 값을 잰다. 아는 곳에 둔 송신기(다른 패널이나 허브)를 여러 안테나로 잰 세기 I [안테나, 단계, 서브캐리어] 에서
    'Z 를 풀고 -> 단계를 다시 맞추는' 것을 되풀이한다. 첫 단계는 0도로 둔다."""
    I = np.asarray(I, float); steps = np.asarray(guess, float).copy(); Y = I.transpose(1, 0, 2).reshape(len(steps), -1)
    for _ in range(iters):
        p = np.radians(steps); A = np.stack([np.ones_like(p), 2 * np.cos(p), -2 * np.sin(p)], 1); sol = np.linalg.pinv(A).dot(Y)
        a, c, s = sol; B = np.stack([2 * c, -2 * s], 1); w = np.linalg.lstsq(B, (Y - a[None, :]).T, rcond=None)[0]      # 단계마다 (cos, sin)
        steps = np.degrees(np.arctan2(w[1], w[0])); steps = (steps - steps[0]) % 360.0
    return steps


def steer(X, spot, Rw, bore, lam):
    """후보 점 X [N,3] 을 볼 때 안테나마다의 위상 [N,M]. 패치의 세기 무늬(앞쪽 cos)를 곱한다"""
    d = np.asarray(X, float) - np.asarray(spot, float); d = d / np.linalg.norm(d, axis=1)[:, None]
    return np.exp(2j * np.pi * d.dot(Rw.T) / lam) * np.clip(d.dot(bore), 0.05, None)[:, None]


def score(Z, R, bore, spot, X, fc=2.437e9, neff=12.0, cal=None, band=BAND):
    """후보 점마다의 로그 점수 [N]. Z [안테나, 서브캐리어], R = grid() 의 자리, bore = 패널이 보는 쪽 (방 좌표), spot = 패널 자리.
    그 방향에서 온다고 볼 때 설명되는 세기의 비율 rho 로 -neff log(1 - rho). cal [안테나] = 안테나마다의 보정 (복소수)
    band: 판이 받는 대역. fc 가 그 밖이면 셈하지 않는다 (5 GHz 기기의 위치는 재지 않는다). 다른 판을 계산할 때만 None"""
    if band is not None and not (band[0] <= fc <= band[1]):
        raise ValueError("판이 받지 못하는 주파수 %.3f GHz: 위치를 재지 않는다 (판은 %.1f~%.1f MHz 용)" % (fc / 1e9, band[0] / 1e6, band[1] / 1e6))
    ex, ey, ez = axes(bore); Rw = R[:, :1] * ex + R[:, 1:2] * ey + R[:, 2:3] * ez
    Z = np.asarray(Z) if cal is None else np.asarray(Z) / np.asarray(cal)[:, None]
    Rm = Z.dot(Z.conj().T) / Z.shape[1]; a = steer(X, spot, Rw, ey, C0 / fc)
    p = np.einsum("nm,mk,nk->n", a.conj(), Rm, a).real; rho = p / np.maximum((np.abs(a) ** 2).sum(1) * np.trace(Rm).real, 1e-30)
    return -neff * np.log(np.maximum(1.0 - rho, 1e-4))


def bearing(Z, R, bore, spot, step=2.0, **kw):
    """가장 점수가 높은 방향 (방 좌표 단위 벡터)과 그 점수. 패널 앞쪽 반구만 본다"""
    az = np.radians(np.arange(0, 360, step)); el = np.radians(np.arange(-70, 31, step)); A, L = np.meshgrid(az, el)
    G = np.stack([np.cos(L) * np.cos(A), np.cos(L) * np.sin(A), np.sin(L)], -1).reshape(-1, 3); b = np.asarray(bore, float) / np.linalg.norm(bore)
    G = G[G.dot(b) > 0.1]; sc = score(Z, R, bore, spot, np.asarray(spot, float) + 6.0 * G, **kw); k = int(np.argmax(sc)); return G[k], float(sc[k])


class Seats(object):
    """자리마다의 후보 점 (책상 위 6 + 가슴 1 + 머리 1). owner[i] = 그 점의 자리 번호"""
    def __init__(self, seats, front=(0.0, -1.0)):
        f = np.asarray(front, float); f = f / np.linalg.norm(f); s = np.array([-f[1], f[0]]); P = []; own = []
        for k, st in enumerate(seats):
            c = np.array([st["x"], st["y"]], float)
            for (du, dv, z) in ((-0.2, 0.42, 0.80), (0.0, 0.42, 0.80), (0.2, 0.42, 0.80), (-0.2, 0.22, 0.80), (0.0, 0.22, 0.80), (0.2, 0.22, 0.80), (0.0, 0.05, 1.02), (0.0, 0.2, 1.22)):
                q = c + du * s + dv * f; P.append((q[0], q[1], z)); own.append(k)
        self.X = np.array(P); self.owner = np.array(own); self.n = len(seats)


class Session(object):
    """잰 것을 모아 자리 확률을 낸다. 카메라가 사진을 보낼 때마다 패널마다 add() 한다"""
    def __init__(self, seats, q=Q_BAD):
        self.pts = seats; self.q = q; self.logp = np.zeros(len(seats.X)); self.count = 0

    def add(self, sc):
        l = np.exp(sc - sc.max()); self.logp += np.log(l + self.q / (1.0 - self.q) * l.mean()); self.count += 1

    def post(self):
        p = np.exp(self.logp - self.logp.max()); p = np.bincount(self.pts.owner, p, minlength=self.pts.n); return p / p.sum()

    def top(self, n=3):
        p = self.post(); o = np.argsort(-p)[:n]; return [(int(i), float(p[i])) for i in o]
