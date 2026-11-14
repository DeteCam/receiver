"""재는 과정을 흉내 내는 모형 한 벌: 카메라 방향, 벽에서 튄 반사파, 판이 못 가리는 누설, 골 모양. 수신기의 가짜 모드(node.py --dry)와 시뮬레이션이 함께 쓴다.

흉내에 넣은 것
  - 직접파 + 벽·바닥·천장 반사 4개. 판은 반사파도 저마다 따로 가린다.
  - 판이 가리지 못하는 누설 성분, 판이 안테나 뒤에 왔을 때 세기가 3~4dB 오르는 현상(손 측정에서 봄)
  - 패킷 수(초당 50개, 한 바퀴 6.14초)와 패킷마다의 흔들림 0.8dB
  - 관절 각도 오차: 받침·끄덕임 1도, 판 0도 맞춤 2도
  - 골 모양 두 가지: W (회절 모형 그대로, 가운데가 볼록) / V (손 측정처럼 하나로 꺼지는 모양). 깊이·폭·가리는 범위는 실험마다 바꾼다.
넣지 않은 것: 편파, 파이 기판·모터·사람 몸의 영향, 카메라 전송량이 변하는 경우. 그래서 숫자는 낙관적일 수 있다.
실험을 돌리는 코드는 sim/ 의 montecarlo.py 에 있다.
"""
import numpy as np
from model import Plate
from estimator import unit, frame, to_scan

TILT3 = float(np.degrees(np.arctan(np.sqrt(2.0))))
REV_S = 4096 * 0.0015        # 한 바퀴 6.14초
BIN = 2.0                    # 곡선을 2도 칸으로 묶는다
ENV = {                      # 환경: 반사 세기(직접파 대비 dB 범위), 누설 최대값, 초당 패킷, 패킷 흔들림
    "보통": dict(k=(6.0, 14.0), leak=0.25, rate=50.0, pkt=0.8),
    "나쁨": dict(k=(1.0, 8.0), leak=0.45, rate=30.0, pkt=1.2),
}
PLATES = None


def plates():
    global PLATES
    if PLATES is None:
        PLATES = [Plate(dist=d) for d in (1500.0, 3000.0, 5000.0)]
    return PLATES


class World:
    """한 번의 실험: 카메라 방향, 반사파, 누설, 골 모양"""
    def __init__(self, rng, c, shape="W", env="보통"):
        e = ENV[env]; self.rng = rng; self.c = c; self.shape = shape; self.rate = e["rate"]; self.pkt = e["pkt"]
        self.P = plates()[int(rng.integers(3))]
        kdb = rng.uniform(*e["k"]); n = 4
        amp = rng.rayleigh(1.0, n); amp *= np.sqrt(10 ** (-kdb / 10) / np.sum(amp ** 2))
        az = rng.uniform(0, 360, n); el = np.clip(rng.normal(0, 22, n), -70, 70)
        el[0] = rng.choice([-1, 1]) * rng.uniform(25, 65)
        self.rays = [(1.0 + 0j, c)] + [(a * np.exp(1j * rng.uniform(0, 2 * np.pi)), unit(z, v)) for a, z, v in zip(amp, az, el)]
        self.leak = rng.uniform(0.0, e["leak"]) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        self.back = rng.uniform(0.15, 0.55)
        self.v = dict(D=rng.uniform(8.0, 13.0), sig=rng.uniform(18.0, 30.0), lo=rng.uniform(-38.0, -22.0), hi=rng.uniform(38.0, 52.0), w=rng.uniform(5.0, 9.0))

    def look(self, beta, dphi):
        if self.shape == "W":
            return self.P.look(beta, dphi)
        v = self.v
        s = 1.0 / (1.0 + np.exp(-(beta - v["lo"]) / v["w"])) / (1.0 + np.exp((beta - v["hi"]) / v["w"]))
        d = (np.asarray(dphi) + 180.0) % 360.0 - 180.0
        return 10 ** (-v["D"] * s * np.exp(-(d / v["sig"]) ** 2) / 20.0) + 0j

    def scan(self, yaw, tilt):
        rng = self.rng
        fr = frame(yaw + rng.normal(0, 1.0), tilt + rng.normal(0, 1.0)); zero = rng.normal(0, 2.0)
        phi = np.arange(0.0, 360.0, BIN) + BIN / 2
        E = np.zeros(len(phi), dtype=complex)
        for k, (a, d) in enumerate(self.rays):
            psi, beta = to_scan(d, fr)
            f = self.look(beta, phi + zero - psi)
            if k == 0:
                x = ((phi + zero - psi) % 360.0) - 180.0
                f = f + self.back * np.exp(-(x / 45.0) ** 2) * np.cos(np.radians(beta)) ** 2
            E += a * f
        E += self.leak
        lv = 20 * np.log10(np.maximum(np.abs(E), 1e-4))
        n = rng.poisson(self.rate * REV_S * BIN / 360.0, len(phi))
        lv = lv + rng.normal(0, 1, len(phi)) * self.pkt / np.sqrt(np.maximum(n, 1))
        lv[n == 0] = np.nan
        return phi, lv


def cameras(n, seed):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:                                    # 구 위에 고르게. 받침에 가리는 바로 밑(-80도 아래)은 뺀다
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        if v[2] > np.sin(np.radians(-80.0)):
            out.append(v)
    return np.array(out)
