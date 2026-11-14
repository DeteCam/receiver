"""판별 단계 흉내: 사람이 '멈춤 / 움직임'을 되풀이할 때 카메라의 전송량이 그 박자를 따라오는지로 카메라를 가려낸다.
    python3 identify_sim.py [횟수]        결과: identify_result.json

묻는 것: 몇 초씩, 몇 번 되풀이해야 하나? 반응이 약한 카메라는 얼마나 놓치나? 카메라가 아닌 기기를 카메라라고 하는 일은 얼마나 되나?

넣은 것 (숫자는 논문에서: DiffLoc+ 표 3 = 초당 37~106개, DeWiCam = 움직인 뒤 약 1초 늦게 오른다, CSI:DeSpy = I 프레임이 주기적으로 튄다)
  카메라      기본 전송량 r0 = 초당 35~110 패킷. 움직임이 보이면 r0 x g (g = 반응 배율). 1초 늦게, 0.7초에 걸쳐 오르내린다.
              2~4초마다 I 프레임(한꺼번에 20~60 패킷). 통신 상태에 따라 전송량이 10~30초마다 ±15% 출렁인다. 패킷 수는 푸아송 + 15% 흔들림.
  카메라 아님  ① 화상 통화처럼 꾸준히 보내는 기기  ② 웹·동영상처럼 뭉텅이로 보내는 기기  ③ 다른 방의 카메라(내 움직임과 상관없이 가끔 오른다)
판정
  0.5초 칸으로 패킷 수를 세고(3칸 가운데값으로 I 프레임 튐을 누른다), 움직임 구간과 멈춤 구간의 평균을 견준다 (바뀐 직후 1.5초는 뺀다).
  같은 자료에서 박자만 엇갈리게 옮긴 것들과 견줘 '우연히 이만큼 차이 날 확률' p 를 구한다. p < 0.03 이고 배율 1.12 이상이고, 되풀이한 것의 대부분에서 올랐으면 '카메라'.
"""
import sys, os, json
import numpy as np
import paths                                   # 계산 코드(core)를 찾는다
from identify import DT, stimulus, decide, PLANS as DEVICE_PLANS       # 판정은 장치와 같은 코드

HERE = os.path.dirname(os.path.abspath(__file__))
# (멈춤, 움직임, 되풀이, 앞에 기다림, 바뀐 직후 버리는 시간, 카메라가 반응하는 데 걸리는 시간). 마지막 줄은 '방을 나갔다 들어오기': 문을 드나드는 데 시간이 걸려 늦게 반응한다고 본다
PLANS = {"10초씩 2번": (10.0, 10.0, 2, 4.0, 1.5, 1.0), "8초씩 3번 (장치 기본)": (8.0, 8.0, 3, 4.0, 1.5, 1.0), "5초씩 4번": (5.0, 5.0, 4, 4.0, 1.5, 1.0),
         "6초씩 5번": (6.0, 6.0, 5, 4.0, 1.5, 1.0), "4초씩 6번": (4.0, 4.0, 6, 4.0, 1.5, 1.0), "8초씩 4번": (8.0, 8.0, 4, 4.0, 1.5, 1.0),
         "나갔다 들어오기 (밖 20초, 안 15초) 2번": (20.0, 15.0, 2, 10.0, 4.0, 2.5)}
assert PLANS["8초씩 3번 (장치 기본)"][:5] == tuple(DEVICE_PLANS["walk"][k] for k in ("still", "move", "n", "lead", "guard"))
assert PLANS["나갔다 들어오기 (밖 20초, 안 15초) 2번"][:5] == tuple(DEVICE_PLANS["leave"][k] for k in ("still", "move", "n", "lead", "guard"))
GAINS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)


def response(stim, rng, lag=1.0, tau=0.7):
    """카메라가 실제로 반응하는 모양: lag 만큼 늦고 tau 로 매끄럽게"""
    n = len(stim); k = int(round(rng.uniform(0.6, 1.4) * lag / DT)); s = np.concatenate([np.zeros(k), stim])[:n]
    a = np.exp(-DT / tau); out = np.zeros(n); v = 0.0
    for i in range(n):
        v = a * v + (1 - a) * s[i]; out[i] = v
    return out


def drift(n, rng):
    """통신 상태에 따른 느린 출렁임 (10~30초마다 ±15%)"""
    out = np.ones(n); i = 0; lvl = 1.0
    while i < n:
        L = int(rng.uniform(10, 30) / DT); lvl = float(np.clip(lvl * np.exp(rng.normal(0, 0.15)), 0.6, 1.6)); out[i:i + L] = lvl; i += L
    return out


def camera(stim, g, rng, lag=1.0):
    n = len(stim); r0 = rng.uniform(35, 110)
    rate = r0 * (1 + (g - 1) * response(stim, rng, lag=lag)) * drift(n, rng)
    gop = rng.uniform(2.0, 4.0); ph = rng.uniform(0, gop); t = np.arange(n) * DT
    iframe = np.zeros(n); idx = ((np.arange(ph, n * DT, gop)) / DT).astype(int); idx = idx[idx < n]; iframe[idx] = rng.uniform(20, 60)
    lam = rate * DT * np.exp(rng.normal(0, 0.15, n)) + iframe
    return rng.poisson(np.maximum(lam, 0.1)).astype(float)


def steady(stim, rng):                     # 화상 통화: 꾸준함
    n = len(stim); r0 = rng.uniform(30, 120)
    return rng.poisson(r0 * DT * drift(n, rng) * np.exp(rng.normal(0, 0.2, n))).astype(float)


def bursty(stim, rng):                     # 웹·동영상: 뭉텅이
    n = len(stim); lam = np.full(n, rng.uniform(1, 8)); i = 0
    while i < n:
        i += int(rng.exponential(8.0) / DT) + 1; L = int(rng.uniform(1, 8) / DT)
        lam[i:i + L] += rng.uniform(40, 300)
        i += L
    return rng.poisson(lam[:n] * DT).astype(float)


def other_cam(stim, rng):                  # 다른 방의 카메라: 제멋대로 오른다
    n = len(stim); fake = np.zeros(n); i = int(rng.uniform(0, 20) / DT)
    while i < n:
        L = int(rng.uniform(3, 15) / DT); fake[i:i + L] = 1; i += L + int(rng.uniform(5, 40) / DT)
    return camera(fake, rng.uniform(1.3, 2.5), rng)


def run(n=600, seed=5):
    out = {}
    for name, (still, move, rep, lead, guard, lag) in PLANS.items():
        rng = np.random.default_rng(seed); stim = stimulus(still, move, rep, lead=lead)
        row = dict(seconds=len(stim) * DT, detect={}, false={})
        for g in GAINS:
            row["detect"]["%.1f" % g] = round(float(100.0 * np.mean([decide(camera(stim, g, rng, lag=lag), stim, guard=guard)["camera"] for _ in range(n)])), 1)
        for label, fn in (("꾸준한 기기", steady), ("뭉텅이 기기", bursty), ("다른 방 카메라", other_cam)):
            row["false"][label] = round(float(100.0 * np.mean([decide(fn(stim, rng), stim, guard=guard)["camera"] for _ in range(n)])), 1)
        out[name] = row
        print("%-18s 걸리는 시간 %2.0f초 | 잡는 비율(반응 배율별) %s | 카메라 아닌데 카메라라고 함 %s" % (name, row["seconds"], "  ".join("x%s %5.1f%%" % kv for kv in row["detect"].items()), row["false"]), flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    res = run(n)
    json.dump(dict(n=n, dt=DT, gains=GAINS, summary=res), open(os.path.join(HERE, "identify_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
