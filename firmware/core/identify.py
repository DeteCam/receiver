# -*- coding: utf-8 -*-
"""판별: 사람이 '멈춤 / 움직임'을 되풀이할 때 어떤 기기의 전송량이 그 박자를 따라오는지 본다.
수신기(파이)와 시뮬레이션이 같은 파일을 쓴다. numpy 만 있으면 된다 (파이썬 3.7, numpy 1.16 에서도 돈다).

쓰는 법
  stim = plan_stimulus("walk")                  # 박자: 멈춤 8초, 움직임 8초를 3번 (0.5초 칸. 0 = 멈춤, 1 = 움직임)
  x = bins(times, t0, len(stim))                # 패킷이 온 시각들을 0.5초 칸으로 센다 (t0 = 박자를 시작한 시각). weights 에 길이를 주면 바이트 수가 된다
  r = decide(x, stim)                           # dict(camera, ratio, p, up, cycles, verdict, message)
판정
  움직임 구간과 멈춤 구간의 평균을 견준다 (바뀐 직후 1.5초는 뺀다. 카메라는 1초쯤 늦게 반응한다고 본다).
  같은 자료에서 박자만 엇갈리게 옮긴 것들과 견줘 '우연히 이만큼 차이 날 확률' p 를 구한다.
  p < 0.03 이고, 배율이 1.12 이상이고, 되풀이한 것 가운데 하나만 빼고 다 올랐으면 (두 번만 했으면 둘 다) '카메라'. 아니면 '반응 없음'(= 의심 기기로 남긴다).
  되풀이가 3번이면 p 가 0.03 아래로 내려가려면 박자가 꽤 또렷이 맞아야 한다: 잘못 잡는 일을 줄이려고 일부러 까다롭게 잡았다.
"""
import numpy as np

DT = 0.5                # 한 칸 (초)
LAG = 1.0               # 카메라가 반응하는 데 걸리는 시간 (초)
GUARD = 1.5             # 박자가 바뀐 직후 버리는 시간 (초)
P_LIM, RATIO_LIM = 0.03, 1.12
# 확인하는 방법 셋. still, move = 멈춤과 움직임의 길이(초), n = 되풀이, lead = 처음에 기다리는 시간, guard = 바뀐 직후 버리는 시간
# 숫자는 sim/ 의 identify_sim.py 로 고른 것이다 (8초씩 3번이 '1분 안에, 잘못 잡는 일은 적게'의 타협점).
PLANS = {
    "walk": dict(still=8.0, move=8.0, n=3, lead=4.0, guard=1.5, name="걸으며 확인", say=("가만히 서 있기", "팔을 크게 흔들며 걷기")),
    "light": dict(still=8.0, move=8.0, n=3, lead=4.0, guard=1.5, name="불을 껐다 켜며 확인", say=("불을 켠 채로 가만히 있기", "불을 1초에 한 번씩 껐다 켜기")),
    "leave": dict(still=20.0, move=15.0, n=2, lead=10.0, guard=4.0, name="방을 나갔다 들어오며 확인", say=("방 밖에서 기다리기 (문을 닫고)", "방에 들어와 걸어 다니기")),
}


def stimulus(still, move, n, lead=4.0, tail=3.0):
    """박자: 앞에 lead 초 멈춤, 그 뒤 (멈춤, 움직임) n 번, 끝에 멈춤 tail 초"""
    seq = [0] * int(lead / DT)
    for _ in range(int(n)):
        seq += [0] * int(still / DT) + [1] * int(move / DT)
    return np.array(seq + [0] * int(tail / DT), dtype=float)


def plan_stimulus(mode):
    p = PLANS[mode]
    return stimulus(p["still"], p["move"], p["n"], lead=p["lead"])


def phases(stim):
    """박자를 [(시작 칸, 끝 칸, 0 또는 1), ...] 로. 화면에 '지금 할 일'과 남은 시간을 보여 줄 때 쓴다"""
    out = []; a = 0
    for i in range(1, len(stim) + 1):
        if i == len(stim) or stim[i] != stim[a]:
            out.append((a, i, int(stim[a] > 0.5))); a = i
    return out


def bins(times, t0, n, weights=None):
    """시각들을 t0 부터 DT 간격의 n 칸으로 센다. weights (패킷 길이) 를 주면 칸마다 그 합"""
    t = np.asarray(times, dtype=float) - t0
    k = np.floor(t / DT).astype(int); ok = (k >= 0) & (k < n)
    w = None if weights is None else np.asarray(weights, dtype=float)[ok]
    return np.bincount(k[ok], weights=w, minlength=n).astype(float)


def medfilt3(x):
    """3칸 가운데값: I 프레임처럼 한 칸만 튀는 것을 누른다"""
    x = np.asarray(x, dtype=float); y = x.copy()
    if len(x) >= 3:
        y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def decide(counts, stim, pkts=None, guard=GUARD):
    """counts = 칸마다의 전송량 (패킷 수나 바이트 수), stim = 박자, pkts = 칸마다의 패킷 수 (counts 가 바이트일 때. '조용함'을 가리는 데 쓴다)"""
    x = medfilt3(counts); n = len(x); g = int(round(guard / DT)); lag = int(LAG / DT)
    stim = np.asarray(stim, dtype=float)
    edge = np.nonzero(np.diff(stim) != 0)[0] + 1
    use = np.ones(n, dtype=bool)
    for e in edge:
        use[e:e + g] = False
    s = np.concatenate([np.zeros(lag), stim])[:n]; use_s = np.concatenate([np.ones(lag, dtype=bool), use])[:n]

    def stat(sh):
        ss = np.roll(s, sh); uu = np.roll(use_s, sh)
        mv = x[(ss > 0.5) & uu]; st = x[(ss < 0.5) & uu]
        return float(mv.mean() - st.mean()) if len(mv) and len(st) else 0.0
    T = stat(0)
    period = int(np.mean(np.diff(edge))) * 2 if len(edge) > 2 else max(n // 2, 1)
    shifts = [k for k in range(3, n - 3) if abs(((k + period // 2) % period) - period // 2) > 2]      # 제 박자와 겹치는 자리는 뺀다
    null = np.array([stat(k) for k in shifts]) if shifts else np.array([T])
    p = float((np.sum(null >= T) + 1.0) / (len(null) + 1.0))
    mv = x[(s > 0.5) & use_s]; st = x[(s < 0.5) & use_s]
    ratio = float(mv.mean() / max(st.mean(), 1e-6)) if len(mv) and len(st) else 1.0
    up = 0; tot = 0
    for e in [q for q in edge if stim[q] > 0.5]:
        e2 = min([q for q in edge if q > e] + [n]); e0 = max([q for q in edge if q < e] + [0])
        a = x[min(e + lag + g, n - 1):min(e2 + lag, n)]; b = x[min(e0 + lag + g, n - 1):min(e + lag, n)]
        if len(a) and len(b):
            tot += 1; up += int(a.mean() > b.mean() * 1.05)
    cam = bool(p < P_LIM and ratio >= RATIO_LIM and up >= max(tot - 1, 2))
    rate = float(np.mean(counts if pkts is None else pkts) / DT)
    if rate < 5.0:
        verdict, msg = "quiet", "패킷이 거의 없습니다 (초당 %.0f개). 이 기기는 지금 영상을 보내고 있지 않습니다." % rate
    elif cam:
        verdict, msg = "camera", "움직일 때 전송량이 %.1f배로 늘었습니다 (%d번 가운데 %d번). 이 방을 찍고 있는 카메라로 보입니다." % (ratio, tot, up)
    else:
        verdict, msg = "unclear", "움직임에 뚜렷이 반응하지 않았습니다 (배율 %.2f). 불을 껐다 켜면서 다시 해 보고, 그래도 같으면 '의심 기기'로 두고 위치를 잽니다." % ratio
    if verdict == "quiet":
        cam = False
    return dict(camera=cam, verdict=verdict, message=msg, ratio=round(ratio, 2), p=round(p, 3), up=int(up), cycles=int(tot), rate=round(rate, 1))
