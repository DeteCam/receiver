"""방 전체를 시간 영역 전파 계산(gprMax, FDTD)으로 풀어, 수신기가 카메라 방향을 맞히는지 처음부터 끝까지 본다 (2차원).

    python3 room2d.py [방 이름 ...]      -> room2d.json   (방 하나에 수십 분. 중간 결과는 room2d_cache/ 에 남아 이어서 돈다)

무엇을 하나
  - 방의 가로 단면(수신기 안테나 높이)을 2차원으로 푼다: 벽(벽돌·타일), 문(나무), 창(유리), 거울·철제 가구(금속), 나무 가구.
  - 수신기: 안테나 자리에 선 모양의 짧은 다이폴(세로 편파). 파이 기판(금속 띠 56mm)과 알루미늄 판(폭 100mm, 회전 반지름 80mm).
  - 판 각을 5도씩 72번 바꿔 가며 푼다. 판 각마다 한 번만 풀면 된다: 상반성(가역성) 때문에 '안테나에서 쏜 전파가 카메라 자리에 닿는 세기'가
    '카메라가 쏜 전파가 안테나에 닿는 세기'와 같아서, 방 안의 카메라 후보 자리 수십 곳을 한 번에 얻는다.
  - 카메라 자리마다 2.427~2.447GHz 의 16개 주파수(CSI 의 서브캐리어 대신)에서 세기를 꺼내, 수신기의 실제 코드
    (firmware/core 의 fusion.trough: 꺼진 구간의 가운데)로 방향을 구하고 참값과 견준다.
  - 같은 계산을 방 없이(벽 없음)도 해서 모형 자체가 맞는지 본다.
한계 (결과를 읽을 때)
  - 2차원이라 판이 위아래로 무한히 길다: 판 위아래로 돌아 들어오는 전파가 없어서 골이 실제보다 깊다. 방위각만 본다 (높이는 못 본다).
  - 기울여 세 번 재기가 아니라 '세워서 한 번'만 본다. 요크·모터·동전은 이 단면에 없어서 넣지 않았다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json, glob, time, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from fusion import trough
import h5py

DX0 = 0.005                                          # 격자 5mm (공기 중 파장의 1/25). 큰 방은 방마다 dx 로 6mm (1/21)
DX = DX0
F0 = 2.437e9
FREQS = np.linspace(2.427e9, 2.447e9, 16)            # 채널 6 의 폭 20MHz 를 16칸으로
PHI = np.arange(0.0, 360.0, 5.0)                     # 판 각 (머리 X 축에서, 위에서 볼 때 반시계)
ARM, HALF_W = 0.080, 0.050                           # 판 안쪽 면까지 반지름, 판 반폭
BOARD = (-0.042, -0.010)                             # 파이 기판의 접지면이 이 단면에서 머리 X 축을 따라 차지하는 범위 (안테나 = 0). 안테나 둘레 10mm 는 금속이 없는 자리로 둔다
CACHE = os.path.join(HERE, "room2d_cache")           # 판 각 하나씩의 계산 결과 (저장소에 넣지 않는다)
DATA = os.path.join(HERE, "room2d_data")             # 방·자리마다 판 각 72개를 모은 것 (저장소에 넣는다: 이것만 있으면 gprMax 없이 다시 평가한다)
# 재료: 상대 유전율, 전도율 (S/m). sim/room.py 의 값(2.437GHz 복소 유전율)을 그대로 바꿔 옮겼다: 전도율 = 허수부 x 2 pi f e0
MAT = dict(brick=(3.75, 0.038), tile=(6.0, 0.068), concrete=(5.31, 0.066), wood=(1.99, 0.012), glass=(6.27, 0.012))

ROOMS = {
    # 이름: 방 안쪽 크기 (m), 벽 재료, 벽 두께, 가구·창·문 [(종류, x0, y0, x1, y1, 재료)], 수신기 자리들 (x, y, 머리가 보는 쪽 yaw)
    "빈 공간": dict(size=(3.0, 3.0), wall=None, t=0.0, items=[], rx=[(1.5, 1.5, 0.0)]),
    "보통 방": dict(size=(3.5, 4.5), wall="brick", t=0.20, dx=0.006,
                   items=[("box", 0.0, 3.2, 0.6, 4.5, "wood_hollow"),       # 옷장 (속은 비었다)
                          ("box", 2.9, 0.0, 3.5, 0.6, "pec"),               # 철제 서랍장
                          ("gap", 1.2, 4.5, 2.4, 4.7, "glass"),             # 창 (벽 속)
                          ("gap", -0.2, 0.8, 0.0, 1.7, "wood")],            # 문 (벽 속)
                   rx=[(0.45, 2.25, 0.0), (1.75, 2.25, 0.0)]),
    "화장실": dict(size=(1.6, 2.2), wall="tile", t=0.15,
                  items=[("box", 0.0, 1.0, 0.01, 1.6, "pec"),               # 거울 (벽에 붙은 금속 판)
                         ("gap", 0.4, -0.15, 1.2, 0.0, "wood")],           # 문
                  rx=[(0.35, 0.6, 0.0), (0.8, 1.1, 0.0)]),
}
# 조건을 하나씩 바꿔 보는 화장실 (가운데 자리만): 거울을 뺀 것, 파이 기판을 뺀 것
ROOMS["화장실 (거울 없음)"] = dict(ROOMS["화장실"], items=[it for it in ROOMS["화장실"]["items"] if it[5] != "pec"], rx=[ROOMS["화장실"]["rx"][1]])
ROOMS["화장실 (기판 없음)"] = dict(ROOMS["화장실"], board=False, rx=[ROOMS["화장실"]["rx"][1]])
TAG = {"빈 공간": "free", "보통 방": "room", "화장실": "bath", "화장실 (거울 없음)": "bathnomirror", "화장실 (기판 없음)": "bathnoboard"}


def cams(R):
    """카메라 후보 자리: 방 안 0.25m 간격, 벽에서 0.12m, 수신기에서 0.6m 넘게, 가구 밖"""
    W, H = R["size"]; out = []
    for x in np.arange(0.12, W - 0.11, 0.25):
        for y in np.arange(0.12, H - 0.11, 0.25):
            if any(min(1e9, np.hypot(x - rx, y - ry)) < 0.6 for rx, ry, _ in R["rx"]):
                continue
            if any(k == "box" and x0 - 0.05 <= x <= x1 + 0.05 and y0 - 0.05 <= y <= y1 + 0.05 for k, x0, y0, x1, y1, m in R["items"]):
                continue
            out.append((round(float(x), 3), round(float(y), 3)))
    return out


def deck(name, R, rx, phi, cam):
    """gprMax 입력. 좌표는 방 안쪽 왼쪽 아래 귀가 (m, m) 이 되게 옮긴다 (m = 벽 두께 + 여유 0.15)"""
    W, H = R["size"]; t = R["t"]; m = t + 0.15
    X, Y = W + 2 * m, H + 2 * m
    s = lambda v: round(v / DX) * DX
    L = ["#title: %s phi %g" % (name, phi), "#domain: %g %g %g" % (s(X), s(Y), DX), "#dx_dy_dz: %g %g %g" % (DX, DX, DX),
         "#time_window: %g" % (1.0e-7 if name != "빈 공간" else 4e-8), "#waveform: gaussiandot 1 %g src" % F0]
    for k, (er, sg) in MAT.items():
        L.append("#material: %g %g 1 0 %s" % (er, sg, k))
    ax, ay, yaw = rx
    if R["wall"]:
        w = R["wall"]
        for (x0, y0, x1, y1) in ((-t, -t, W + t, 0), (-t, H, W + t, H + t), (-t, 0, 0, H), (W, 0, W + t, H)):
            L.append("#box: %g %g 0 %g %g %g %s" % (s(x0 + m), s(y0 + m), s(x1 + m), s(y1 + m), DX, w))
    for k, x0, y0, x1, y1, mat in R["items"]:
        if mat == "wood_hollow":                                   # 2cm 나무판으로 된 빈 상자
            for (a0, b0, a1, b1) in ((x0, y0, x1, y0 + 0.02), (x0, y1 - 0.02, x1, y1), (x0, y0, x0 + 0.02, y1), (x1 - 0.02, y0, x1, y1)):
                L.append("#box: %g %g 0 %g %g %g wood" % (s(a0 + m), s(b0 + m), s(a1 + m), s(b1 + m), DX))
        else:
            L.append("#box: %g %g 0 %g %g %g %s" % (s(x0 + m), s(y0 + m), s(x1 + m), s(y1 + m), DX, mat))
    ex = np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]); ey = np.array([-ex[1], ex[0]])    # 머리 X, 그에 직각 (위에서 볼 때 반시계)
    o = np.array([ax + m, ay + m])
    b0, b1 = o + ex * BOARD[0], o + ex * BOARD[1]                   # 파이 기판: 머리 X 축을 따라 (yaw 0 이면 세계 +Y 와 나란하다)
    bx0, by0 = s(min(b0[0], b1[0])), s(min(b0[1], b1[1]))
    if R.get("board", True):
        L.append("#box: %g %g 0 %g %g %g pec" % (bx0, by0, max(s(max(b0[0], b1[0])), bx0 + DX), max(s(max(b0[1], b1[1])), by0 + DX), DX))     # 두께는 한 칸
    p = np.radians(phi); r = np.cos(p) * ex + np.sin(p) * ey; tng = np.array([-r[1], r[0]])
    c = o + r * ARM                                                   # 판: 격자 칸으로 직접 그린다 (판 각마다 비스듬해도 틈 없이 이어지게 판 두께 쪽으로 반 칸씩 넓힌다)
    cells = set()
    for u in np.arange(-HALF_W, HALF_W + 1e-9, DX / 5):
        for v in (-DX / 2, 0.0, DX / 2):
            q = c + tng * u + r * v
            cells.add((int(np.floor(q[0] / DX)), int(np.floor(q[1] / DX))))
    for (i, j) in sorted(cells):
        L.append("#box: %g %g 0 %g %g %g pec" % (i * DX, j * DX, (i + 1) * DX, (j + 1) * DX, DX))
    L.append("#hertzian_dipole: z %g %g 0 src" % (s(o[0]), s(o[1])))
    for (cx, cy) in cam:
        L.append("#rx: %g %g 0" % (s(cx + m), s(cy + m)))
    return "\n".join(L) + "\n"


def run_one(name, R, ri, phi, cam):
    global DX
    DX = R.get("dx", DX0)
    os.makedirs(CACHE, exist_ok=True)
    tag = "%s_%d_%03d" % (TAG[name], ri, int(phi))
    npz = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(npz):
        return np.load(npz)["S"]
    fin = os.path.join(CACHE, tag + ".in")
    open(fin, "w").write(deck(name, R, R["rx"][ri], phi, cam))
    subprocess.run([sys.executable, "-m", "gprMax", fin, "-n", "1", "--allow-underresolved"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)   # 넓은 펄스의 높은 주파수 쪽만 격자가 성기다. 쓰는 2.4GHz 에서는 유리 속에서도 파장당 10칸
    fout = fin[:-3] + ".h5"                                           # gprMax 4 는 결과를 .h5 로 쓴다
    with h5py.File(fout, "r") as f:
        E = np.stack([f["rxs"]["rx%d" % (i + 1)]["Ez"][:] for i in range(len(cam))])
        dt = float(f.attrs["dt"]) * max(1, int(np.atleast_1d(f.attrs.get("rxsteps", 1))[0]))
        tt = np.arange(E.shape[1]) * dt
    K = np.exp(-2j * np.pi * FREQS[None, :] * tt[:, None])           # 주파수 16개에서의 성분 (DFT)
    S = np.abs(E @ K)                                                 # (카메라, 주파수)
    np.savez_compressed(npz, S=S)
    for f_ in (fin, fout):
        os.remove(f_)
    return S


def curve(A):
    """판 각 x 주파수의 세기 -> 수신기와 같은 방식의 곡선 (dB): 각도끼리 상대값, 골이 뚜렷한 주파수만 골라 가운데값"""
    db = 20 * np.log10(A + 1e-30)
    db = db - np.median(db, axis=0)
    n = 3; k = np.ones(n) / n
    sm = np.stack([np.convolve(np.r_[db[-n:, j], db[:, j], db[:n, j]], k, mode="same")[n:-n] for j in range(db.shape[1])], axis=1)
    contrast = np.median(sm, axis=0) - sm.min(axis=0)
    sel = np.argsort(-contrast)[: max(5, db.shape[1] // 3)]
    return np.median(db[:, sel], axis=1)


def evaluate(name, R, ri, cam, S):
    ax, ay, yaw = R["rx"][ri]; out = []
    for j, (cx, cy) in enumerate(cam):
        lv = curve(S[:, j, :])
        psi, depth, noise = trough(PHI, lv)
        az_est = (psi + 90.0 + yaw) % 360.0                           # 판이 카메라 쪽에 있을 때 꺼진다: 판 각 psi 의 방향 = 머리 X 에서 psi -> 세계 방위 psi + 90 + yaw
        az_true = np.degrees(np.arctan2(cy - ay, cx - ax)) % 360.0
        err = abs((az_est - az_true + 180.0) % 360.0 - 180.0)
        out.append(dict(cam=[cx, cy], dist=round(float(np.hypot(cx - ax, cy - ay)), 2), az_true=round(float(az_true), 1), az_est=round(float(az_est), 1),
                        err=round(float(err), 1), depth=round(float(depth), 1), noise=round(float(noise), 2), seen=bool(depth >= max(4.5, 3.0 * noise))))
    return out


def summary(rows):
    e = np.array([r["err"] for r in rows]); s = np.array([r["seen"] for r in rows])
    d = dict(n=len(rows), med=round(float(np.median(e)), 1), p90=round(float(np.percentile(e, 90)), 1), over30=round(float((e > 30).mean() * 100), 1),
             seen=round(float(s.mean() * 100), 1))
    if s.any():
        d.update(med_seen=round(float(np.median(e[s])), 1), over30_seen=round(float((e[s] > 30).mean() * 100), 1))
    return d


if __name__ == "__main__":
    args = sys.argv[1:]; part = None
    if args and args[0].startswith("--part="):                        # --part=k/n: 판 각 가운데 k 번째부터 n 개마다 하나씩 계산만 해 둔다 (코어마다 하나씩 띄운다). 맨 끝에 --part 없이 한 번 더 돌리면 모아서 쓴다
        k, n = map(int, args.pop(0).split("=")[1].split("/")); part = (k, n)
    names = args or list(ROOMS)
    path = os.path.join(HERE, "room2d.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for name in names:
        R = ROOMS[name]; cam = cams(R)
        for ri in range(len(R["rx"])):
            if part:
                for i, p_ in enumerate(PHI):
                    if i % part[1] == part[0]:
                        run_one(name, R, ri, float(p_), cam)
                print("%s 수신기 %d: 몫 %d/%d 끝" % (name, ri + 1, part[0], part[1]), flush=True)
                continue
            t0 = time.time(); data = os.path.join(DATA, "%s_%d.npz" % (TAG[name], ri))
            if os.path.exists(data):
                S = np.load(data)["S"].astype(float)
            else:
                S = np.stack([run_one(name, R, ri, float(p), cam) for p in PHI])      # (판 각, 카메라, 주파수)
                os.makedirs(DATA, exist_ok=True); np.savez_compressed(data, S=S.astype(np.float32), phi=PHI, freqs=FREQS, cams=np.array(cam), rx=np.array(R["rx"][ri]))
            rows = evaluate(name, R, ri, cam, S)
            key = "%s · 수신기 %d" % (name, ri + 1)
            res[key] = dict(rx=R["rx"][ri], summary=summary(rows), rows=rows, dx=R.get("dx", DX0), phi_step=float(PHI[1] - PHI[0]), n_freq=len(FREQS))
            print("%s: 카메라 %d곳, 방향 오차 가운데값 %.1f도, 90%% %.1f도, 30도 넘게 %.1f%%, 골이 보인 비율 %.0f%%  [%.0f초]" % (
                key, len(cam), res[key]["summary"]["med"], res[key]["summary"]["p90"], res[key]["summary"]["over30"], res[key]["summary"]["seen"], time.time() - t0), flush=True)
            json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
