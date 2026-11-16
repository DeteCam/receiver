"""수신기의 전파 모형 (NEC2 모멘트법). 머리 좌표로 만든다: 원점 = 파이 안테나, X = 끄덕임 축, Y = 기판 법선, Z = 판 회전축. 단위 mm.

무엇을 넣나 (금속만. 플라스틱은 전파에 거의 영향이 없어 뺀다)
  ant    파이 기판(접지면 85 x 56) + 안테나(기판 윗변의 홈을 가로지르는 먹임 선)
  conn   파이의 USB·랜 단자 덩어리
  head   머리 안: 판 회전 모터(모터 3), 드라이버 3장, 동글 2개, 점퍼선 다발, 모터 3 전선, 전원선(머리 안 구간)
  yoke   요크: 모터 1, 모터 2, 그 전선, 전원선(요크 구간)
  base   받침: 배터리
  plate  도는 금속판 (크기·모양은 바꿀 수 있다)
  cw     균형추 (100원 동전 더미)
  esp    ESP32 (2026-11-16): FPC 안테나(머리 천장, 파이 안테나 바로 위), 동축선, XIAO 기판(뚜껑 안쪽), 파이 GPIO 까지 점퍼선.
         먹이지 않는 쪽 안테나의 단자에는 50옴을 단다 (그 칩의 입력). esp_feed='fpc' 이면 FPC 를 먹이고 파이 안테나에 50옴.
먹임 선에 1V 를 걸어 먼 곳의 전기장을 구한다. 보내는 무늬와 받는 무늬는 같으므로(상반성), 이것이 곧 '그 방향에서 온 전파를 얼마나 세게 받는가'다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys
import numpy as np
HERE = empaths.HERE
import p2 as D
import necgeo as G

CELL = 7.0
TH = np.arange(0.0, 180.1, 5.0)          # 먼 곳 전기장을 구하는 방향 (머리 좌표의 극각, 방위각)
PH = np.arange(0.0, 360.0, 5.0)
CACHE = empaths.CACHE


def Rz(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; return T


def Ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); T = np.eye(4); T[0, 0] = c; T[0, 2] = s; T[2, 0] = -s; T[2, 2] = c; return T


def Tr(v):
    T = np.eye(4); T[:3, 3] = v; return T


def yoke_to_head(pitch):
    """요크 좌표의 점 -> 머리 좌표. parts2.pose_head 의 거꾸로: p_h = Rz(-90) Ry(-pitch) (p_y - (0,0,H_T)) - (0, -Y_T, -S_T)"""
    return Tr([0, D.Y_T, D.S_T]) @ Rz(-90) @ Ry(-pitch) @ Tr([0, 0, -D.H_T])


def world_to_head(yaw, pitch):
    return yoke_to_head(pitch) @ Rz(-yaw)


def ap(T, p):
    return (T @ np.r_[np.asarray(p, float), 1.0])[:3]


# ----------------------------------------------------------------- 부품
def add_board(m, feed="notch"):
    """파이 기판: 7mm 철망. 안테나는 윗변 한가운데(원점 근처)의 홈."""
    nx, nz = 8, 12; xs = np.linspace(D.BX0, D.BX1, nx + 1); zs = np.linspace(D.BZ0, D.BZ1, nz + 1); rad = 7.1 / (2 * np.pi)
    i0 = int(np.argmin(np.abs(xs)))                                        # x = 0 인 줄
    for j in range(nz + 1):
        for i in range(nx):
            if feed == "notch" and j == nz and i in (i0 - 1, i0):
                continue                                                  # 홈 입구: 먹임 선이 대신 지나간다
            if feed == "slot" and j in (nz, nz - 1, nz - 2) and i == i0 - 1:
                continue                                                  # 틈: 가로줄 세 개를 뺀다 (가운데 것은 먹임 선으로 다시 넣는다)
            m.wire((xs[i], 0, zs[j]), (xs[i + 1], 0, zs[j]), rad, 1, "ant")
    for i in range(nx + 1):
        for j in range(nz):
            if feed == "notch" and j == nz - 1 and i == i0:
                continue                                                  # 홈: 가운데 세로줄 한 칸을 뺀다
            m.wire((xs[i], 0, zs[j]), (xs[i], 0, zs[j + 1]), rad, 1, "ant")
    if feed == "notch":
        t = m.wire((xs[i0 - 1], 0, zs[nz]), (xs[i0 + 1], 0, zs[nz]), 0.5, 3, "ant"); m.feed = (t, 2)
    elif feed == "slot":                                                  # 윗변에서 안으로 판 틈(폭 7, 깊이 21)을 가로질러 먹인다: 실제 안테나(기판의 홈)와 가장 비슷한 꼴
        t = m.wire((xs[i0 - 1], 0, zs[nz - 1]), (xs[i0], 0, zs[nz - 1]), 0.5, 3, "ant"); m.feed = (t, 2)
    else:                                                                 # 견줘 보는 다른 모형: 기판에 수직으로 선 짧은 막대 (편파가 다르다)
        t = m.wire((xs[i0], 0, zs[nz - 1]), (xs[i0], -15.0, zs[nz - 1]), 0.5, 3, "ant"); m.feed = (t, 1)
    return xs, zs


def add_conn(m, zs):
    m.box(D.BX0, D.BX1, -16.4, 0.0, zs[0], zs[3], "conn", cell=7.0)


def add_head(m, xs, zs, cable=True, esp=False):
    m.cylinder((-D.M_OFFSET, 0, D.M1_Z0), (-D.M_OFFSET, 0, D.M1_ZT), D.M_BODY_R, "head", nphi=8, cell=9.5)            # 모터 3
    for k, zc in enumerate(D.DRV_ZC):                                                                            # 드라이버 3장 (구리가 꽉 찼다고 본다: 나쁜 쪽)
        m.sheet((16.0, 3.3, zc - 14.0), (32.0, 0, 0), (0, 0, 28.0), "head", cell=7.0)                               # 보드 사이 틈 1mm 는 철망으로 못 그려서 위아래 2mm 씩 줄였다
    for xc, node_x in ((D.BX0 + D.DONGLE["x"], 0.0), (D.BX0 + D.DONGLE2["x"], -21.0)):      # node_x: 단자 덩어리 아랫면의 이을 점                           # 동글 2개 (USB 단자에 이어져 있다)
        m.box(xc - 6, xc + 6, -16.8, -8.8, -105.0, -83.0, "head", cell=7.0)                                          # 폭 16 을 12 로: 두 동글 사이 틈 2mm 는 철망으로 못 그린다
        m.wire((xc - 6, -16.8, -83.0), (node_x, -16.4, zs[0]), 0.8, 1, "head")
    for jz, zc in ((2, D.DRV_ZC[0]), (5, D.DRV_ZC[1]), (8, D.DRV_ZC[2])):                                        # 점퍼선 다발 3개: 파이 GPIO -> 드라이버
        z0 = zs[len(zs) - 1 - jz]
        m.polyline([(xs[-1], 0, z0), (xs[-1] + 2, -11.0, z0), (30.0, -15.0, zc), (30.0, -4.0, zc)], 1.2, "head")
    m.polyline([(-27.0, -4.0, 22.0), (-20.0, -15.0, 9.0), (-20.0, -15.0, -6.0), (0.0, -21.0, -13.0), (24.0, -13.0, -22.0)], 1.0, "head")   # 모터 3 전선
    if esp:                                                                                                                               # 버튼 선: ESP32 가 있으면 XIAO 뒤(기판 쪽)로 돌아간다
        m.polyline([(-46.0, -22.0, -36.0), (-32.0, -7.0, -40.0), (6.0, -7.0, -40.0), (6.0, -20.0, -40.0)], 0.8, "head")
    else:
        m.polyline([(-46.0, -22.0, -36.0), (-30.0, -22.0, -46.0), (6.0, -20.0, -40.0)], 0.8, "head")                                     # 버튼 선
    if cable:                                                                                                                             # 전원선 (머리 안 구간): 파이 접지에 이어져 있다
        z0 = zs[len(zs) - 1 - 2]
        m.polyline([(xs[0], 0, z0), (-48.6, -4.0, z0), (-48.6, -4.0, -60.0), (-48.6, -4.0, -108.0)], 1.5, "head")


def add_yoke(m, pitch, cable=True):
    T = yoke_to_head(pitch)
    m.cylinder(ap(T, (-23.1, 32.5, 45.2)), ap(T, (-23.1, 32.5, 64.2)), D.M_BODY_R, "yoke", nphi=8, cell=9.5)               # 모터 1 (회전판 위)
    m.cylinder(ap(T, (-61.2, 67.0, 61.4)), ap(T, (-61.2, 86.0, 61.4)), D.M_BODY_R, "yoke", nphi=8, cell=9.5)               # 모터 2 (오른팔 바깥)
    bot = lambda x, y: (x, y, D.HZ0 - 8.0)                                                                             # 머리 아래로 나온 점 (머리 좌표)
    m.polyline([ap(T, (-74.0, 60.0, 52.0)), ap(T, (-60.0, 30.0, 58.0)), ap(T, (-40.0, 20.0, 75.0)), bot(30.0, -12.0), (30.0, -12.0, -92.0)], 1.0, "yoke")   # 모터 2 전선
    m.polyline([ap(T, (-36.0, 22.0, 56.0)), ap(T, (-30.0, 10.0, 78.0)), bot(22.0, -18.0), (22.0, -18.0, -60.0)], 1.0, "yoke")                                 # 모터 1 전선
    if cable:
        m.polyline([(-48.6, -4.0, -108.0), ap(T, (-34.0, -30.0, 70.0)), ap(T, (-10.0, -8.0, 56.0)), ap(T, (0.0, 0.0, 46.0))], 1.5, "yoke")                  # 전원선: 머리 아래 -> 상자 -> 가운데 통


def add_base(m, yaw, pitch, size="S", cable=True):
    T = world_to_head(yaw, pitch); b = D.BASE[size]; R = b["R"]; lx, ly = b["bw"] - 1.0, b["bl"] - 1.0; y0 = -R + D.BATT_IN + 0.5
    m.box(-lx / 2, lx / 2, y0, y0 + ly, 2.3, 2.3 + D.BATT_T, "base", cell=9.0, T=T)                                    # 배터리
    if cable:
        Ty = yoke_to_head(pitch)
        zc = max(26.0, 2.3 + D.BATT_T + 2.5)                                                                               # 선은 배터리 윗면보다 2.5 위로 지나간다
        m.polyline([ap(Ty, (0.0, 0.0, 46.0)), ap(T, (0.0, 0.0, zc)), ap(T, (0.0, y0 - 6.0, zc - 2.0 if zc == 26.0 else zc)), ap(T, (0.0, y0, 2.3 + D.BATT_T))], 1.5, "base")   # 전원선: 가운데 통 -> 배터리 앞 플러그


FPC = dict(x0=-19.5, x1=19.5, y=-2.2, z=7.0)     # FPC 안테나 (39 x 8 띠를 선 하나로). 실물은 z 8.3 (모터 3 밑면 10 과 1.7 차이)인데, 철망 선 굵기 때문에 1.3 낮췄다


def add_esp(m, xs, zs, feed="pi", fpc=None):
    """ESP32 부품. 돌려주는 것: FPC 먹임 (tag, seg)"""
    f = dict(FPC, **(fpc or {})); x0, x1, y, z = f["x0"], f["x1"], f["y"], f["z"]
    t = m.wire((x0, y, z), (x1, y, z), 0.5, 7, "esp")                                     # 가운데 조각(4)이 단자
    xb = -23.4; ys = np.linspace(-28.5, -10.7, 4); zz = np.linspace(-55.0, -36.0, 4)          # XIAO 접지면 (21 x 17.8 -> 19 x 17.8: 아래 변이 단자 덩어리에 1.8 까지 닿아 철망으로는 2 줄였다)
    m.sheet((xb, ys[0], zz[0]), (0, ys[-1] - ys[0], 0), (0, 0, zz[-1] - zz[0]), "esp", cell=7.0)
    cx = f.get("coax_end", (x0, y, z))
    m.polyline([cx, (cx[0] + 5.5, -24.0, cx[2] - 5.0), (xb, ys[1], zz[-1])], 0.6, "esp")      # 동축선 겉선: FPC 끝 -> 뚜껑 쪽 -> XIAO 안테나 단자 (윗변)
    m.polyline([(xb, ys[2], zz[-1]), (-6.0, -12.0, -28.0), (xs[-1] + 2.0, -12.0, zs[-2]), (xs[-1], 0, zs[-2])], 0.6, "esp")         # 점퍼선 5V -> GPIO 2번
    m.polyline([(xb, ys[3], zz[2]), (-8.0, -9.0, -36.0), (6.0, -7.0, -24.0), (xs[-1] + 2.0, -11.0, zs[-3])], 0.9, "esp")             # 점퍼선 GND·TX·RX 다발 -> GPIO 8·9·10번 (드라이버 다발의 시작점에 붙는다)
    return (t, 4)


def rot_pts(P, spin):
    c, s = np.cos(np.radians(spin)), np.sin(np.radians(spin)); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.asarray(P, float) @ R.T


def add_plate(m, spin, kind="flat", R=None, w=None, z0=None, z1=None, cell=7.5, **kw):
    """도는 판. kind: flat(평판), arc(원통면을 따라 휜 판), vee(가운데를 세로로 접은 판: ang > 0 이면 양옆이 바깥으로, < 0 이면 안테나 쪽으로),
    wings(평판 + 안테나 쪽으로 접은 옆날개), flange(평판 + 바깥으로 접은 옆날개), split(가운데 세로 틈), mesh(구멍이 큰 판 = 철망),
    slotv / sloth(가운데에 길이 L 의 세로 / 가로 틈), tilt(가로선 둘레로 눕힌 판)"""
    R = D.R_FOIL if R is None else R; w = D.PLATE_W if w is None else w; z0 = D.BL_Z0 if z0 is None else z0; z1 = z0 + 150.0 if z1 is None else z1
    def sh(o, u, v):
        o, u, v = rot_pts([o, u, v], spin); m.sheet(o, u, v, "plate", cell)
    if kind == "flat":
        sh((R, -w / 2, z0), (0, w, 0), (0, 0, z1 - z0))
    elif kind == "arc":                                    # 반지름 R 의 원통면 위, 호 길이 w
        n = max(4, int(round(w / cell))); a = np.linspace(-w / 2 / R, w / 2 / R, n + 1)
        for k in range(n):
            p0 = (R * np.cos(a[k]), R * np.sin(a[k]), z0); p1 = (R * np.cos(a[k + 1]), R * np.sin(a[k + 1]), z0)
            sh(p0, (p1[0] - p0[0], p1[1] - p0[1], 0), (0, 0, z1 - z0))
    elif kind == "vee":                                    # 가운데가 R, 양쪽이 ang 만큼 뒤(바깥)로 꺾임
        ang = np.radians(kw.get("ang", 30.0)); h = w / 2
        for s in (1, -1):
            sh((R, 0, z0), (np.sin(ang) * h, s * np.cos(ang) * h, 0), (0, 0, z1 - z0))
    elif kind == "wings":                                  # 평판 + 양옆에서 안테나 쪽으로 접은 날개 (깊이 d)
        d = kw.get("d", 30.0)
        sh((R, -w / 2, z0), (0, w, 0), (0, 0, z1 - z0))
        for s in (1, -1):
            sh((R, s * w / 2, z0), (-d, 0, 0), (0, 0, z1 - z0))
    elif kind == "split":                                  # 가운데에 세로 틈(폭 g)이 난 판: 같은 폭의 띠 두 장
        g = kw.get("g", 20.0); b = (w - g) / 2
        for y0 in (-w / 2, g / 2):
            sh((R, y0, z0), (0, b, 0), (0, 0, z1 - z0))
    elif kind == "flange":                                 # 평판 + 양옆을 바깥(안테나 반대쪽)으로 접은 날개 (깊이 d)
        d = kw.get("d", 15.0)
        sh((R, -w / 2, z0), (0, w, 0), (0, 0, z1 - z0))
        for s in (1, -1):
            sh((R, s * w / 2, z0), (d, 0, 0), (0, 0, z1 - z0))
    elif kind in ("mesh", "slotv", "sloth"):               # 철망(구멍이 큰 판) 또는 가운데에 틈(슬롯)을 낸 판. 선을 하나씩 놓는다
        c = kw.get("mesh", cell) if kind == "mesh" else cell
        nu = max(1, int(round(w / c))); nv = max(1, int(round((z1 - z0) / c)))
        ys = np.linspace(-w / 2, w / 2, nu + 1); zs = np.linspace(z0, z1, nv + 1)
        rad = kw.get("rad", 0.5) if kind == "mesh" else max(w / nu, (z1 - z0) / nv) / (2 * np.pi)
        L = kw.get("L", 60.0); zc = kw.get("zc", 0.0); iy = nu // 2; jz = int(np.argmin(np.abs((zs[:-1] + zs[1:]) / 2 - zc)))
        def wr(a, b):
            a, b = rot_pts([a, b], spin); m.wire(a, b, rad, 1, "plate")
        for i in range(nu + 1):                            # 세로 선
            for j in range(nv):
                zm = (zs[j] + zs[j + 1]) / 2
                if kind == "sloth" and j == jz and abs((ys[i]) - 0.0) < L / 2 and 0 < i < nu:
                    continue                               # 가로 틈: 그 줄의 세로 선을 뺀다
                wr((R, ys[i], zs[j]), (R, ys[i], zs[j + 1]))
        for j in range(nv + 1):                            # 가로 선
            for i in range(nu):
                if kind == "slotv" and i == iy and abs(zs[j] - zc) < L / 2 and 0 < j < nv:
                    continue                               # 세로 틈: 가운데 칸의 가로 선을 뺀다
                wr((R, ys[i], zs[j]), (R, ys[i + 1], zs[j]))
    elif kind == "tilt":                                   # 평판을 가운데 가로선 둘레로 tau 만큼 눕힌다 (위쪽이 바깥으로): 거울처럼 튀는 전파가 위아래로 빠진다
        tau = np.radians(kw.get("tau", 20.0)); h = z1 - z0; zc = (z0 + z1) / 2
        v = (h * np.sin(tau), 0, h * np.cos(tau)); sh((R - v[0] / 2, -w / 2, zc - v[2] / 2), (0, w, 0), v)
    else:
        raise ValueError(kind)


def add_cw(m, spin, n_coin=D.N_COIN):
    """균형추: 판 반대편 통에 든 100원 동전 n 개 (지름 24, 두께 1.5 x n 의 금속 원기둥으로 본다. 통의 막힌 쪽 = 위쪽부터 쌓인다). parts2.coins 와 같은 자리."""
    c, s = np.cos(np.radians(spin)), np.sin(np.radians(spin))
    p = lambda x, y, z: (c * x - s * y, s * x + c * y, z)
    m.cylinder(p(D.POD_XC, 0.0, D.POD_Z1 - D.COIN_T * n_coin), p(D.POD_XC, 0.0, D.POD_Z1), D.COIN_D / 2, "cw", nphi=8, cell=8.0)


def build(spin, parts=("ant", "plate"), yaw=0.0, pitch=D.TILT3, feed="notch", plate=None, size="S", cable=True, cw=None, esp_feed="pi", fpc=None, esp_load=50.0):
    m = G.Model(cell=CELL)
    if "ant" in parts:
        xs, zs = add_board(m, feed)
    elif feed == "mz":                                    # 기판 없이 작은 고리 하나 (Z 방향 자기 쌍극자: 수평 편파로 사방에 고르게)
        q = [(-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0)]; xs = zs = None; m.feed = []
        for k in range(4):                                # 네 변을 같은 방향으로 먹여 전류가 고르게 돈다
            t = m.wire(q[k], q[(k + 1) % 4], 0.3, 3, "ant"); m.feed.append((t, 2))
    else:                                                 # 기판 없이 짧은 막대 하나 (이론 모형과 견줄 때)
        ax = dict(z=(0, 0, 1), x=(1, 0, 0), y=(0, 1, 0))[feed]
        t = m.wire(tuple(-6.0 * np.array(ax)), tuple(6.0 * np.array(ax)), 0.4, 3, "ant"); m.feed = (t, 2); xs = zs = None
    if "conn" in parts:
        add_conn(m, zs)
    if "head" in parts:
        add_head(m, xs, zs, cable, esp="esp" in parts)
    if "yoke" in parts:
        add_yoke(m, pitch, cable)
    if "base" in parts:
        add_base(m, yaw, pitch, size, cable)
    if "esp" in parts:
        ft = add_esp(m, xs, zs, esp_feed, fpc)
        if esp_feed == "fpc":
            m.loads.append(m.feed + (50.0, 0.0)); m.feed = ft
        else:
            m.loads.append(ft + (float(esp_load), 0.0))                  # ESP32 가 꺼져 있거나 보내는 중이면 그 단자 임피던스가 50옴이 아니다 (esp_load 로 바꿔 본다)
    if "plate" in parts and spin is not None:
        add_plate(m, spin, **(plate or {}))
    if "cw" in parts and spin is not None:
        add_cw(m, spin, **(cw or {}))
    return m


def table(name, parts, spins=None, force=False, **kw):
    """판 각마다 푼 결과: E[spin, theta, phi, 2] (theta 성분, phi 성분). 한 번 푼 것은 cache 에 둔다."""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path) and not force:
        d = np.load(path); return d["spins"], d["E"]
    spins = np.arange(0.0, 360.0, 5.0) if spins is None else np.asarray(spins, float)
    E = np.zeros((len(spins), len(TH), len(PH), 2), dtype=np.complex64); nseg = 0
    for i, s in enumerate(spins):
        m = build(s, parts, **kw); et, ep, z = m.solve(TH, PH); E[i, :, :, 0] = et; E[i, :, :, 1] = ep; nseg = m.nseg
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, spins=spins, E=E, th=TH, ph=PH, nseg=nseg)
    return spins, E


def no_plate(name, parts, **kw):
    """판 없이 푼 것 (기준). E[theta, phi, 2]"""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        return np.load(path)["E"]
    m = build(None, [p for p in parts if p not in ("plate", "cw")], **kw); et, ep, z = m.solve(TH, PH)
    E = np.stack([et, ep], -1).astype(np.complex64); os.makedirs(CACHE, exist_ok=True); np.savez_compressed(path, E=E, th=TH, ph=PH, nseg=m.nseg, z=z)
    return E


if __name__ == "__main__":
    import time
    for parts in (("plate",), ("ant", "plate"), ("ant", "conn", "plate"), ("ant", "conn", "head", "plate"), ("ant", "conn", "head", "yoke", "plate"),
                  ("ant", "conn", "head", "yoke", "base", "plate", "cw")):
        for spin in (0.0, 135.0):
            t0 = time.time()
            try:
                m = build(spin, parts, feed="notch" if "ant" in parts else "z"); et, ep, z = m.solve(TH, PH)
                print("%-46s 판 %5.1f도: 조각 %4d개, %.1f초, 임피던스 %s" % ("+".join(parts), spin, m.nseg, time.time() - t0, np.round(z, 1)))
            except Exception as e:
                print("%-46s 판 %5.1f도: 실패 (%s) 조각 %d" % ("+".join(parts), spin, e, m.nseg))
