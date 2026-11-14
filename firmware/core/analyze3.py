#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""여러 번 잰 스캔에서 카메라 방향(방위각, 높이각)을 구한다. 노트북에서 실행.

  python3 core/analyze3.py data/scan3_YYYYmmdd_HHMMSS [--mac ..] [--truth 35 -40] [--no-show]
  python3 core/analyze3.py --selftest        장치 없이, 가짜 데이터로 처음부터 끝까지 돌려 본다

폴더 안에 있어야 하는 것 (scan3.py 가 만든다)
  csi.pcap      nexmon CSI (파이에서 tcpdump 로 받은 것)
  motion.jsonl  스캔마다 한 줄: rx3.py 가 적은 자세(yaw, pitch)와 판 회전 시간표
  meta.json     채널, MAC 등
결과: result.json, result.png

방향 약속: 방위각은 받침 덮개의 앞 표시를 0도로 하고 위에서 볼 때 반시계가 +. 높이각은 수평이 0도, 위가 +, 바닥 쪽이 -.
세기 계산은 tools/manual_scan.py 와 같다 (패킷마다 CSI 전력으로 나누고 RSSI 를 더해 자동 이득 조절을 되돌린다).
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(rel_candidates, what, must=None):
    """HERE 에서 본 후보 폴더 가운데 처음으로 있는 것 (must 를 주면 그 파일이 들어 있는 것)"""
    for c in rel_candidates:
        p = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(p, must) if must else p):
            return p
    sys.exit("%s 를 찾지 못했습니다. 찾아본 곳: %s" % (what, ", ".join(rel_candidates)))


# 파이에서는 모든 파일이 한 폴더(~/detecam)에 있고, 저장소(firmware/)에서는 core / pi / host 폴더로 나뉘어 있다
SIM = os.environ.get("DETECAM_SIM") or _find([".", "../core"], "estimator.py 가 있는 폴더 (core)", "estimator.py")
PI = _find([".", "../pi"], "csi.py 가 있는 폴더 (pi)", "csi.py")
for _p in (SIM, PI):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from estimator import azel, frame, to_scan, unit          # noqa: E402
from fusion import Fusion, next_scan, consistency         # noqa: E402
import csi as CSI                                          # noqa: E402

BIN = 5.0                             # 판 각도를 5도 칸으로 묶는다


# ----------------------------------------------------------------- 패킷 -> 세기
def packet_levels(pcap, mac=None):
    """(시각 배열, 세기 행렬[패킷, 서브캐리어] dB, MAC). 계산은 pi/csi.py 에 있다 (tools/manual_scan.py 와 같은 계산)."""
    pk = CSI.read_csi(pcap)
    if not pk:
        sys.exit("CSI 패킷이 없습니다: " + pcap)
    t, C, mac = CSI.packet_levels(pk, mac)
    if not len(t):
        sys.exit("그 MAC 의 패킷이 없습니다: %s" % mac)
    return t, C, mac


# ----------------------------------------------------------------- 시간표 -> 판 각도
def plate_angle(rec, t):
    """시각 t 에서의 판 각도(도). rec 은 rx3.py 의 scan 출력."""
    k = np.interp(t, rec["t"], rec["k"])
    return (rec["spin0"] + 360.0 * k / rec["per_rev"]) % 360.0


def circ_smooth(x, n=3):
    k = np.ones(n) / n
    y = np.convolve(np.r_[x[-n:], x, x[:n]], k, mode="same")[n:-n]
    return y


def scan_curve(t, C, rec, bin_deg=BIN):
    """스캔 하나의 (판 각도 칸 가운데, 세기 곡선 dB, 칸마다 패킷 수). 빈 칸은 NaN."""
    m = (t >= rec["t"][0]) & (t <= rec["t"][-1])
    phi = plate_angle(rec, t[m]); c = C[m]
    nb = int(round(360.0 / bin_deg)); idx = np.minimum((phi / bin_deg).astype(int), nb - 1)
    V = np.full((nb, C.shape[1]), np.nan); cnt = np.zeros(nb, dtype=int)
    for b in range(nb):
        s = idx == b; cnt[b] = int(s.sum())
        if cnt[b]:
            V[b] = np.median(c[s], axis=0)
    centres = (np.arange(nb) + 0.5) * bin_deg
    if cnt.sum() < 20 or (cnt > 0).sum() < nb // 2:
        return centres, np.full(nb, np.nan), cnt
    db = V - np.nanmedian(V, axis=0)                      # 각도끼리 상대 비교 (DiffLoc 방식). 빈 칸은 NaN
    filled = np.where(np.isnan(db), 0.0, db)
    sm = np.stack([circ_smooth(filled[:, j]) for j in range(db.shape[1])], axis=1)
    contrast = np.median(sm, axis=0) - sm.min(axis=0)     # 골이 뚜렷한 서브캐리어를 고른다
    sel = np.argsort(-contrast)[: max(5, db.shape[1] // 3)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # 패킷이 없는 칸은 NaN 으로 둔다
        curve = np.nanmedian(db[:, sel], axis=1)
    return centres, curve, cnt


# ----------------------------------------------------------------- 방향 구하기
def solve(recs, t, C, truth=None, quiet=False):
    frames = [(r["yaw"], r["pitch"]) for r in recs]
    curves = []; counts = []
    for r in recs:
        ph, lv, cnt = scan_curve(t, C, r); curves.append((ph, lv)); counts.append(cnt)
    usable = [i for i, (ph, lv) in enumerate(curves) if np.isfinite(lv).sum() >= len(lv) // 2]
    fus = Fusion()
    res = {"frames": frames, "scans": [], "n_seen": 0, "az": None, "el": None}
    if not usable:
        if not quiet:
            print("패킷이 너무 적어 곡선을 만들지 못했습니다. 카메라가 영상을 보내는 중인지, MAC 과 채널이 맞는지 확인하세요.")
        return res, curves, None
    est, obs = fus.solve([curves[i] for i in usable], [frames[i] for i in usable])
    full = [None] * len(recs)
    for i, o in zip(usable, obs):
        full[i] = o
    for i, r in enumerate(recs):
        o = full[i]
        res["scans"].append({"yaw": r["yaw"], "pitch": r["pitch"], "packets": int(counts[i].sum()),
                             "seen": bool(o["seen"]) if o else False, "depth_db": round(o["depth"], 1) if o else None,
                             "centre_deg": round(o["psi"] % 360, 1) if o else None, "noise_db": round(o["noise"], 2) if o else None})
    res["n_seen"] = int(sum(s["seen"] for s in res["scans"]))
    if est is not None:
        az, el = azel(est)
        res["az"] = round(float(az), 1); res["el"] = round(float(el), 1)
        res["spread_deg"] = round(fus.spread(obs, [frames[i] for i in usable], est), 1)
        ok, detail = consistency(est, obs, [frames[i] for i in usable])                # 찾은 방향이 스캔 하나하나와 맞는지
        for i, d in zip(usable, detail):
            res["scans"][i]["fits"] = d["ok"]; res["scans"][i]["off_plane_deg"] = round(d["beta"], 1)
            if d["resid"] is not None:
                res["scans"][i]["centre_error_deg"] = round(d["resid"], 1)
        res["consistent"] = bool(ok and res["n_seen"] >= 2)
        if truth is not None:
            c = unit(truth[0], truth[1])
            res["truth"] = list(truth); res["error_deg"] = round(float(np.degrees(np.arccos(np.clip(est @ c, -1, 1)))), 1)
    if not quiet:
        describe(res)
    return res, curves, (obs, usable)


def describe(res):
    print("")
    for i, s in enumerate(res["scans"]):
        print("  스캔 %d  받침 %+6.1f도, 끄덕임 %5.1f도  패킷 %4d개  %s" % (
            i + 1, s["yaw"], s["pitch"], s["packets"],
            ("골 보임: 가운데 %5.1f도, 깊이 %.1f dB" % (s["centre_deg"], s["depth_db"])) if s["seen"] else
            ("골 안 보임" + (" (가장 낮은 곳 %.1f dB)" % s["depth_db"] if s["depth_db"] is not None else " (패킷 부족)"))))
    if res["az"] is None:
        print("\n결과: 어느 스캔에서도 골이 보이지 않았습니다. 판이 가리지 못하는 방향이거나(바로 아래), 신호가 반사로만 들어오는 경우입니다.")
        return
    conf = "두 스캔 이상에서 보임" if res["n_seen"] >= 2 else "한 스캔에서만 보임: 방향이 한 선 위 어딘가라 높이가 불확실 (한 번 더 재면 정해진다)"
    print("\n결과: 방위 %+.0f도, 높이 %+.0f도   [%s, 퍼짐 %.0f도]" % (res["az"], res["el"], conf, res.get("spread_deg", 0)))
    if res.get("consistent"):
        print("점검: 찾은 방향이 모든 스캔과 맞는다.")
    else:
        why = ["스캔 %d (%s)" % (i + 1, "골 위치가 %.0f도 어긋남" % s["centre_error_deg"] if s.get("centre_error_deg") is not None and s["seen"] else "보였어야 하는데 안 보임")
               for i, s in enumerate(res["scans"]) if s.get("fits") is False]
        print("점검: 스캔끼리 딱 맞지는 않는다%s. 틀렸을 수 있으니 장치를 30~60도 돌려 놓고 한 번 더 재서 같은 방향이 나오는지 본다." % ((" - " + ", ".join(why)) if why else " (골이 한 스캔에서만 보임)"))
    if "error_deg" in res:
        print("      실제 방위 %+.0f도, 높이 %+.0f도  ->  오차 %.1f도" % (res["truth"][0], res["truth"][1], res["error_deg"]))


def suggest_extra(res, extra, tilt):
    """한 스캔에서만 골이 보였을 때 다음에 잴 받침 각 (없으면 None)"""
    if res["n_seen"] != 1 or extra is None:
        return None
    obs, usable = extra
    frames = [tuple(res["frames"][i]) for i in usable]
    yaw, t = next_scan(obs, frames, tilt)
    return float(np.clip(yaw, -174.0, 174.0)), float(t)


# ----------------------------------------------------------------- 그림
def plot(res, curves, path, show=False, title=""):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    for f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f in x.name for x in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f; break
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for i, (ax, (ph, lv), s) in enumerate(zip(axes, curves, res["scans"])):
        y = lv - np.nanmedian(lv) if np.isfinite(lv).any() else lv
        ax.plot(ph, y, color="#2a78d6", lw=2)
        if s["seen"]:
            ax.axvline(s["centre_deg"], color="#eb6834", ls="--", lw=1.5)
        ax.set_title("스캔 %d: 받침 %+.0f°, 끄덕임 %.0f°\n%s" % (i + 1, s["yaw"], s["pitch"], ("골 %.0f°, 깊이 %.1f dB" % (s["centre_deg"], s["depth_db"])) if s["seen"] else "골 안 보임"), fontsize=10, loc="left")
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.grid(alpha=.3); ax.set_xlabel("판 각도 (°)")
    axes[0].set_ylabel("세기 (dB, 스캔 가운데값 기준)")
    head = title + ("  →  방위 %+.0f°, 높이 %+.0f°" % (res["az"], res["el"]) if res["az"] is not None else "  →  골이 보이지 않음")
    if "error_deg" in res:
        head += "  (실제 %+.0f°, %+.0f° · 오차 %.1f°)" % (res["truth"][0], res["truth"][1], res["error_deg"])
    fig.suptitle(head, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------- 가짜 데이터 (장치 없이 흐름 확인)
def fake_packets(recs, cam, seed=3, rate=80.0, shape="V", env="보통", nsc=52):
    """world.py 의 방 모형으로, rx3.py 가 적은 시간표에 맞춰 패킷을 만든다."""
    if not hasattr(np.random, "default_rng"):
        raise RuntimeError("가짜 데이터를 만들려면 numpy 1.17 이상이 필요합니다 (지금 %s). 가짜 모드의 '재기'는 노트북에서 해 보세요." % np.__version__)
    import world as mc
    rng = np.random.default_rng(seed)
    w = mc.World(np.random.default_rng(seed + 1), unit(cam[0], cam[1]), shape, env)
    gain = rng.uniform(0.6, 1.2, nsc)                     # 서브캐리어마다 골 깊이가 조금씩 다르다
    T = []; Cs = []
    for r in recs:
        t0, t1 = r["t"][0], r["t"][-1]
        n = rng.poisson(rate * (t1 - t0)); t = np.sort(rng.uniform(t0, t1, n))
        phi = plate_angle(r, t)
        fr = frame(r["yaw"] + rng.normal(0, 1.0), r["pitch"] + rng.normal(0, 1.0))
        E = np.zeros(n, dtype=complex)
        for k, (a, d) in enumerate(w.rays):
            psi, beta = to_scan(d, fr)
            f = w.look(beta, phi - psi)
            if k == 0:
                x = ((phi - psi) % 360.0) - 180.0
                f = f + w.back * np.exp(-(x / 45.0) ** 2) * np.cos(np.radians(beta)) ** 2
            E += a * f
        E += w.leak
        lv = 20 * np.log10(np.maximum(np.abs(E), 1e-4))
        T.append(t); Cs.append(-52.0 + lv[:, None] * gain[None, :] + rng.normal(0, 0.8, (n, nsc)))
    return np.concatenate(T), np.concatenate(Cs)


def selftest():
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "pi")))
    import rx3
    ok = 0; cases = [((35.0, -42.0), "tilt3"), ((-100.0, 5.0), "up+tilt3"), ((150.0, 40.0), "tilt3"), ((10.0, -15.0), "tilt4")]
    for cam, plan in cases:
        rig = rx3.Rig(dry=True)
        recs = [rig.scan(y, p) for y, p in rx3.PLANS[plan]]
        t, C = fake_packets(recs, cam, seed=int(abs(cam[0]) + 7))
        print("\n== 가짜 카메라: 방위 %+.0f도, 높이 %+.0f도  (계획 %s, 패킷 %d개)" % (cam[0], cam[1], plan, len(t)))
        res, curves, extra = solve(recs, t, C, truth=cam)
        nxt = suggest_extra(res, extra, rx3.TILT3)
        if nxt:
            print("  -> 한 스캔에서만 보여서 받침 %+.0f도에서 한 번 더 잰다" % nxt[0])
            recs.append(rig.scan(nxt[0], nxt[1])); t2, C2 = fake_packets(recs[-1:], cam, seed=99)
            res, curves, extra = solve(recs, np.r_[t, t2], np.vstack([C, C2]), truth=cam)
        ok += int(res.get("error_deg", 999) < 20)
    print("\n자체 시험: %d / %d 가 20도 안으로 들어왔습니다 (가짜 데이터라 숫자 자체에 큰 뜻은 없고, 흐름이 끝까지 도는지 보는 용도)." % (ok, len(cases)))
    return ok


def load(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "motion.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(outdir, "meta.json"))) if os.path.exists(os.path.join(outdir, "meta.json")) else {}
    return recs, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?"); ap.add_argument("--mac"); ap.add_argument("--truth", nargs=2, type=float, metavar=("방위", "높이"), help="실제 방위와 높이 (도). 예: --truth 35 -40")
    ap.add_argument("--no-show", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    if not a.outdir:
        ap.error("폴더를 주거나 --selftest 를 쓰세요")
    recs, meta = load(a.outdir)
    truth = tuple(a.truth) if a.truth else (tuple(meta["truth"]) if meta.get("truth") else None)
    t, C, mac = packet_levels(os.path.join(a.outdir, "csi.pcap"), a.mac or meta.get("mac"))
    res, curves, _ = solve(recs, t, C, truth=truth)
    res["mac"] = mac
    json.dump(res, open(os.path.join(a.outdir, "result.json"), "w"), ensure_ascii=False, indent=1)
    plot(res, curves, os.path.join(a.outdir, "result.png"), show=not a.no_show, title=os.path.basename(os.path.normpath(a.outdir)))
    print("저장:", os.path.join(a.outdir, "result.json"), os.path.join(a.outdir, "result.png"))
