# -*- coding: utf-8 -*-
"""빠른 시험 도우미. ESP32-C5 둘(보내는 쪽, 받는 쪽)을 노트북 USB 에 꽂고 쓴다.

    pip install pyserial numpy matplotlib
    python3 probe.py ports                               # 꽂힌 보드 찾기
    python3 probe.py setup TX포트 RX포트                  # 두 보드에 역할 주기 (한 번만)
    python3 probe.py t1 TX포트 RX포트                     # 시험 1: CSI 세기가 고른가 (약 6분)
    python3 probe.py t2 TX포트 RX포트                     # 시험 2 + 3: 간섭, 반사판, 돌린 무늬 (안내를 따라, 약 1시간)
    python3 probe.py report                              # 모은 것으로 다시 판정 (data/ 폴더)

기록은 data/ 에 쌓인다. 줄 형식은 ../c5/csi_probe/main/main.c 의 머리말.
판정 기준은 docs/빠른 시험.md 와 같다. 기준의 근거: 사전 검증(세기 흔들림), sim/pre/coupling.py (간섭).
"""
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BANDS = [("2.4", 6), ("5.2", 40), ("5.8", 157)]      # 시험에 쓰는 채널 (HT20)

# 판정 기준 (docs/빠른 시험.md)
T1_STEP_DB = 0.5          # 이어진 두 패킷의 세기 차 (가운데값)
T1_SPREAD_DB = 2.0        # 네 패킷(위상 한 번 재기) 안의 세기 폭 (90%)
T1_RATE = 50.0            # 초당 받은 패킷
T2_FRONT_OK = 4.0         # 이웃이 있을 때 정면 세기 변화가 이 안이면 보정 없이도 된다 (결합 -20 dB 쯤)
T2_FRONT_BAD = 10.0       # 이보다 크게 줄면 안테나가 크게 어긋난 것 (감도가 모자랄 수 있다)
T3_RIPPLE_OK = 3.0        # 돌린 무늬의 출렁임 (이웃 있음 / 없음, 봉우리-골)
T3_FB_OK = 10.0           # 반사판을 단 뒤 앞뒤 차


# ---------------------------------------------------------------- 시리얼
def open_port(port):
    import serial
    s = serial.Serial(port, 115200, timeout=0.2)
    time.sleep(0.3)
    s.reset_input_buffer()
    return s


def send(s, cmd, wait=0.3):
    s.write((cmd + "\n").encode())
    s.flush()
    time.sleep(wait)


class Reader(object):
    """받는 쪽의 줄을 뒤에서 계속 읽는다. record() 동안의 C 줄만 모은다"""
    def __init__(self, s):
        self.s = s; self.buf = None; self.info = []; self.stop = False
        self.t = threading.Thread(target=self.run); self.t.daemon = True; self.t.start()

    def run(self):
        part = b""
        while not self.stop:
            try:
                d = self.s.read(4096)
            except Exception:
                time.sleep(0.2); continue
            if not d:
                continue
            part += d
            *lines, part = part.split(b"\n")
            for ln in lines:
                ln = ln.decode(errors="ignore").strip()
                if ln.startswith("C,"):
                    if self.buf is not None:
                        self.buf.append(ln)
                elif ln.startswith("I,"):
                    self.info.append(ln)

    def record(self, secs, path, live=True):
        self.buf = []
        t0 = time.time()
        while time.time() - t0 < secs:
            time.sleep(0.5)
            if live:
                n = len(self.buf)
                sys.stdout.write("\r  받는 중 %4.1f초  %5d개 (%3.0f/초)  " % (time.time() - t0, n, n / max(time.time() - t0, 0.1)))
                sys.stdout.flush()
        lines, self.buf = self.buf, None
        if live:
            print()
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)


def set_channel(tx, rx, ch):
    send(tx, "ch %d" % ch, 0.2)
    send(rx, "ch %d" % ch, 2.0)                               # 받는 쪽은 채널을 바꾸면 이득 기준을 새로 잡는다 (패킷 100개, 1초쯤)


# ---------------------------------------------------------------- 읽기
def load(path):
    """C 줄 -> dict(seq, rssi, agc, fft, comp, ch, P[dB], H[패킷, 서브캐리어] 복소)"""
    seq, rssi, agc, fft, comp, ch, rows = [], [], [], [], [], [], []
    L = None
    for ln in open(path):
        p = ln.strip().split(",")
        if len(p) < 12 or p[0] != "C":
            continue
        n = int(p[9]); hx = p[11]
        if len(hx) != 2 * n or (L is not None and n != L):
            continue
        L = n
        b = np.frombuffer(bytes.fromhex(hx), dtype=np.int8).astype(float)
        if int(p[10]):
            b[:4] = 0.0                                       # first_word_invalid
        rows.append(b[1::2] + 1j * b[0::2])                   # (허수, 실수) 차례
        seq.append(int(p[1])); rssi.append(int(p[2])); agc.append(int(p[4])); fft.append(int(p[5]))
        comp.append(int(p[6]) / 1000.0); ch.append(int(p[7]))
    if not rows:
        return None
    H = np.array(rows)
    used = np.abs(H).mean(0) > 0.5                            # 늘 0 인 서브캐리어(가운데, 가장자리)는 뺀다
    H = H[:, used]
    comp = np.array(comp); comp[comp <= 0] = 1.0
    P = 10 * np.log10(np.mean(np.abs(H) ** 2, 1) + 1e-9) + 20 * np.log10(comp)
    return dict(seq=np.array(seq), rssi=np.array(rssi), agc=np.array(agc), fft=np.array(fft), comp=comp, ch=np.array(ch), H=H, P=P)


def level(path):
    """한 기록의 대표 세기 (dB, 이득을 되돌린 CSI 세기의 가운데값)와 RSSI 가운데값"""
    d = load(path)
    if d is None or len(d["P"]) < 10:
        return None
    return float(np.median(d["P"])), float(np.median(d["rssi"])), len(d["P"])


# ---------------------------------------------------------------- 시험 1
def t1_eval(path, secs=None):
    d = load(path)
    if d is None or len(d["P"]) < 50:
        return dict(file=os.path.basename(path), ok=False, why="패킷이 거의 없다")
    P = d["P"]; step = np.abs(np.diff(P))
    w = np.lib.stride_tricks.sliding_window_view(P, 4) if hasattr(np.lib.stride_tricks, "sliding_window_view") else np.array([P[i:i + 4] for i in range(len(P) - 3)])
    spread = w.max(1) - w.min(1)
    gch = int(np.sum((np.diff(d["agc"]) != 0) | (np.diff(d["fft"]) != 0)))
    rate = len(P) / secs if secs else float("nan")
    lost = float(1 - len(P) / max(d["seq"].max() - d["seq"].min() + 1, 1))
    r = dict(file=os.path.basename(path), packets=len(P), rate=round(rate, 1), lost=round(lost, 3),
             step_med_db=round(float(np.median(step)), 3), jump_over_1db=round(float(np.mean(step > 1.0)), 3),
             spread4_p90_db=round(float(np.percentile(spread, 90)), 2), gain_changes=gch,
             rssi_med=float(np.median(d["rssi"])))
    r["ok"] = bool(r["step_med_db"] <= T1_STEP_DB and r["spread4_p90_db"] <= T1_SPREAD_DB and (secs is None or rate >= T1_RATE))
    return r


def cmd_t1(txp, rxp, secs=60):
    os.makedirs(DATA, exist_ok=True)
    tx, rx = open_port(txp), open_port(rxp); rd = Reader(rx)
    print("시험 1. 두 보드를 3 m 떨어뜨려 마주 보게 두고, 둘 사이로 사람이 지나가지 않게 한다.")
    input("준비되면 Enter ")
    out = []
    for gain in ("auto", "lock"):
        send(rx, "gain %s" % gain, 0.5)
        for name, ch in BANDS:
            set_channel(tx, rx, ch)
            path = os.path.join(DATA, "t1_%s_ch%d.csv" % (gain, ch))
            print("- %s GHz (%d번), 이득 %s, %d초" % (name, ch, gain, secs))
            rd.record(secs, path)
            r = t1_eval(path, secs); r.update(band=name, gain=gain); out.append(r)
            print("  이어진 패킷 차 %.2f dB, 4개 폭(90%%) %.2f dB, 1 dB 넘게 뜀 %.0f%%, 이득 바뀜 %d번, %.0f개/초 -> %s"
                  % (r.get("step_med_db", -1), r.get("spread4_p90_db", -1), 100 * r.get("jump_over_1db", 0), r.get("gain_changes", 0),
                     r.get("rate", 0), "통과" if r["ok"] else "다시 볼 것"))
    send(rx, "gain auto", 0.5)
    rd.stop = True
    save_summary("t1", out)
    verdict_t1(out)


def verdict_t1(rows):
    print("\n== 시험 1 판정 ==")
    for name, ch in BANDS:
        rr = [r for r in rows if r.get("band") == name]
        ok = any(r["ok"] for r in rr)
        best = min(rr, key=lambda r: r.get("spread4_p90_db", 99)) if rr else None
        print("%s GHz: %s%s" % (name, "통과" if ok else "실패",
                                "" if not best else " (가장 좋은 것: 이득 %s, 4개 폭 %.2f dB)" % (best.get("gain"), best.get("spread4_p90_db", -1))))
    print("실패하면: 패킷을 더 모아 평균(재기가 느려짐) -> 그래도 안 되면 받는 쪽을 AX210 으로 (docs/빠른 시험.md 5절)")


# ---------------------------------------------------------------- 시험 2 + 3
ANGLES = list(range(0, 360, 15))


def rec_bands(tx, rx, rd, tag, secs):
    for name, ch in BANDS:
        set_channel(tx, rx, ch)
        path = os.path.join(DATA, "%s_ch%d.csv" % (tag, ch))
        n = rd.record(secs, path, live=False)
        lv = level(path)
        print("   %s GHz: %s" % (name, "패킷 없음" if lv is None else "세기 %.1f dB (RSSI %d), %d개" % (lv[0], lv[1], n)))


def rotate(tx, rx, rd, tag, secs=3):
    print("  각도기 위에서 판을 돌린다. 0도 = 안테나 면이 보내는 쪽을 정면으로 봄. 15도씩 24번.")
    for a in ANGLES:
        input("  %3d도에 맞추고 Enter " % a)
        rec_bands(tx, rx, rd, "%s_a%03d" % (tag, a), secs)


def cmd_t2(txp, rxp, secs=10):
    os.makedirs(DATA, exist_ok=True)
    tx, rx = open_port(txp), open_port(rxp); rd = Reader(rx)
    send(rx, "gain auto", 0.5)
    print("시험 2 + 3. 보내는 보드는 2 m 앞, 받는 판과 같은 높이에 둔다. 방 가운데, 주변 1 m 안에 금속이 없게.")
    print("안테나 번호: 도안(도안.pdf 1쪽)의 1~9. 받는 보드(XIAO)의 U.FL 은 처음부터 끝까지 가운데 1번에만 꽂는다 (빼고 꽂기 1번).")
    input("\n[1] 우드락에 1번 안테나만 붙이고, 받는 보드를 1번에 꽂는다. 판을 0도로. Enter ")
    rec_bands(tx, rx, rd, "t2_alone", secs)
    if yes("  혼자일 때 돌린 무늬도 잴까? (권함, 약 5분) [Y/n] "):
        rotate(tx, rx, rd, "t3_alone")
    input("\n[2] 2~9번을 도안 자리에 붙인다 (선은 뒤로 늘어뜨리고 끝은 아무 데도 꽂지 않는다). 판을 0도로. Enter ")
    rec_bands(tx, rx, rd, "t2_arr", secs)
    if yes("  넷일 때 돌린 무늬를 잴까? (권함, 약 5분) [Y/n] "):
        rotate(tx, rx, rd, "t3_arr")
    for mm in (15, 20, 31):
        input("\n[3] 반사판(호일 판)을 안테나 면 뒤 %d mm 에 댄다 (도안 1쪽의 간격 기둥). 판을 0도로. Enter " % mm)
        rec_bands(tx, rx, rd, "t2_r%d" % mm, secs)
    best = best_reflector()
    print("\n  가장 나은 반사판 거리: %s mm" % best)
    if yes("  그 거리로 돌린 무늬(앞뒤 차)를 잴까? (권함, 약 5분) [Y/n] "):
        input("  반사판을 %s mm 에 두고 Enter " % best)
        rotate(tx, rx, rd, "t3_r%s" % best)
    rd.stop = True
    report()


def yes(q):
    a = input(q).strip().lower()
    return a in ("", "y", "yes", "ㅇ", "네", "응")


def lv(tag, ch):
    p = os.path.join(DATA, "%s_ch%d.csv" % (tag, ch))
    return level(p) if os.path.exists(p) else None


def best_reflector():
    """세 대역 가운데 가장 나쁜 대역의 (반사판 - 반사판 없음) 이 가장 큰 거리"""
    best, bv = None, -1e9
    for mm in (15, 20, 31):
        g = []
        for _, ch in BANDS:
            a, b = lv("t2_r%d" % mm, ch), lv("t2_arr", ch)
            if a and b:
                g.append(a[0] - b[0])
        if len(g) == len(BANDS) and min(g) > bv:
            best, bv = mm, min(g)
    return best


def pattern(tag, ch):
    out = []
    for a in ANGLES:
        x = lv("%s_a%03d" % (tag, a), ch)
        out.append(np.nan if x is None else x[0])
    return np.array(out)


def report():
    rows = {"t1": [], "t2": [], "t3": []}
    for f in sorted(glob.glob(os.path.join(DATA, "t1_*_ch*.csv"))):
        r = t1_eval(f)
        g, ch = os.path.basename(f)[3:-4].split("_ch")
        r.update(gain=g, band=[n for n, c in BANDS if c == int(ch)][0]); rows["t1"].append(r)
    if rows["t1"]:
        verdict_t1(rows["t1"])
    if lv("t2_alone", BANDS[0][1]):
        print("\n== 시험 2 판정 (정면 세기, dB) ==")
        print("대역    혼자   넷   변화   반사판 15 / 20 / 31 mm (넷 대비)")
        for name, ch in BANDS:
            a, n = lv("t2_alone", ch), lv("t2_arr", ch)
            if not (a and n):
                continue
            ch_db = n[0] - a[0]
            refl = [lv("t2_r%d" % mm, ch) for mm in (15, 20, 31)]
            rtxt = " / ".join("%+5.1f" % (r[0] - n[0]) if r else "  -  " for r in refl)
            state = "보정 없이 됨" if abs(ch_db) <= T2_FRONT_OK else ("보정 필요" if ch_db > -T2_FRONT_BAD else "안테나가 크게 어긋남")
            print("%s GHz %6.1f %6.1f %+6.1f   %s   -> %s" % (name, a[0], n[0], ch_db, rtxt, state))
            rows["t2"].append(dict(band=name, alone=a[0], arr=n[0], change=ch_db, refl={mm: (r[0] - n[0]) if r else None for mm, r in zip((15, 20, 31), refl)}, state=state))
        b = best_reflector()
        if b:
            print("반사판: %d mm 가 세 대역 모두에서 가장 낫다 (가장 나쁜 대역 기준)." % b)
    tags = sorted(set(os.path.basename(f).rsplit("_a", 1)[0] for f in glob.glob(os.path.join(DATA, "t3_*_a*_ch*.csv"))))
    if tags:
        print("\n== 시험 3 판정 (돌린 무늬) ==")
        for name, ch in BANDS:
            pa = pattern("t3_alone", ch) if "t3_alone" in tags else None
            pn = pattern("t3_arr", ch) if "t3_arr" in tags else None
            r = dict(band=name)
            if pa is not None and pn is not None:
                front = np.array([a <= 80 or a >= 280 for a in ANGLES])
                d = (pn - pa)[front]; d = d[~np.isnan(d)]
                if len(d):
                    r["ripple_db"] = float(d.max() - d.min())
            ref = [t for t in tags if t.startswith("t3_r")]
            if ref:
                pr = pattern(ref[0], ch); f = pr[ANGLES.index(0)]
                back = np.nanmean([pr[ANGLES.index(a)] for a in (150, 165, 180, 195, 210)])
                r["front_back_db"] = float(f - back)
            rows["t3"].append(r)
            txt = []
            if "ripple_db" in r:
                txt.append("무늬 출렁임 %.1f dB (%s)" % (r["ripple_db"], "보정 없이 됨" if r["ripple_db"] <= T3_RIPPLE_OK else "보정 필요"))
            if "front_back_db" in r:
                txt.append("앞뒤 차 %.1f dB (%s)" % (r["front_back_db"], "됨" if r["front_back_db"] >= T3_FB_OK else "반사판 다시 볼 것"))
            print("%s GHz: %s" % (name, ", ".join(txt) if txt else "기록 없음"))
        plot(tags)
    save_summary("report", rows)


def plot(tags):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axs = plt.subplots(1, len(BANDS), figsize=(5 * len(BANDS), 4.5), subplot_kw=dict(projection="polar"))
    for ax, (name, ch) in zip(axs, BANDS):
        for t in tags:
            p = pattern(t, ch)
            if np.all(np.isnan(p)):
                continue
            ax.plot(np.radians(ANGLES + [360]), np.append(p, p[0]) - np.nanmax(p), label=t[3:])
        ax.set_title("%s GHz" % name); ax.set_rlim(-30, 3); ax.set_theta_zero_location("N")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); out = os.path.join(DATA, "무늬.png"); fig.savefig(out, dpi=120)
    print("그림: %s" % out)


def save_summary(name, rows):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "판정_%s.json" % name), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=float)


# ---------------------------------------------------------------- 그 밖
def cmd_ports():
    from serial.tools import list_ports
    for p in list_ports.comports():
        print(p.device, "-", p.description, p.hwid)


def cmd_setup(txp, rxp):
    """역할을 바꾸면 보드가 다시 켜지며 USB 가 잠깐 끊긴다. 닫았다가 다시 연다"""
    for port, role in ((txp, "tx"), (rxp, "rx")):
        s = open_port(port); send(s, "role %s" % role, 0.5); s.close()
    time.sleep(4.0)
    for port, nm in ((txp, "보내는 쪽"), (rxp, "받는 쪽")):
        s = open_port(port); send(s, "status", 0.8)
        got = [l for l in s.read(65536).decode(errors="ignore").splitlines() if l.startswith("I,status")]
        print(nm, port, got[-1] if got else "(답 없음: 포트를 바꿔 꽂았거나 펌웨어가 안 올라갔다)"); s.close()


def main(a):
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); return
    c = a[0]
    if c == "ports":
        cmd_ports()
    elif c == "setup":
        cmd_setup(a[1], a[2])
    elif c == "t1":
        cmd_t1(a[1], a[2], int(a[3]) if len(a) > 3 else 60)
    elif c == "t2":
        cmd_t2(a[1], a[2], int(a[3]) if len(a) > 3 else 10)
    elif c == "report":
        report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
