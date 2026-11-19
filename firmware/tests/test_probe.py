# -*- coding: utf-8 -*-
"""bench/probe.py 의 읽기와 판정을 가짜 기록으로 검사한다 (장치 없이).  python3 tests/test_probe.py"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import probe  # noqa: E402


def fake_line(rng, seq, amp, ch, jitter_db=0.05, agc=30, fft=0, comp=1.0):
    n = 57
    g = 10 ** (rng.normal(0, jitter_db) / 20)
    H = amp * g * np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(1j * np.linspace(0, 3, n))
    H[28] = 0
    b = np.empty(2 * n, np.int8); b[0::2] = np.clip(np.round(H.imag), -127, 127); b[1::2] = np.clip(np.round(H.real), -127, 127)
    return "C,%d,-40,-95,%d,%d,%d,%d,1,%d,0,%s" % (seq, agc, fft, int(comp * 1000), ch, len(b), b.tobytes().hex())


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(); probe.DATA = tmp
    try:
        # 시험 1: 고른 기록은 통과, 흔들리는 기록은 실패
        write(os.path.join(tmp, "t1_auto_ch6.csv"), [fake_line(rng, i, 40, 6, 0.05) for i in range(3000)])
        write(os.path.join(tmp, "t1_auto_ch40.csv"), [fake_line(rng, i, 40, 40, 2.5) for i in range(3000)])
        a = probe.t1_eval(os.path.join(tmp, "t1_auto_ch6.csv"), 30)
        b = probe.t1_eval(os.path.join(tmp, "t1_auto_ch40.csv"), 30)
        assert a["ok"] and not b["ok"], (a, b)
        assert a["packets"] == 3000 and abs(a["lost"]) < 1e-9
        # 이득 보정: comp 가 2 배면 세기가 6 dB 오른다
        write(os.path.join(tmp, "c1.csv"), [fake_line(rng, i, 40, 6, 0.0, comp=1.0) for i in range(50)])
        write(os.path.join(tmp, "c2.csv"), [fake_line(rng, i, 40, 6, 0.0, comp=2.0) for i in range(50)])
        d = probe.level(os.path.join(tmp, "c2.csv"))[0] - probe.level(os.path.join(tmp, "c1.csv"))[0]
        assert abs(d - 6.02) < 0.1, d
        # 시험 2: 반사판 20 mm 가 가장 나은 경우
        gains = {"t2_alone": 0.0, "t2_arr": -2.0, "t2_r15": -1.0, "t2_r20": 2.0, "t2_r31": 1.0}
        for tag, gdb in gains.items():
            for _, ch in probe.BANDS:
                amp = 30 * 10 ** (gdb / 20) * (1.4 if (tag == "t2_r31" and ch == 6) else 1.0)
                write(os.path.join(tmp, "%s_ch%d.csv" % (tag, ch)), [fake_line(rng, i, amp, ch) for i in range(100)])
        assert probe.best_reflector() == 20, probe.best_reflector()
        # 시험 3: 앞뒤 차
        for a_ in probe.ANGLES:
            for _, ch in probe.BANDS:
                r = math_cos(a_)
                for tag, k in (("t3_alone", 1.0), ("t3_arr", 1.0), ("t3_r20", 1.0)):
                    amp = 40 * max(r, 0.05) if tag == "t3_r20" else 40 * (0.5 + 0.5 * max(r, 0.2))
                    write(os.path.join(tmp, "%s_a%03d_ch%d.csv" % (tag, a_, ch)), [fake_line(rng, i, amp, ch) for i in range(30)])
        probe.report()
        import json
        rep = json.load(open(os.path.join(tmp, "판정_report.json"), encoding="utf-8"))
        t3 = rep["t3"][0]
        assert t3["ripple_db"] < 0.5 and t3["front_back_db"] > 20, t3
        assert rep["t2"][0]["state"] == "보정 없이 됨"
    finally:
        shutil.rmtree(tmp)
    print("test_probe ok")


def math_cos(a):
    return float(np.cos(np.radians(a)))


if __name__ == "__main__":
    main()
