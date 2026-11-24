"""Markdown rows of the sensitivity table (classroom 2.4 GHz, front spots).  Usage: sens_rows.py"""
import os, numpy as np
ROWS = (("s_base", "기준(오차 없음, 자리마다 위상 오차 4.5°, 잡음 −30 dB)"), ("s_dim10", "방 크기를 10 cm 틀리게 앎"), ("s_dim25", "방 크기를 25 cm 틀리게 앎"),
        ("s_pos10", "수신기 위치를 10 cm 틀리게 앎"), ("s_pos25", "수신기 위치를 25 cm 틀리게 앎"), ("s_head3", "수신기가 놓인 각도가 3° 틀어짐"),
        ("s_head6", "놓인 각도가 6° 틀어짐 *"), ("s_head6cal", "놓인 각도 6° + 현장 보정 *"), ("s_all", "각도 6° · 위치 20 cm · 방 크기 20 cm 모두 *"), ("s_allcal", "위 세 가지 + 현장 보정 *"),
        ("s_jit8", "팔 각도가 자리마다 8° 틀림"), ("s_cal30", "자리마다 위상이 30° 흔들림(안테나선 휨)"), ("s_n20", "잡음 −20 dB"),
        ("s_tv50", "재는 동안 반사의 절반이 흔들림"), ("s_tv50x4", "반사의 절반이 흔들림 + 네 바퀴 평균"), ("s_tv100", "재는 동안 반사가 전부 흔들림"), ("s_tv100x4", "반사가 전부 흔들림 + 네 바퀴 평균"))
q = lambda e: f"{np.mean(e):.1f}° / {np.median(e):.1f}° / {np.percentile(e, 90):.1f}°"
for tag, label in ROWS:
    f = f"res_pe2_{tag}.npy"
    if not os.path.exists(f): print(f"| {label} | (not run) | |"); continue
    d = np.load(f, allow_pickle=True).item(); e = np.array([x["err"] for x in d["rows"] if not x["beacon"] and x["n"] in (0, 1)])
    pe = np.array([x["err"] for x in d["pos"] if x["sub"] == "x2 adj" and not x["beacon"]])
    print(f"| {label} | {q(e)} | {100 * np.mean(pe <= 1):.0f}% |")
