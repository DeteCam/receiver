"""gprMax(FDTD) 입력 파일을 만든다: z 방향 짧은 다이폴 + 금속판 100 x 150 (반지름 80). 수신점은 반지름 0.30m 원 위 72개.
    python3 make_in.py  ->  plate.in (판 있음), free.in (판 없음)
NEC2(모멘트법)와 전혀 다른 풀이법(시간 영역 격자)이라, 둘이 같은 답을 내면 두 도구를 다 믿을 수 있다.
"""
import numpy as np
d = 0.004; X = Y = 0.72; Z = 0.52; c = (0.36, 0.36, 0.24); R = 0.30
snap = lambda v: round(v / d) * d
def deck(plate):
    L = ["#title: DeteCam plate shadow (%s)" % ("plate" if plate else "free"), "#domain: %g %g %g" % (X, Y, Z), "#dx_dy_dz: %g %g %g" % (d, d, d), "#time_window: 7e-9",
         "#waveform: gaussiandot 1 2.437e9 src", "#hertzian_dipole: z %g %g %g src" % c]
    if plate:
        x = snap(c[0] + 0.080); L.append("#plate: %g %g %g %g %g %g pec" % (x, snap(c[1] - 0.050), snap(c[2] - 0.058), x, snap(c[1] + 0.050), snap(c[2] + 0.092)))
    for k in range(72):
        a = np.radians(5.0 * k); L.append("#rx: %g %g %g" % (snap(c[0] + R * np.cos(a)), snap(c[1] + R * np.sin(a)), c[2]))
    return "\n".join(L) + "\n"
open("plate.in", "w").write(deck(True)); open("free.in", "w").write(deck(False)); print(deck(True)[:420])
