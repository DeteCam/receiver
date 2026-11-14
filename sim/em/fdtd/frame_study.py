"""플라스틱 판 틀이 전파를 방해하는지: gprMax(FDTD)로 '판만' 과 '판 + 틀(PLA)' 을 견준다.
    python3 frame_study.py make      ->  free.in, plate.in, frame.in (격자 2.5mm)
    python3 frame_study.py run       ->  gprMax 로 셋을 푼다 (하나에 10분쯤)
    python3 frame_study.py report    ->  frame_study.json: 판이 가리는 골의 깊이·폭·가운데가 틀 때문에 얼마나 바뀌나

NEC2 는 금속만 다룰 수 있어서 플라스틱 틀은 시간 영역 격자 풀이(FDTD)인 gprMax 로 본다. 플라스틱(PLA)은 유전율 2.75, 손실각 0.01 로 둔다.
안테나(짧은 세로 막대) 둘레 반지름 0.30m 원 위 72곳에서 받는 세기를 판 없을 때와 견준다. 판을 돌리는 것과 같은 그림이다 (판은 0도 쪽에 있다).
틀은 실제보다 조금 두껍고 넓게 넣었다 (격자 한 칸 2.5mm. 실제 바닥은 2.0, 판 앞으로 나온 테두리·턱·누름쇠는 0.15~2.4): 나쁜 쪽으로 본 것이다."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = 0.0025; X = Y = 0.72; Z = 0.52; C = (0.36, 0.36, 0.24); R = 0.30; F0 = 2.437e9
EPS, TAND = 2.75, 0.01
SIG = 2 * np.pi * F0 * 8.854e-12 * EPS * TAND
s = lambda v: round(v / D) * D
# 판 (머리 좌표의 mm 값을 그대로: 반지름 80, 폭 100, 아래 -58, 위 92). 판 틀 (p2.py 와 같은 치수, 나쁜 쪽으로 둥글림, 2026-11-15 한 장짜리 틀):
#   판 뒤: 바닥 테두리(판 둘레 3mm) + 가로대(z 8~34) + 받침(|y| 28, z 2~40), 두께 2.5
#   판 앞: 홈 둘레 벽(옆 2, 아래 1, 위 띠 8.5), 아래 턱(|y| 46, 판 위 2.5mm), 누름쇠(|y| 30.5, z 89~100), 두께 2.5
PL = dict(x=0.080, y0=-0.050, y1=0.050, z0=-0.058, z1=0.092)
FR = dict(out_y=0.0525, out_z0=-0.0600, out_z1=0.1005, win_y=0.047, win_z0=-0.055, win_z1=0.089, band_z0=0.0075, band_z1=0.0350, pad_y=0.0285, pad_z0=0.0, pad_z1=0.040,
          lip_y=0.046, lip_z1=-0.0555, cl_y=0.0305, cl_z0=0.0875, cl_z1=0.1005, t=0.0025)


def deck(name, plate, frame):
    L = ["#title: DeteCam frame study (%s)" % name, "#domain: %g %g %g" % (X, Y, Z), "#dx_dy_dz: %g %g %g" % (D, D, D), "#time_window: 7e-9",
         "#waveform: gaussiandot 1 %g src" % F0, "#hertzian_dipole: z %g %g %g src" % C, "#material: %g %g 1 0 pla" % (EPS, SIG)]
    cx, cy, cz = C; xp = s(cx + PL["x"])
    if frame:
        t = FR["t"]; F = lambda x0, y0, z0, x1, y1, z1, mat: L.append("#box: %g %g %g %g %g %g %s" % (x0, s(cy + y0), s(cz + z0), x1, s(cy + y1), s(cz + z1), mat))
        F(xp - t, -FR["out_y"], FR["out_z0"], xp, FR["out_y"], FR["out_z1"], "pla")                                    # 판 뒤 바닥
        F(xp - t, -FR["win_y"], FR["win_z0"], xp, FR["win_y"], FR["win_z1"], "free_space")                             # 바닥 창
        F(xp - t, -FR["win_y"], FR["band_z0"], xp, FR["win_y"], FR["band_z1"], "pla")                                  # 가로대
        F(xp - t, -FR["pad_y"], FR["pad_z0"], xp, FR["pad_y"], FR["pad_z1"], "pla")                                    # 받침
        F(xp, -FR["out_y"], FR["out_z0"], xp + t, FR["out_y"], FR["out_z1"], "pla")                                    # 판 앞: 홈 둘레 벽과 위 띠
        F(xp, PL["y0"], PL["z0"], xp + t, PL["y1"], PL["z1"], "free_space")                                           # 홈 (판 앞면 위는 비어 있다)
        F(xp, -FR["lip_y"], PL["z0"], xp + t, FR["lip_y"], FR["lip_z1"], "pla")                                        # 아래 턱
        F(xp, -FR["cl_y"], FR["cl_z0"], xp + t, FR["cl_y"], FR["cl_z1"], "pla")                                        # 누름쇠
    if plate:
        L.append("#plate: %g %g %g %g %g %g pec" % (xp, s(cy + PL["y0"]), s(cz + PL["z0"]), xp, s(cy + PL["y1"]), s(cz + PL["z1"])))
    for k in range(72):
        a = np.radians(5.0 * k); L.append("#rx: %g %g %g" % (s(cx + R * np.cos(a)), s(cy + R * np.sin(a)), cz))
    return "\n".join(L) + "\n"


RUNS = (("free", False, False), ("plate", True, False), ("frame", True, True))


def amp(path):
    import h5py
    f = h5py.File(path, "r"); dt = float(f.attrs["dt"]); n = int(f.attrs["Iterations"]); t = np.arange(n) * dt; w = np.exp(-2j * np.pi * F0 * t)
    keys = sorted(f["rxs"].keys(), key=lambda k: int(k[2:]))
    return np.array([abs(np.sum(f["rxs"][k]["Ez"][:] * w)) for k in keys])


def shadow(db):
    """곡선(dB, 0도 = 판 쪽)에서 골의 깊이, -3dB 폭, 가운데(-1dB 아래 구간의 무게중심)"""
    ang = np.arange(72) * 5.0; a = (ang + 180) % 360 - 180; o = np.argsort(a); a, d = a[o], db[o]
    depth = float(d.min()); half = d <= depth + 3.0; width = float(half.sum() * 5.0)
    w = np.clip(-1.0 - d, 0, None); centre = float(np.sum(w * a) / np.sum(w)) if w.sum() > 0 else 0.0
    return dict(depth_db=round(depth, 2), width_3db_deg=width, centre_deg=round(centre, 2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    os.chdir(HERE)
    if cmd == "make":
        for nm, p, fr in RUNS:
            open("fs_%s.in" % nm, "w").write(deck(nm, p, fr))
        print("ok:", [("fs_%s.in" % r[0]) for r in RUNS])
    elif cmd == "run":
        for nm, p, fr in RUNS:
            if not os.path.exists("fs_%s.h5" % nm):
                subprocess.run([sys.executable, "-m", "gprMax", "fs_%s.in" % nm], check=True, stdout=open("fs_%s.log" % nm, "w"), stderr=subprocess.STDOUT)
                print("done", nm, flush=True)
    else:
        a0 = amp("fs_free.h5"); ap = amp("fs_plate.h5"); af = amp("fs_frame.h5")
        dp = 20 * np.log10(ap / a0); df = 20 * np.log10(af / a0)
        out = dict(plate=shadow(dp), frame=shadow(df), max_diff_db=round(float(np.max(np.abs(df - dp))), 2),
                   rms_diff_db=round(float(np.sqrt(np.mean((df - dp) ** 2))), 2), curve_plate=np.round(dp, 2).tolist(), curve_frame=np.round(df, 2).tolist(),
                   eps=EPS, tand=TAND, grid_mm=D * 1e3)
        json.dump(out, open("frame_study.json", "w"), ensure_ascii=False, indent=1)
        print("판만:", out["plate"]); print("판+틀:", out["frame"]); print("두 곡선의 차이: 최대 %.2f dB, 평균 %.2f dB" % (out["max_diff_db"], out["rms_diff_db"]))
