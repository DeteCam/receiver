"""ESP32 를 머리에 넣으면 방향 찾기가 얼마나 나빠지나 (2026-11-16).   python3 esp32_parts.py   결과: esp32_parts.json

ESP32 부품(FPC 안테나는 파이 안테나 바로 위 머리 천장, 동축선, XIAO 기판, 점퍼선 4가닥)을 NEC 모형에 넣고 (device.add_esp, run_tables.py esp_pi / esp_fpc)
  1) 지금 설계 (ESP32 없음), 파이 안테나로 받기                        : 기준
  2) ESP32 를 넣었는데 무늬 표는 예전 것 (ESP32 없는 표)                : 표를 다시 만들지 않으면 얼마나 틀리나
  3) ESP32 를 넣고 표도 다시 만듦, 파이 안테나로 받기                   : 파이 쪽에 남는 손해
  4) ESP32 의 FPC 안테나로 받기 (ESP32 를 넣은 표, FPC 를 먹여 만든 것)  : ESP32 가 혼자 방향을 찾을 수 있나
  5) 표는 ESP32 가 켜진 채로 만들었는데 잴 때 ESP32 가 꺼져 있으면 (FPC 단자 50옴 -> 거의 열림)   : ESP32 상태를 늘 같게 해야 하나
를 own_parts.py 와 같은 방법(받침 각 3개, 반사 없음 / 보통 방, 카메라 120곳)으로 돌린다. 골이 얼마나 뚜렷한지(metrics.py)도 함께 본다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, json, time
import numpy as np
import device as V
import study as S, tmatch as T, metrics as MT

YAWS = (-120.0, 0.0, 120.0)
N = 120


def resp(prefix):
    return {y: S.Resp("%s_y%+04d" % (prefix, y)) for y in YAWS}


def tmpl(rs):
    return {y: T.Templates(r) for y, r in rs.items()}


def impedance(name):
    d = np.load(os.path.join(V.CACHE, name + "_0.npz")); return complex(d["z"])


def main():
    dev, epi, efp = resp("dev"), resp("esp_pi"), resp("esp_fpc")
    t_dev, t_epi, t_efp = tmpl(dev), tmpl(epi), tmpl(efp)
    cases = [("1 지금 설계 (ESP32 없음) · 파이", dev, t_dev), ("2 ESP32 넣음 · 파이 · 예전 표", epi, t_dev),
             ("3 ESP32 넣음 · 파이 · 표 다시", epi, t_epi), ("4 ESP32 넣음 · FPC 로 받기", efp, t_efp)]
    if all(os.path.exists(os.path.join(V.CACHE, "esp_open_y%+04d.npz" % y)) for y in YAWS):          # 5) 기준 곡선은 ESP32 가 켜진(듣는) 채로 쟀는데, 잴 때 ESP32 가 꺼져 있으면 (단자가 거의 열림)
        cases.append(("5 ESP32 꺼짐 · 파이 · 켜짐 표", resp("esp_open"), t_epi))
    out = dict(n=N, cases={}, metrics={}, impedance={})
    for lab, world, tm in cases:
        for mp, env in ((False, "반사 없음"), (True, "보통 방")):
            t0 = time.time(); s = T.run(world, tm, n=N, multipath=mp)
            out["cases"]["%s|%s" % (lab, env)] = {k: v for k, v in s.items() if k not in ("err", "margin")}
            print(T.line("%s | %s" % (lab, env), s), " (%.0f초)" % (time.time() - t0), flush=True)
    for nm in ("dev_y+000", "esp_pi_y+000", "esp_fpc_y+000"):
        out["metrics"][nm] = MT.table_metrics(nm); print(MT.line(nm, out["metrics"][nm]), flush=True)
    for nm in ("dev_y+000", "esp_pi_y+000", "esp_fpc_y+000"):
        z = impedance(nm); g = abs((z - 50) / (z + 50)); out["impedance"][nm] = dict(R=round(z.real, 1), X=round(z.imag, 1), mismatch_db=round(float(-10 * np.log10(1 - g * g)), 1))
        print(nm, out["impedance"][nm])
    json.dump(out, open(os.path.join(empaths.STUDIES, "esp32_parts.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
