"""판 모양을 바꾸면 방향이 더 정확해지나: 판마다 전파 표(NEC2, 안테나 모형 4가지)로 재는 과정을 흉내 내어 두 가지 읽는 법으로 견준다.
    python3 shape_eval.py [안테나 모형 ...]      ->  shape_eval.json (이미 한 것은 건너뛴다)
    python3 shape_eval.py --report              ->  판마다 안테나 모형 4가지의 평균
읽는 법
  골   수신기에 지금 들어 있는 계산 (가장 깊은 골의 가운데). 판 모양이 골을 또렷하게 만드는지가 그대로 드러난다
  맞춤 곡선 전체를 그 판의 기준 곡선과 맞추는 계산 (tmatch.py). 기준 곡선을 실물로 재 두었을 때의 가장 좋은 경우
방: 반사 없음 / 보통 방 / 반사가 심한 방. 자리 하나에서 기울여 세 번 잰다. 카메라 n 곳 (같은 씨앗이라 판끼리 같은 카메라 자리를 쓴다)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json, time
import numpy as np
import device as V, study as S, tmatch as T

OUT = os.path.join(empaths.STUDIES, "shape_eval.json")
N = int(os.environ.get("SHAPE_N", "120"))
PLATES = ["w100_R80", "split10", "split20", "split40", "mesh10", "mesh20", "slotv60", "sloth60", "veein20", "veeout20", "veeout40", "arc100", "flange15", "wings15",
          "w100h200_R80", "w100_R100", "vee150_R80", "w150_R80"]
LABEL = {"w100_R80": "지금 판 (평판 100 x 150)", "split10": "가운데 세로 틈 10mm", "split20": "가운데 세로 틈 20mm", "split40": "가운데 세로 틈 40mm",
         "mesh10": "구멍이 큰 판 (철망, 칸 10mm)", "mesh20": "구멍이 큰 판 (철망, 칸 20mm)", "slotv60": "가운데 세로 슬롯 60mm", "sloth60": "가운데 가로 슬롯 60mm",
         "veein20": "세로로 접기, 양옆이 안테나 쪽으로 20°", "veeout20": "세로로 접기, 양옆이 바깥으로 20°", "veeout40": "세로로 접기, 양옆이 바깥으로 40°",
         "arc100": "곡면 (회전축을 따라 휜 판)", "flange15": "양옆을 바깥으로 15mm 접기", "wings15": "양옆을 안테나 쪽으로 15mm 접기",
         "w100h200_R80": "높이 200 (DiffLoc+ 크기)", "w100_R100": "팔 10cm", "vee150_R80": "폭 150, 바깥으로 30° 접기", "w150_R80": "폭 150"}
ANTS = ["ez", "mz", "slot", "mono"]
ENVS = (("보통", False, "반사없음"), ("보통", True, "보통"), ("나쁨", True, "나쁨"))


def load():
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def evaluate(ants):
    res = load()
    for pl in PLATES:
        for a in ants:
            nm = "geo_%s_%s" % (a, pl)
            if not os.path.exists(os.path.join(V.CACHE, nm + ".npz")):
                continue
            R = None; Tm = None
            for env, mp, lab in ENVS:
                for how in ("골", "맞춤"):
                    key = "%s|%s|%s|%s" % (pl, a, lab, how)
                    if key in res:
                        continue
                    R = R or S.Resp(nm); t0 = time.time()
                    if how == "골":
                        s = S.run(R, n=N, env=env, multipath=mp)
                        res[key] = {k: round(float(v), 3) for k, v in s.items() if k in ("med", "p80", "over15", "over30", "none", "seen_in_cov", "depth_med", "dpsi_med", "dpsi_p90")}
                    else:
                        Tm = Tm or T.Templates(R); s = T.run(R, Tm, n=N, env=env, multipath=mp)
                        res[key] = {k: round(float(v), 3) for k, v in s.items() if k in ("med", "p80", "over15", "over30")}
                    print("%-46s %s  (%.0f초)" % (key, " ".join("%s %.1f" % (k, v) for k, v in res[key].items() if k in ("med", "over30", "depth_med")), time.time() - t0), flush=True)
                    cur = load(); cur.update({k: v for k, v in res.items()}); json.dump(cur, open(OUT, "w"), ensure_ascii=False); res = cur


def report():
    res = load(); rows = {}
    for pl in PLATES:
        row = {}
        for env, mp, lab in ENVS:
            for how in ("골", "맞춤"):
                vals = [res.get("%s|%s|%s|%s" % (pl, a, lab, how)) for a in ANTS]
                vals = [v for v in vals if v]
                if len(vals) == len(ANTS):
                    row["%s|%s" % (lab, how)] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows[pl] = row
    json.dump(rows, open(os.path.join(empaths.STUDIES, "shape_summary.json"), "w"), ensure_ascii=False, indent=1)
    print("%-36s | %-26s | %-26s | %-22s" % ("판", "골 읽기: 보통 방 (가운데값/30도↑)", "골 읽기: 반사 심함", "맞춤: 보통 / 심함"))
    for pl, row in rows.items():
        g = lambda k: row.get(k)
        if not g("보통|골"):
            continue
        print("%-36s | %5.1f° %5.1f%% (골 %4.1fdB) | %5.1f° %5.1f%% | %4.1f° %4.1f%% / %4.1f° %4.1f%%" % (
            LABEL.get(pl, pl), g("보통|골")["med"], g("보통|골")["over30"], g("보통|골")["depth_med"], g("나쁨|골")["med"], g("나쁨|골")["over30"],
            g("보통|맞춤")["med"], g("보통|맞춤")["over30"], g("나쁨|맞춤")["med"], g("나쁨|맞춤")["over30"]))
    return rows


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        evaluate(sys.argv[1:] or ANTS)
