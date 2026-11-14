"""실물로 잰 곡선을 모형과 견준다: 어느 전파 모형이 실제와 닮았는지, 그 모형으로 계산하면 방향이 얼마나 맞는지.

    python3 compare_real.py 내려받은파일.json --az 0 --el 0 [--pos 1]

쓰는 법 (첫 실물 실험)
  1. 넓은 곳에서, 전파를 계속 내는 기기(카메라나 폰 핫스팟)를 수신기에서 2~3m 떨어진 '아는 방향'에 둔다.
     방향은 수신기에서 본 각: --az = 받침 앞 표시(0도)에서 왼쪽으로 돈 각, --el = 안테나 높이에서 올려다본 각.
  2. 폰 화면에서 평소처럼 한 자리 잰다. (자세를 바꿔 여러 번 재면 더 좋다: 기기를 옮기는 대신 수신기를 돌려 놓아도 된다)
  3. 폰 화면의 '결과 내려받기'로 받은 JSON 을 이 프로그램에 준다.
보여 주는 것: 모형마다 ① 아는 방향에서 모형 곡선이 잰 곡선과 얼마나 다른가 (dB), ② 그 모형으로 찾은 방향이 아는 방향과 몇 도 다른가.
  - ①이 작고 ②가 10도 안쪽인 모형이 있으면: 그 모형의 무늬로 계산하면 된다.
  - 어느 모형도 안 맞으면: 실물 곡선으로 무늬를 만드는 보정(수신기를 돌려 가며 아는 기기를 재는 것)이 필요하다.
모형: 전파 표(NEC2 로 푼 것, tables/ 폴더)들. tmatch.py 의 '곡선 전체 맞추기'로 견준다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json, argparse
import numpy as np
HERE = empaths.HERE
import study as S, tmatch as T, tables as TB
from estimator import unit, azel

MODELS = [("이상적인 안테나 (전기)", "ideal_z"), ("이상적인 안테나 (자기)", "geo_mz_w100_R80"), ("기판 + 틈 안테나", "geo_slot_w100_R80"),
          ("기판 + 막대 안테나 + 머리 부품", "head_mono"), ("기판 + 홈 안테나 + 머리 부품", "head"), ("지금 설계 전체 (받침 각마다)", ("dev_y+000", "dev_y+120", "dev_y-120"))]


def have(name):
    return os.path.exists(os.path.join(TB.DIR, name + ".npz")) or os.path.exists(os.path.join(TB.CACHE, name + ".npz"))


def templates(spec, phi):
    if isinstance(spec, tuple):
        if not all(have(n) for n in spec):
            return None
        return {float(n.split("_y")[1]): T.Templates(S.Resp(n), phi) for n in spec}
    return T.Templates(S.Resp(spec), phi) if have(spec) else None


def compare(curves, truth, quiet=False):
    """curves: [dict(yaw, pitch, lv, bin_deg)], truth: 단위 벡터(받침 좌표). 돌아오는 값: [(모형 이름, 남는 차이 dB, 찾은 방향 오차 도)]"""
    plan = [(c["yaw"], c["pitch"]) for c in curves]; out = []
    cv = []
    for c in curves:
        lv = np.array([np.nan if v is None else v for v in c["lv"]], float); phi = (np.arange(len(lv)) + 0.5) * c["bin_deg"]; cv.append((phi, lv))
    phi = cv[0][0]
    for label, spec in MODELS:
        tm = templates(spec, phi)
        if tm is None:
            continue
        def Tm(y):
            if isinstance(tm, dict):
                ks = np.array(list(tm.keys())); return tm[float(ks[int(np.argmin(np.abs((ks - y + 180.0) % 360.0 - 180.0)))])]
            return tm
        J = [float(Tm(y).cost_at(T.to_head(truth[None, :], y, t), lv)[0]) for (ph, lv), (y, t) in zip(cv, plan)]
        resid_db = float(np.sqrt(np.mean(J)) * 10.0 / np.log(10.0))               # 상대 오차 -> dB 어림
        est, info = T.solve(cv, plan, tm)
        err = float(np.degrees(np.arccos(np.clip(est @ truth, -1, 1)))); az, el = azel(est)
        out.append(dict(model=label, resid_db=round(resid_db, 2), err_deg=round(err, 1), az=round(float(az), 1), el=round(float(el), 1), margin=round(info["margin"], 1)))
    out.sort(key=lambda r: r["resid_db"])
    if not quiet:
        print("%-34s %14s %18s %16s" % ("모형", "곡선 차이 (dB)", "찾은 방향 (방위, 높이)", "아는 방향과의 차이"))
        for r in out:
            print("%-34s %14.2f %10.1f, %5.1f %15.1f도" % (r["model"], r["resid_db"], r["az"], r["el"], r["err_deg"]))
        good = [r for r in out if r["err_deg"] <= 10.0]
        if good and good[0] is out[0]:
            print("-> '%s' 모형이 곡선도 가장 닮았고 방향도 맞는다 (차이 %.1f도): 이 모형의 무늬로 계산하면 된다." % (out[0]["model"], out[0]["err_deg"]))
        elif good:
            print("-> 방향이 10도 안으로 맞는 모형은 있다 (%s). 다만 곡선이 가장 닮은 모형은 아니다: 자세를 바꿔 몇 번 더 재서 확인한다." % ", ".join(r["model"] for r in good))
        else:
            print("-> 어느 모형도 방향을 10도 안으로 맞추지 못했다: 실물로 무늬를 만드는 보정이 필요하다.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--az", type=float, required=True); ap.add_argument("--el", type=float, default=0.0); ap.add_argument("--pos", type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(a.file, encoding="utf-8")); P = d["positions"]
    for i, p in enumerate(P):
        if a.pos and i + 1 != a.pos:
            continue
        if not p.get("curves"):
            print("자리 %d: 곡선이 없습니다 (수신기 코드가 2026-11-14 판보다 옛것이면 곡선을 내보내지 않습니다. 코드를 새로 넣고 다시 잽니다)." % (i + 1)); continue
        cv = [c for c in p["curves"] if sum(v is not None for v in c["lv"]) >= len(c["lv"]) // 2]
        print("== 자리 %d (%s, 스캔 %d개). 수신기가 계산한 방향: 방위 %s, 높이 %s" % (i + 1, p.get("time"), len(cv), p.get("az"), p.get("el")))
        if p.get("az") is not None:
            e0 = unit(p["az"], p["el"]); print("   수신기의 지금 계산(골 찾기)과 아는 방향의 차이: %.1f도" % np.degrees(np.arccos(np.clip(e0 @ unit(a.az, a.el), -1, 1))))
        compare(cv, unit(a.az, a.el))
