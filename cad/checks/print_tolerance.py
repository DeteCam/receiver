"""출력 오차 시뮬레이션: 3D 프린터가 설계와 조금 다르게 뽑아도 맞는 곳이 맞는지 (몬테카를로, 프린터 상태 세 가지 x 2만 번).
설계 치수(모델 값)는 그대로 두고, 출력물의 치수만 확률로 흔든다.

출력 오차 모델 (가정. FDM/PLA 에서 흔히 보는 크기):
  - 구멍(안쪽 윤곽)은 작게 나온다: 한 번 출력할 때 공통으로 bias, 구멍마다 따로 0.03
  - 바깥 윤곽(핀, 기어 이)은 조금 크게 나온다
  - 구멍 자리(축 사이 거리)는 pos 만큼 흔들린다
  - 큰 홈(판 틀 홈 100.4 x 150.4)은 작은 구멍보다 덜 줄어든다
  프린터 상태: 잘 맞춘 / 보통 / 덜 맞춘(유량 과다, 습한 필라멘트). 값은 아래 PRINTERS.
사는 부품의 흔들림: 모터 축 지름 4.98±0.015, 납작 폭 2.98±0.02, 판 100.0±0.1 x 150.0±0.1, M3 너트 맞변 5.45±0.03.

보는 것 (맞춤을 고르는 곳은 0판 시험 조각으로 고른다고 보고, 고르지 않고 맞춤 2번만 뽑았을 때도 견준다):
  1 모터 축 구멍 (작은 기어, 회전 막대): 손으로 눌러 들어가고(납작 면 겹침 0.08 이하, 지름 겹침 0.10 이하) 헛돌지 않음(납작 면 틈 0.30 이하).
    틈이 0.12 이하인 비율도 따로 센다 (틈이 크면 작은 기어가 축 위에서 조금 놀아 기어 유격에 더해진다. 회전 막대는 고정 나사가 누르므로 상관없다)
  2 끄덕임 축 핀: 빡빡한 구멍에 겹침 0.12 이하·틈 0.03 이하로 끼고, 헐거운 구멍에서는 틈 0.03 이상으로 도는 핀이 있음
  3 판 틀 홈에 판이 들어감 (가로·세로 모두 틈 0 이상)
  4 너트 자리 (맞변 5.8): 너트가 손으로 들어감 (겹침 0.05 이하)
  5 나사가 스스로 나사산을 내는 구멍 (M3 2.6): 2.35~2.80 이면 손으로 돌려 들어가고 단단히 물림
  6 기어 (끄덕임, 받침 회전): 이가 끼어 돌지 않는 일(유격 0.05mm 아래), 출력 오차를 넣은 유격(머리·요크 각도)
결과: 결과 폴더의 print_tolerance.json, 그림 폴더의 36_출력_오차.png. 10초쯤."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
import paths
from params import FITS
from p2 import PIN_FITS, PIN_R_TIGHT, PIN_R_LOOSE, FR_GAP, MOD, ZP

RNG = np.random.default_rng(7)
N = 20000
PRINTERS = {   # 구멍 bias, 그 흔들림(출력마다), 바깥 윤곽 bias, 그 흔들림, 구멍 자리, 큰 홈 bias
    "잘 맞춘 프린터": dict(hole=-0.10, hole_s=0.04, out=0.03, out_s=0.03, pos=0.05, big=-0.06),
    "보통": dict(hole=-0.15, hole_s=0.06, out=0.05, out_s=0.04, pos=0.07, big=-0.10),
    "덜 맞춘 프린터": dict(hole=-0.25, hole_s=0.10, out=0.12, out_s=0.06, pos=0.12, big=-0.20),
}
FEAT = 0.03                     # 같은 출력 안에서 구멍마다 따로 흔들리는 크기
AM = json.load(open(os.path.join(paths.RES, "audit_motion.json"), encoding="utf-8"))
RP = MOD * ZP / 2.0             # 작은 기어 피치 반지름 6.25
GEARS = {"끄덕임": (AM["nod_gear"]["pinion_free_deg"], AM["nod_gear"]["head_play_deg"]),
         "받침 회전": (AM["yaw_gear"]["pinion_free_deg"], AM["yaw_gear"]["yoke_play_deg"])}


def n(mu, s, size=N):
    return RNG.normal(mu, s, size)


def run(p):
    out = {}
    hole_b = n(p["hole"], p["hole_s"]); out_b = n(p["out"], p["out_s"])            # 이번 출력의 공통 치우침
    # 1 모터 축 구멍: 시험 조각의 D자 구멍 3개로 고르고, 같은 출력 조건의 부품(작은 기어)에 쓴다
    sd, sf = n(4.98, 0.015), n(2.98, 0.02)

    def bore(k, b):
        d, f, _ = FITS[k]
        return d + b + n(0, FEAT), f + b + n(0, FEAT)

    def ok_bore(cd, cf, hi=0.30):
        return (cd >= -0.10) & (cf >= -0.08) & (cf <= hi)
    cpn = {k: bore(k, hole_b) for k in FITS}                                          # 시험 조각
    # 고르는 법 (조립 안내 3절): 손으로 무리 없이 들어가는 것(시험 조각에서 겹침 0.05 이하) 가운데 가장 꼭 맞는 것. 다 빡빡하면 3번
    score = np.stack([np.where(((cpn[k][0] - sd) >= -0.05) & ((cpn[k][1] - sf) >= -0.05), cpn[k][1] - sf, 9.0 - k) for k in FITS])
    pick = np.array(list(FITS))[np.argmin(score, 0)]
    part = {k: bore(k, hole_b) for k in FITS}                                         # 실제 부품 (같은 치우침, 구멍마다 다른 흔들림)
    cd = np.choose(pick - 1, [part[k][0] - sd for k in FITS]); cf = np.choose(pick - 1, [part[k][1] - sf for k in FITS])
    out["축 구멍"] = dict(ok=float(ok_bore(cd, cf).mean()), ok_fit2_only=float(ok_bore(part[2][0] - sd, part[2][1] - sf).mean()),
                       snug=float(ok_bore(cd, cf, 0.12).mean()), cf_med=float(np.median(cf)),
                       pick={str(k): float((pick == k).mean()) for k in FITS},
                       spin_deg_med=float(np.degrees(np.median(np.clip(cf, 0, None)) / 2.0)))
    # 2 핀: 빡빡한 구멍(지름 8.0)과 헐거운 구멍(8.3)
    th = 2 * PIN_R_TIGHT + hole_b + n(0, FEAT); lh = 2 * PIN_R_LOOSE + hole_b + n(0, FEAT)
    good_any = np.zeros(N, bool); good2 = None
    for k, r in PIN_FITS.items():
        pin = 2 * r + out_b + n(0, FEAT)
        g = (pin - th <= 0.12) & (pin - th >= -0.03) & (lh - pin >= 0.03)
        good_any |= g
        if k == 2:
            good2 = g
    out["끄덕임 축 핀"] = dict(ok=float(good_any.mean()), ok_fit2_only=float(good2.mean()))
    # 3 판 틀 홈
    big_b = hole_b - p["hole"] + p["big"]                                              # 큰 홈은 덜 줄어든다 (출력마다의 흔들림은 같이 간다)
    gw = 100.0 + 2 * FR_GAP + big_b + n(0, FEAT); gl = 150.0 + 2 * FR_GAP + big_b + n(0, FEAT)
    pw, pl = n(100.0, 0.1), n(150.0, 0.1)
    fit = (gw - pw >= 0) & (gl - pl >= 0)
    out["판 틀 홈"] = dict(ok=float(fit.mean()), gap_med=float(np.median(np.minimum(gw - pw, gl - pl))))
    # 4 너트 자리
    pocket = 5.8 + hole_b + n(0, FEAT); nut = n(5.45, 0.03)
    out["너트 자리"] = dict(ok=float((pocket - nut >= -0.05).mean()))
    # 5 스스로 나사산을 내는 구멍 (M3)
    d = 2.6 + hole_b + n(0, FEAT)
    out["M3 나사 구멍"] = dict(ok=float(((d >= 2.35) & (d <= 2.80)).mean()), tight=float((d < 2.35).mean()), loose=float((d > 2.80).mean()))
    # 6 기어: 이 두께가 두 기어 다 out 만큼 커지고(유격이 준다), 축 사이가 pos 만큼 흔들린다 (넓어지면 유격이 는다, 압력각 20°)
    out["기어"] = {}
    for name, (free_deg, play_deg) in GEARS.items():
        B0 = np.radians(free_deg) * RP                                                 # 설계 유격 (피치원 둘레 방향, mm)
        dt = (out_b + n(0, FEAT)) + (out_b + n(0, FEAT))
        da = n(0, p["pos"]) + n(0, p["pos"])
        B = B0 - dt + 2 * da * np.tan(np.radians(20))
        ratio = free_deg / play_deg
        play = np.degrees(np.clip(B, 0, None) / RP + np.clip(cf, 0, None) / 2.0) / ratio   # 기어 유격 + 작은 기어가 축 위에서 노는 각 (납작 면 틈 / 2mm)
        out["기어"][name] = dict(design_mm=round(float(B0), 3), bind=float((B < 0.05).mean()), play_med=float(np.median(play)),
                                play_p95=float(np.percentile(play, 95)), design_play_deg=play_deg)
    return out


def main():
    res = dict(n=N, printers=PRINTERS, results={k: run(v) for k, v in PRINTERS.items()})
    json.dump(res, open(os.path.join(paths.RES, "print_tolerance.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, r in res["results"].items():
        print("\n[%s]" % k)
        for c in ("축 구멍", "끄덕임 축 핀", "판 틀 홈", "너트 자리", "M3 나사 구멍"):
            e = r[c]
            print("  %-10s 맞음 %5.1f%%%s%s" % (c, 100 * e["ok"], ("   (맞춤 2번만 뽑으면 %5.1f%%)" % (100 * e["ok_fit2_only"])) if "ok_fit2_only" in e else "",
                  ("  틈 0.12 이하 %.0f%%, 납작 면 틈 가운데 %.2f" % (100 * e["snug"], e["cf_med"])) if "snug" in e else ""))
        for g, e in r["기어"].items():
            print("  기어 %-6s 끼어 안 돎 %4.1f%%, 유격 %.2f° (95%%: %.2f°, 설계 %.2f°)" % (g, 100 * e["bind"], e["play_med"], e["play_p95"], e["design_play_deg"]))
    fig(res)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    INK, SUB, GRID = "#22262B", "#5B6470", "#E3DED6"
    COL = ["#9DB4D9", "#4A72B8", "#1B2A57"]                                       # 한 색의 진하기: 프린터 상태가 나쁠수록 진하다
    checks = [("축 구멍", "모터 축 구멍"), ("끄덕임 축 핀", "끄덕임 축 핀"), ("판 틀 홈", "판 틀 홈"), ("너트 자리", "너트 자리"), ("M3 나사 구멍", "M3 나사 구멍")]
    R = res["results"]; names = list(R)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    fig.patch.set_facecolor("#F4F2EE")
    y = np.arange(len(checks)); h = 0.26
    for i, nm in enumerate(names):
        v = [100 * R[nm][c]["ok"] for c, _ in checks]
        a1.barh(y + (i - 1) * h, v, h * 0.9, color=COL[i], label=nm)
        for yy, vv in zip(y + (i - 1) * h, v):
            a1.text(min(vv, 100) + 0.8, yy, "%.0f%%" % vv, va="center", fontsize=9, color=INK)
        fo = [100 * R[nm][c]["ok_fit2_only"] if "ok_fit2_only" in R[nm][c] else None for c, _ in checks]
        for yy, vv in zip(y + (i - 1) * h, fo):
            if vv is not None:
                a1.plot([vv], [yy], marker="|", ms=13, mew=2.2, color="#D9480F")
    a1.set_yticks(y); a1.set_yticklabels([l for _, l in checks], fontsize=11, color=INK); a1.invert_yaxis()
    a1.set_xlim(0, 112); a1.set_xlabel("맞는 비율 (%)  ·  주황 눈금 = 맞춤을 고르지 않고 2번만 뽑았을 때", color=SUB)
    a1.set_title("맞는 곳이 맞나 (2만 번씩)", loc="left", fontsize=13, color=INK)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
    gx = np.arange(2); w = 0.26
    for i, nm in enumerate(names):
        md = [R[nm]["기어"][g]["play_med"] for g in ("끄덕임", "받침 회전")]
        hi = [R[nm]["기어"][g]["play_p95"] for g in ("끄덕임", "받침 회전")]
        a2.bar(gx + (i - 1) * w, md, w * 0.9, color=COL[i], label=nm)
        a2.errorbar(gx + (i - 1) * w, md, yerr=[[0, 0], np.array(hi) - np.array(md)], fmt="none", ecolor=INK, capsize=3, lw=1)
        for xx, g in zip(gx + (i - 1) * w, ("끄덕임", "받침 회전")):
            b = 100 * R[nm]["기어"][g]["bind"]
            a2.text(xx, 0.04, "끼임\n%.0f%%" % b, ha="center", va="bottom", fontsize=8.5, color="#FFFFFF" if i else INK)
    for j, g in enumerate(("끄덕임", "받침 회전")):
        dp = R[names[0]]["기어"][g]["design_play_deg"]
        a2.plot([j - 0.45, j + 0.45], [dp, dp], ls="--", color="#D9480F", lw=1.5)
    a2.set_xticks(gx); a2.set_xticklabels(["끄덕임 (머리 각)", "받침 회전 (요크 각)"], fontsize=11, color=INK)
    a2.set_ylabel("기어 유격 (°)", color=SUB); a2.set_title("기어 유격 (막대 = 가운데 값, 선 = 95%)", loc="left", fontsize=13, color=INK)
    a2.text(0.98, 0.97, "점선 = 설계 유격", transform=a2.transAxes, ha="right", va="top", fontsize=9, color="#D9480F")
    for a in (a1, a2):
        a.set_facecolor("#F4F2EE"); a.grid(axis="x" if a is a1 else "y", color=GRID); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=SUB)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(paths.IMG, "36_출력_오차.png"), dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
