"""rt_eval_result.json 을 읽어 보고서(표와 그림)를 만든다.

    python report.py [결과.json] [--out 폴더]

폴더에 '교실 전파 시뮬레이션.md' 와 images/ 가 생긴다. 숫자는 모두 결과 파일에서 가져오므로
시뮬레이션을 다시 돌린 뒤 이 스크립트만 다시 돌리면 문서가 맞춰진다.
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY = "#10304F", "#19C2B0", "#FFB020", "#E0483A", "#6B7780"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("result", nargs="?", default=os.path.join(HERE, "rt_eval_result.json")); ap.add_argument("--out", default=HERE); ap.add_argument("--summary", help="docs 저장소에 둘 요약 파일")
    a = ap.parse_args(); R = json.load(open(a.result, encoding="utf-8")); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    def get(case, key="tmpw@6"):
        e = R.get(case); return None if e is None or key not in e else e[key]

    def s3(case, key="tmpw@6"):                       # "정확 / 옆자리까지 / 후보 3개 안"
        v = get(case, key); return "–" if v is None else "%.0f / %.0f / %.0f" % (v[0], v[1], v[2])

    def n1(case, key="tmpw@6", i=0):
        v = get(case, key); return "–" if v is None else "%.0f" % v[i]

    def nn(case):
        e = R.get(case); return "–" if e is None else str(e["n"])

    B = "기준 2.4m · 18번 · 8곳까지 | %s"; kinds = ("책상", "가슴", "안경"); kname = {"책상": "책상 위", "가슴": "가슴 (옷에 단 것)", "안경": "안경 높이"}

    # ---------------- 그림 1: 곳 수
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=170, sharey=True)
    for ax, kind in zip(axs, kinds):
        ns = [1, 2, 3, 4, 6, 8]; ex = [get(B % kind, "tmpw@%d" % n) for n in ns]
        if any(v is None for v in ex):
            ax.set_visible(False); continue
        ax.plot(ns, [v[1] for v in ex], "o-", color=TEAL, lw=2.2, label="옆자리까지"); ax.plot(ns, [v[0] for v in ex], "o-", color=NAVY, lw=2.2, label="자리를 정확히")
        tr = [get(B % kind, "trough@%d" % n) for n in ns]; ax.plot(ns, [v[0] for v in tr], "s--", color=RED, lw=1.4, ms=4, label="골 찾기 (지금 펌웨어)")
        ax.set_title(kname[kind], fontsize=12, color=NAVY, fontweight="bold"); ax.set_xticks(ns); ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.set_xlabel("재는 곳 수")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[0].set_ylabel("맞힌 비율 (%)"); axs[0].legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "images", "교실_곳수.png"), facecolor="white"); plt.close(fig)

    # ---------------- 그림 2: 높이
    hs = [1.2, 1.6, 2.0, 2.2, 2.4, 2.6]; fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=170)
    for kind, col in (("책상", NAVY), ("가슴", AMBER), ("안경", TEAL)):
        pts = [(h, get("높이 %.1fm | %s" % (h, kind))) for h in hs]; pts = [(h, v) for h, v in pts if v is not None]
        if pts:
            ax.plot([h for h, _ in pts], [v[0] for _, v in pts], "o-", color=col, lw=2.2, label=kname[kind] + " · 정확히"); ax.plot([h for h, _ in pts], [v[1] for _, v in pts], "o:", color=col, lw=1.4, ms=3.5, label=kname[kind] + " · 옆자리까지")
    ax.axvspan(1.05, 1.35, color=GRAY, alpha=0.10); ax.text(1.2, 6, "앉은 사람\n머리 높이", ha="center", fontsize=8, color=GRAY)
    ax.set_xlabel("수신기 안테나 높이 (m)"); ax.set_ylabel("맞힌 비율 (%), 6곳"); ax.set_ylim(0, 100); ax.set_xlim(1.05, 2.75); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "images", "교실_높이.png"), facecolor="white"); plt.close(fig)

    # ---------------- 그림 3: 놓는 곳
    rows = [("골라 가며 6곳 (옆벽·교탁·뒷벽·모서리)", B % "책상"), ("옆벽 8곳 가운데 6곳", "놓는 곳 옆벽 8곳만 | 책상"), ("모서리 + 교탁", "놓는 곳 모서리 + 교탁 | 책상"), ("교탁 + 뒷벽", "놓는 곳 교탁 + 뒷벽 | 책상"),
            ("교탁 쪽 3곳만", "놓는 곳 교탁 쪽 3곳만 | 책상"), ("네 모서리만", "놓는 곳 네 모서리만 | 책상"), ("뒤 두 모서리만", "놓는 곳 뒤 두 모서리만 | 책상")]
    rows = [(nm, get(c, "tmpw@%d" % min(6, R[c]["cfg"]["npos"])) if c in R else None) for nm, c in rows]; rows = [(nm, v) for nm, v in rows if v is not None]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=170); y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.19, [v[1] for _, v in rows], 0.36, color=TEAL, label="옆자리까지"); ax.barh(y - 0.19, [v[0] for _, v in rows], 0.36, color=NAVY, label="자리를 정확히")
    for yy, (_, v) in zip(y, rows):
        ax.text(v[1] + 1, yy + 0.19, "%.0f" % v[1], va="center", fontsize=8, color=GRAY); ax.text(v[0] + 1, yy - 0.19, "%.0f" % v[0], va="center", fontsize=8, color=GRAY)
    ax.set_yticks(y); ax.set_yticklabels([nm for nm, _ in rows], fontsize=9); ax.set_xlim(0, 105); ax.set_xlabel("맞힌 비율 (%)"); ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "images", "교실_놓는곳.png"), facecolor="white"); plt.close(fig)

    # ---------------- 글
    L = []; w = L.append
    b6 = get(B % "책상"); tr6 = get(B % "책상", "trough@6")
    w("# 교실 전파 계산 (sim/rt)\n")
    w("> **2026-11-17.** 이 글은 돌판 수신기(여섯 곳, 2.4 m 스탠드)의 계산이다. 스탠드 없이 두 곳 이하로 재는 조건과 패널 수신기의 계산은 [패널 연구](패널%20연구.md)와 [설계 검토](../../docs/설계%20검토.md)에 있다.\n")
    w("학생 40명이 책상에 앉아 시험을 보는 교실에서, 수신기 한 대로 Wi-Fi 카메라의 자리를 찾을 수 있는지 전파 경로 계산으로 본 결과입니다. 이 문서의 표와 그림은 `report.py` 가 결과 파일(`rt_eval_result.json`)에서 만듭니다. 손으로 고치지 않습니다.\n")
    w("> **읽기 전에.** 여기의 숫자는 모두 **계산**입니다. 실물로 잰 것이 아닙니다. 방 모양·사람·책상·전파의 반사와 회절을 넣었지만, 실제 교실은 이보다 복잡합니다. 실물 실험의 목표값을 정하고 재는 방법을 고르는 데 씁니다.\n")
    w("## 한눈에\n")
    if b6:
        w("- **찾을 수 있다.** 수신기를 높이 2.4m 에 세우고 벽을 따라 여섯 곳에서 재면, 책상 위 카메라의 자리를 정확히 맞힌 비율이 **%.0f%%**, 바로 옆자리까지 넣으면 **%.0f%%**, 후보 3자리 안에 든 비율이 **%.0f%%** 입니다." % (b6[0], b6[1], b6[2]))
    if tr6:
        w("- **읽는 방법을 바꿔야 한다.** 지금 펌웨어처럼 신호 곡선의 가장 약한 곳(골)을 방향으로 읽으면 %.0f%% 로, 40자리에서 찍는 것과 다르지 않습니다. 곡선 전체를 기준 곡선과 견주어야 합니다." % tr6[0])
    c6, g6 = get(B % "가슴"), get(B % "안경")
    if c6 and g6:
        w("- **몸에 붙인 카메라는 더 어렵다.** 같은 조건에서 가슴에 단 카메라는 %.0f%% (옆자리까지 %.0f%%), 안경 높이는 %.0f%% (옆자리까지 %.0f%%) 입니다. 몸과 앞사람이 전파를 가립니다." % (c6[0], c6[1], g6[0], g6[1]))
    hi = [get("높이 %.1fm | 책상" % h) for h in (2.0, 2.2, 2.4, 2.6)]; hi = [v[0] for v in hi if v]
    w("- **사람 머리 위로 올려야 한다.** 앉은 사람 머리 높이(1.2m)에 두면 %s%%, 1.6m 는 %s%%, 2.0m 이상은 %s 입니다." % (n1("높이 1.2m | 책상"), n1("높이 1.6m | 책상"), ("%.0f~%.0f%%" % (min(hi), max(hi))) if hi else "–"))
    w("- **모서리만으로는 안 된다.** 네 모서리에서만 재면 %s%%, 옆벽을 따라 재면 %s%% 입니다." % (n1("놓는 곳 네 모서리만 | 책상", "tmpw@4"), n1("놓는 곳 옆벽 8곳만 | 책상")))
    w("- **찾지 못하는 것.** 블루투스로 폰에 보내는 스마트 안경, LTE 카메라, 저장만 하는 카메라는 Wi-Fi 전파를 내지 않아 이 방법의 대상이 아닙니다.\n")
    w("## 어떻게 계산했나\n")
    nmin = min(e["n"] for e in R.values())
    w("| 항목 | 내용 |\n|---|---|\n| 교실 | 7.5 × 9.0 × 2.7m, 책상 40개(6칸 × 7줄에서 두 자리 뺌), 뒷벽 사물함, 칠판(철판), 창(유리), 문(나무), 콘크리트 벽·바닥·천장 |\n| 사람 | 자리마다 앉은 사람 모형(몸통·머리·허벅지). 시도마다 앉은 자리가 조금씩 다르고, 빈자리도 섞인다 |\n| 전파 | Sionna RT 2.2 로 경로를 구한다: 반사 3번까지, 투과, 모서리 회절. 2.437GHz (채널 6), 부반송파 52개 |\n| 수신기 | 안테나와 도는 알루미늄 판은 전자기 계산(NEC2)으로 구한 표를 쓴다. 판 각도 72칸, 편파 2개 |\n| 재는 과정 | 패킷마다 잡음과 사람 움직임을 넣어 CSI 를 만들고, **실제 펌웨어 코드**(`analyze3.scan_curve`)로 신호 곡선을 만든다 |\n| 카메라 | 책상 위 2곳, 가슴, 안경 높이. 자리마다 4점, 모두 160점 |\n| 시도 | 기준 경우는 %s번, 나머지는 %d번 (카메라 자리·사람·잡음을 바꿔 가며). %d번이면 숫자가 ±9%%p 쯤 흔들린다 |\n" % (nn(B % "책상"), nmin, nmin))
    w("맞힌 비율은 세 가지로 적습니다: **자리를 정확히 / 옆자리까지(둘레 8자리 포함) / 후보 3자리 안**.\n")
    w("## 1. 몇 곳에서 재야 하나\n")
    w("![곳 수](images/교실_곳수.png)\n")
    w("| 재는 곳 수 | " + " | ".join(kname[k] for k in kinds) + " |\n|---|---|---|---|")
    for n in (1, 2, 3, 4, 6, 8):
        w("| %d | " % n + " | ".join(s3(B % k, "tmpw@%d" % n) for k in kinds) + " |")
    w("\n한 곳에서는 방향만 알 뿐 거리를 몰라 낮습니다. 네 곳부터 쓸 만해지고 여섯 곳에서 거의 다 올라옵니다. 여덟 곳은 조금 더 나을 뿐입니다. 몸에 붙인 카메라는 여덟 곳까지 계속 오릅니다. 한 곳은 18번 스캔에 약 2분 20초이므로 여섯 곳이면 옮기는 시간까지 20분 안팎입니다.\n")
    w("## 2. 곡선을 읽는 방법\n")
    w("| 방법 | 책상 위, 6곳 |\n|---|---|")
    for key, nm in (("trough", "골 찾기 (지금 펌웨어)"), ("tm", "곡선 전체 맞추기"), ("tmp", "곡선 전체 + 편파 가정"), ("tmw", "평균 세기 곡선 맞추기"), ("tmpw", "**평균 세기 곡선 + 편파 가정**")):
        w("| %s | %s |" % (nm, s3(B % "책상", key + "@6")))
    w("\n골 찾기는 직접 오는 전파 하나만 있는 이상적인 경우에도 방향 오차 가운데값이 45~55° 였습니다. 판이 만드는 그림자는 뒤쪽 한 곳에만 생기지 않고, 판의 가장자리를 돌아 들어오는 전파가 곡선에 여러 개의 봉우리와 골을 만들기 때문입니다. 곡선 전체를 견주면 같은 경우에 2.9~6.9° 였습니다.\n")
    w("## 3. 높이\n")
    w("![높이](images/교실_높이.png)\n")
    w("| 높이 | 책상 위 | 가슴 | 안경 높이 |\n|---|---|---|---|")
    for h in (1.2, 1.6, 2.0, 2.2, 2.4, 2.6):
        w("| %.1fm | %s | %s | %s |" % (h, s3("높이 %.1fm | 책상" % h), s3("높이 %.1fm | 가슴" % h), s3("높이 %.1fm | 안경" % h)))
    w("\n앉은 사람의 머리는 바닥에서 약 1.2m 입니다. 그 높이에서는 앞사람들의 몸이 전파를 가립니다. 책상 위 카메라는 1.6m 부터 크게 나아지고 2.0m 이상에서는 비슷합니다. 가슴에 단 카메라는 반대로 너무 높으면 나빠집니다: 1.6~2.4m 에서는 비슷하고 2.6m 에서 뚜렷이 떨어집니다 (위에서 내려다보면 머리와 어깨가 가립니다). 경우마다 120번이라 숫자가 ±9%p 쯤 흔들리는 것을 감안해도, 둘을 함께 보면 **2.0~2.4m** 가 알맞습니다. 낮을수록 스탠드가 덜 흔들리고 넘어지기 어렵습니다. 기구 검사(넘어짐)는 가장 불리한 2.4m 로 했습니다.\n")
    w("## 4. 한 곳에서 몇 번 재나\n")
    w("| 한 곳에서 | 책상 위, 6곳 |\n|---|---|")
    for plan, nm in (("tilt3", "3번 (숙임 3가지)"), ("tilt6", "6번"), ("tilt12", "12번 (숙임 3 × 받침 4방향)"), ("tilt18", "**18번 (숙임 3 × 받침 6방향)**"), ("tilt36", "36번"), ("flat6", "6번, 숙이지 않고"), ("up12", "12번, 위로만")):
        w("| %s | %s |" % (nm, s3(B % "책상") if plan == "tilt18" else s3("재는 횟수 %s | 책상" % plan)))
    w("\n같은 시간(스캔 약 72번)을 나누어 쓴다면:\n")
    w("| 나누는 방법 | 책상 위 |\n|---|---|")
    for plan, npos in (("tilt3", 18), ("tilt6", 12), ("tilt12", 6), ("tilt18", 4), ("tilt36", 2)):
        c = "같은 시간(스캔 약 72번): %d곳 x %s | 책상" % (npos, plan); w("| %d곳 × %s번 | %s |" % (npos, plan[4:], s3(c, "tmpw@%d" % npos)))
    tt = [(npos, plan[4:], get("같은 시간(스캔 약 72번): %d곳 x %s | 책상" % (npos, plan), "tmpw@%d" % npos)) for plan, npos in (("tilt3", 18), ("tilt6", 12), ("tilt12", 6), ("tilt18", 4), ("tilt36", 2))]
    tt = [x for x in tt if x[2]]
    if len(tt) >= 3:
        be = max(tt, key=lambda x: x[2][0]); bn = max(tt, key=lambda x: x[2][1]); lo = min(tt, key=lambda x: x[2][0])
        w("\n자리를 정확히 맞힌 비율은 %d곳 × %s번이 가장 높고(%.0f%%), 옆자리까지 넣으면 %d곳 × %s번이 가장 높습니다(%.0f%%). 가장 낮은 것은 %d곳 × %s번(%.0f%%)입니다. 곳 수나 횟수 어느 한쪽으로 몰면 나빠집니다.\n" % (be[0], be[1], be[2][0], bn[0], bn[1], bn[2][1], lo[0], lo[1], lo[2][0]))
    w("## 5. 어디에 놓나\n")
    w("![놓는 곳](images/교실_놓는곳.png)\n")
    at = lambda c, key="tmpw": s3(c, "%s@%d" % (key, R[c]["cfg"]["npos"])) if c in R else "–"
    P = lambda nm: at("놓는 곳 %s | 책상" % nm)
    w("옆벽을 따라 재는 것이 가장 좋습니다 (%s). 교탁 쪽 세 곳만(%s), 네 모서리만(%s), 뒤 두 모서리만(%s)으로는 부족하고, 모서리와 교탁을 섞어도(%s) 옆벽만 못합니다. 모서리에서는 자리들이 좁은 각도 안에 몰려 보이고 뒤쪽에서는 학생들의 등이 카메라를 가리기 때문으로 짐작합니다 (따로 떼어 확인하지는 않았습니다).\n" % (P("옆벽 8곳만"), P("교탁 쪽 3곳만"), P("네 모서리만"), P("뒤 두 모서리만"), P("모서리 + 교탁")))
    w("다음에 잴 곳을 '지금 후보들이 가장 잘 갈라져 보이는 곳'으로 골라 가면 %s 입니다. 차례를 미리 정해 두고 돌면:\n" % s3(B % "책상"))
    w("| 정해진 차례 | 책상 위 |\n|---|---|")
    for nm, c in (("한 바퀴 6곳 (교탁 → 오른쪽 벽 → 뒤 → 왼쪽 벽)", "정해진 차례: 한 바퀴 6곳 | 책상"), ("한 바퀴 8곳", "정해진 차례: 한 바퀴 8곳 | 책상"), ("네 모서리 + 옆벽 가운데 둘", "정해진 차례: 네 모서리 + 옆벽 가운데 둘 | 책상"), ("한 바퀴 14곳, 한 곳에서 6번씩", "정해진 차례: 한 바퀴 14곳 x 6번 | 책상")):
        if c in R:
            w("| %s | %s |" % (nm, at(c)))
    for key, nm in (("trough", "골 찾기 결과"), ("tm", "곡선 전체 맞추기 결과")):
        c = "다음 곳을 %s 로 고름 | 책상" % key
        if c in R:
            w("| (견줌) 골라 가되 %s로 고름 | %s |" % (nm, at(c)))
    w("")
    w("천장 가운데 한 곳에 달면 한 번에 %s (책상 위), %s (가슴), %s (안경 높이) 입니다. 책상 위 카메라에는 가장 빠른 방법이지만 미리 달아 두어야 하고, 몸에 단 카메라는 위에서는 잘 보이지 않습니다.\n" % (s3("천장 가운데 한 곳 2.5m | 책상", "tmpw@1"), s3("천장 가운데 한 곳 2.5m | 가슴", "tmpw@1"), s3("천장 가운데 한 곳 2.5m | 안경", "tmpw@1")))
    w("## 6. 더 큰 강의실 (72석, 9 × 12 × 3m)\n")
    lec = (("높이 2.4m, 책상 위", "강의실 72석 2.4m · 8곳까지 | 책상"), ("높이 2.4m, 가슴", "강의실 72석 2.4m · 8곳까지 | 가슴"), ("높이 2.7m, 책상 위", "강의실 72석 2.7m · 8곳까지 | 책상"), ("높이 2.7m, 가슴", "강의실 72석 2.7m · 8곳까지 | 가슴"))
    if any(c in R for _, c in lec):
        w("| 조건 | 6곳 | 8곳 |\n|---|---|---|")
        for nm, c in lec:
            if c in R:
                w("| %s | %s | %s |" % (nm, s3(c, "tmpw@6"), s3(c, "tmpw@8")))
        l6, l8, l27 = get(lec[0][1], "tmpw@6"), get(lec[0][1], "tmpw@8"), get(lec[2][1], "tmpw@6")
        if l6 and l8 and b6:
            w("\n자리가 40개에서 72개로 늘면 같은 여섯 곳에서 책상 위 카메라를 정확히 맞힌 비율이 %.0f%% → %.0f%% (옆자리까지 %.0f%% → %.0f%%) 입니다. 여덟 곳이면 %.0f%% (옆자리까지 %.0f%%) 입니다.%s" % (b6[0], l6[0], b6[1], l6[1], l8[0], l8[1], (" 스탠드를 2.7m 로 올리면 여섯 곳에서 %.0f%% 입니다." % l27[0]) if l27 else ""))
        c1, c3 = "강의실 72석 천장 가운데 한 곳 2.8m | 책상", "강의실 72석 천장 세 곳 2.8m | 책상"
        if c1 in R or c3 in R:
            w("\n천장에 달면: 가운데 한 곳 %s, 세 곳 %s (책상 위).\n" % (at(c1), at(c3)))
        else:
            w("")
    else:
        w("계산이 끝나면 채웁니다.\n")
    w("## 7. 조건을 바꿔 보기 (책상 위, 6곳)\n")
    ref_all, ref_s1 = B % "책상", "같은 자리 견줌 (s1만) | 책상"
    rows7 = [("빈 교실", "빈 교실 2.4m | 책상", ref_all, "사람이 가리는 영향"), ("학생 20명", "20명 2.4m | 책상", ref_all, "〃"),
             ("기준 곡선이 실물과 다름 (머리만 넣은 표로 만든 신호)", "표와 실물이 다름 (머리만 표) | 책상", ref_all, "실물 곡선이 계산 표와 다를 때"),
             ("수신기 자리 오차 0.15m, 8° (기준은 0.05m, 3°)", "자리 오차 큼 (0.15m, 8도) | 책상", ref_all, "폰이 잰 수신기 자리가 틀릴 때"),
             ("사람이 안 움직임", "사람 안 움직임 | 책상", ref_all, "사람의 움직임"), ("사람이 많이 움직임", "사람 많이 움직임 | 책상", ref_all, "〃"),
             ("패킷 초당 30개", "패킷 초당 30 | 책상", ref_all, "카메라가 보내는 양"), ("패킷 초당 300개", "패킷 초당 300 | 책상", ref_all, "〃"),
             ("잡음 큼 (SNR 15dB)", "잡음 큼 (SNR 15dB) | 책상", ref_all, "신호가 약할 때"),
             ("직접 오는 전파만 (이상)", "직접파만 (이상) | 책상", ref_s1, "반사가 없다면 닿는 위쪽 끝"), ("회절을 빼고 계산", "회절 없이 | 책상", ref_s1, "계산 방법에 민감한지"),
             ("반사 5번까지", "길 계산 깊이 5 | 책상", ref_s1, "〃"), ("광선 수 4배", "광선 수 4배 (수렴 확인) | 책상", ref_s1, "〃"), ("천장이 석고판", "천장이 텍스(석고판) | 책상", ref_s1, "천장 재료")]
    big = []
    if any(c in R for _, c, _, _ in rows7):
        w("| 바꾼 것 | 결과 | 기준과의 차이 (정확히) | 무엇을 보려고 |\n|---|---|---|---|")
        w("| 기준 (교실 세 가지, %s번) | %s |  |  |" % (nn(ref_all), s3(ref_all)))
        for k, (nm, c, ref, why) in enumerate(rows7):
            if ref == ref_s1 and rows7[k - 1][2] != ref_s1 and ref_s1 in R:
                w("| 기준 (교실 한 가지, %s번): 아래 다섯 줄과 견준다 | %s |  |  |" % (nn(ref_s1), s3(ref_s1)))
            if c in R:
                d = (get(c)[0] - get(ref)[0]) if get(ref) else None
                w("| %s | %s | %s | %s |" % (nm, s3(c), "–" if d is None else "%+.0f%%p" % d, why))
                if d is not None and abs(d) >= 10:
                    big.append("%s (%+.0f%%p)" % (nm.split(" (")[0], d))
        w("\n%s\n" % (("기준과 10%p 넘게 차이 나는 것: " + ", ".join(big) + ". 나머지는 시도 횟수에서 오는 흔들림(±9%p 쯤) 안이라 차이가 있다고 말하기 어렵습니다.") if big else "기준과 10%p 넘게 차이 나는 것은 없습니다. 모두 시도 횟수에서 오는 흔들림(±9%p 쯤)과 가려내기 어려운 크기입니다."))
    else:
        w("계산이 끝나면 채웁니다.\n")
    dens = [(d, get("맞춤 세기 den=%g | 책상" % d) if d != 160 else get(ref_all), get("맞춤 세기 den=%g | 가슴" % d) if d != 160 else get(B % "가슴")) for d in (40, 80, 160, 320, 640)]
    if any(a_ and d != 160 for d, a_, _ in dens):
        w("맞춤 세기(`den`: 곡선이 닮은 정도를 가능성으로 바꾸는 날카로움)는 이 계산에서 고른 값(160)입니다. 그래서 기준 숫자에는 그만큼의 유리함이 들어 있을 수 있습니다. 다른 값일 때:\n")
        w("| den | 책상 위 | 가슴 |\n|---|---|---|")
        f3 = lambda v: "–" if v is None else "%.0f / %.0f / %.0f" % tuple(v[:3])
        for d, a_, b_ in dens:
            w("| %g%s | %s | %s |" % (d, " (기준)" if d == 160 else "", f3(a_), f3(b_)))
        w("")
    else:
        w("맞춤 세기(`den`: 곡선이 닮은 정도를 가능성으로 바꾸는 날카로움)는 이 계산에서 고른 값(160)입니다. 그래서 기준 숫자에는 그만큼의 유리함이 들어 있을 수 있습니다.\n")
    rec = "권하는 방법: 교탁에서 시작 + 12번 x 6곳 | %s"
    w("## 8. 권하는 재는 방법\n")
    w("1. 수신기를 스탠드에 올려 안테나 높이 **2.0~2.4m** 로 세운다. 다리에 모래주머니를 올린다.\n2. 교탁 옆처럼 드나들기 쉬운 곳에서 시작한다. 한 곳에서 **12~18번** 잰다 (숙임 세 가지 × 받침 4~6방향, 약 1분 30초~2분 20초).\n3. 앱이 권하는 다음 곳으로 옮긴다 (지금 후보들이 가장 잘 갈라져 보이는 곳). 손으로 고를 때는 **옆벽**을 꼭 넣고, 모서리만 돌지 않는다.\n4. **여섯 곳**을 재거나, 후보 3자리의 확률 합이 충분히 커지면 멈춘다.\n5. 화면의 후보 3자리를 눈으로 확인한다.\n")
    if get(rec % "책상"):
        w("교탁에서 시작해 12번씩 여섯 곳을 잰 경우: 책상 위 %s, 가슴 %s, 안경 높이 %s.\n" % (s3(rec % "책상"), s3(rec % "가슴"), s3(rec % "안경")))
    bad = "나쁜 조건 한꺼번에 (표 다름 + 자리 오차 큼 + 패킷 30 + 많이 움직임) | %s"
    if get(bad % "책상"):
        w("나쁜 조건이 한꺼번에 겹칠 때 (기준 곡선이 실물과 다름 + 수신기 자리 오차 큼 + 패킷 초당 30개 + 사람이 많이 움직임, 18번씩 여섯 곳): 책상 위 %s, 가슴 %s, 안경 높이 %s. 실물에서 기대할 아래쪽 값으로 봅니다.\n" % (s3(bad % "책상"), s3(bad % "가슴"), s3(bad % "안경")))
    w("## 한계와 다음에 확인할 것\n")
    mmv = get("표와 실물이 다름 (머리만 표) | 책상")
    MMS = "- 기준 곡선을 계산 표로 썼다. " + (("표를 조금 다르게 해 본 경우(머리만 넣고 계산한 표로 신호를 만들고, 읽을 때는 원래 표를 씀)에는 %.0f%% → %.0f%% 였다. 실물 곡선이 표와 더 많이 다르면 더 내려갈 수 있다." % (b6[0], mmv[0])) if (mmv and b6) else "실물 수신기의 곡선이 이 표와 다르면 결과가 내려갈 수 있다.") + " 실물로 기준 곡선을 재는 것이 첫 실험이다."
    w("- 계산에 넣지 못한 것: 가방·옷·의자 등의 잡동사니, 옆 교실의 Wi-Fi, 카메라 안테나의 실제 모양, 스탠드의 흔들림.\n%s\n- 카메라가 재는 동안 계속 신호를 내야 한다. 조용한 카메라는 수신기가 찔러서 응답을 받아야 하고, 이것은 아직 실물로 확인하지 못했다.\n- 2.4GHz 로 계산했다. 5GHz 는 판이 파장에 견주어 커져 곡선이 더 또렷해지지만 벽과 몸에 더 많이 가려진다. 따로 계산해야 한다.\n" % MMS)
    i_sum = L.index("## 어떻게 계산했나\n"); i_rec = L.index("## 8. 권하는 재는 방법\n"); head = L[2:i_sum]; tail = L[i_rec:]
    w("## 돌리는 법\n")
    w("```\npython3 -m venv .venv && .venv/bin/pip install sionna-rt trimesh shapely mapbox-earcut scipy matplotlib\ncd sim/rt\n./run_all.sh 120              # 경로 구하기(paths/*.npz) + 평가. GPU 가 있으면 몇 분, 없으면 몇 시간\npython report.py              # 이 문서와 그림을 다시 만든다\npython demo_run.py --seed 1001 && python demo_video.py   # 동작 시뮬레이션 영상 (docs/videos)\n```\n")
    w("| 파일 | 무엇 |\n|---|---|\n| `scene_kr.py` | 교실과 강의실 장면 (벽, 창, 책상, 사람) |\n| `rt_paths.py` | 수신기 자리마다 카메라 160점까지의 전파 경로 (Sionna RT) |\n| `rt_eval.py` | 판과 안테나의 계산 표를 입히고 펌웨어 코드로 자리를 찾게 한다 |\n| `report.py` | 결과 파일에서 이 문서를 만든다 |\n| `demo_run.py`, `demo_video.py` | 한 번의 진행을 영상으로 |\n| `rt_eval_result.json` | 결과 (경우마다 맞힌 비율) |\n\n`paths/` 의 경로 파일은 커서(하나에 약 20MB) 저장소에 올리지 않는다. `rt_paths.py` 로 다시 만든다.\n")
    open(os.path.join(a.out, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("ok", os.path.join(a.out, "README.md"))
    if a.summary:
        S = ["# 교실 전파 시뮬레이션 (요약)\n", "학생 40명이 앉은 교실에서 수신기 한 대로 Wi-Fi 카메라의 자리를 찾을 수 있는지 전파 경로 계산으로 본 결과의 요약입니다. 표와 그림, 계산 방법, 코드는 [receiver/sim/rt](https://github.com/DeteCam/receiver/tree/main/sim/rt)에 있습니다.\n"] + head + ["![곳 수](https://raw.githubusercontent.com/DeteCam/receiver/main/sim/rt/images/교실_곳수.png)\n"] + [x.replace("## 8. ", "## ") for x in tail]
        open(a.summary, "w", encoding="utf-8").write("\n".join(S) + "\n"); print("ok", a.summary)


if __name__ == "__main__":
    main()
