#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""점검 기록 읽기: 수신기가 남긴 점검 기록(~/detecam/data/watch_*.jsonl, 바퀴마다의 탐지 표)을 core/score.py 로 다시 계산해
실험 S1~S5 (docs/의심 점수.md 10절)의 숫자를 낸다. 점수 규칙을 고친 뒤 같은 기록으로 다시 돌려 견줄 수 있다.

  python3 watch_report.py 기록.jsonl [--camera 주소 ...] [--hub 주소 ...] [--mine 주소 ...] [--ignore-walk]

  --camera  정답 카메라의 주소 (여럿 가능)        --hub  정답 허브(핫스팟·카메라가 붙은 공유기)의 주소
  --mine    내 기기 (점수를 매기지 않는다)        --ignore-walk  기록 속 걷기 확인 결과를 빼고 계산
나오는 것
  정답 기기마다: 처음 들린 때, '의심 기기'·'카메라 가능성 높음'에 처음 오른 때 (점검 시작에서 몇 초), 마지막 점수와 이유
  잘못 울림: 정답이 아닌데 '의심 기기' 이상에 오른 기기 수와, 한 시간에 몇 번 꼴인지 (S5)
  확인 요청: 걷기 확인을 하자고 한 때가 있었는지 (S1, S2)"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import score as S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log"); ap.add_argument("--camera", nargs="*", default=[]); ap.add_argument("--hub", nargs="*", default=[])
    ap.add_argument("--mine", nargs="*", default=[]); ap.add_argument("--ignore-walk", action="store_true")
    a = ap.parse_args()
    truth = set(m.lower() for m in a.camera + a.hub)
    T = S.Tracker(); first_seen = {}; asked = []
    for m in a.mine:
        T.mine(m.lower(), True)
    for line in open(a.log, encoding="utf-8"):
        c = json.loads(line)
        if not a.ignore_walk:
            for m, v in (c.get("walk") or {}).items():
                T.walked(m, v)
        for m in c.get("mine") or []:
            T.mine(m, True)
        T.update(c["rows"], c["t"])
        for r in c["rows"]:
            first_seen.setdefault(r["mac"], c["t"])
        if T.result()["need_confirm"]:
            asked.append(c["t"])
    R = T.result(); t0 = T.t0; t1 = T.last_t; hours = max((t1 - t0) / 3600.0, 1e-9)
    dev = dict((d["mac"], d) for d in R["devices"]); first_alarm = {}
    for x in R["alarms"]:
        first_alarm.setdefault((x["mac"], x["level"]), x["t"])
    print("점검 %d바퀴, %.1f분, 기준선 기기 %d개" % (R["cycles"], (t1 - t0) / 60.0, R["baseline"]))
    for m in sorted(truth):
        d = dev.get(m)
        if d is None:
            print("  [정답] %s: 한 번도 들리지 않음" % m); continue
        f = lambda lv: ("%.0f초" % (first_alarm[(m, lv)] - t0)) if (m, lv) in first_alarm else "-"
        print("  [정답] %s: 처음 들림 %.0f초, 의심 %s, 카메라 %s, 마지막 %d점 %s" % (m, first_seen[m] - t0, f("suspect"), f("camera"), d["score"], S.LEVEL_NAME[d["level"]]))
        print("         이유: %s" % " + ".join(d["reasons"]))
    false = sorted(set(x["mac"] for x in R["alarms"] if x["mac"] not in truth))
    print("  잘못 울림: %d개 기기 (한 시간에 %.1f번 꼴)%s" % (len(false), len(false) / hours, (": " + ", ".join(false)) if false else ""))
    for m in false[:10]:
        print("    %s %d점: %s" % (m, dev[m]["score"], " + ".join(dev[m]["reasons"])))
    print("  확인 요청: %s" % (("%d바퀴에서 (처음 %.0f초)" % (len(asked), asked[0] - t0)) if asked else "없음"))


if __name__ == "__main__":
    main()
