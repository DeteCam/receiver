"""구매 전 검증 계산의 결과를 한 파일(result.json)로 모은다.

    python3 collect.py            # ../rt/study/log_pre.txt, room_*.json 을 읽는다 (run_pre.sh, run_room.sh 가 만든 것)

study 폴더는 커서 저장소에 올리지 않는다. 문서와 그림(figs.py)은 이 result.json 의 숫자를 쓴다.
"""
import os, re, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ST = os.path.join(HERE, "..", "rt", "study")
ROW = re.compile(r"^(.*?)\s+(?:(\d+)순간\s+)?책상\s+(\d+)/\s*(\d+)/\s*(\d+) \| 가슴\s+(\d+)/\s*(\d+)/\s*(\d+) \| 안경\s+(\d+)/\s*(\d+)/\s*(\d+)\s*$")


def parse_pre(path):
    out = {}; key = None; sec = None
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln.startswith("## "):
            p = ln[3:].split(); key = " ".join(p[:2]) if len(p) > 1 and p[0] != "single" else p[0]; out[key] = {}; sec = None; continue    # 같은 이름이 다시 나오면 뒤의 것이 남는다
        if ln.startswith("-- "):
            sec = ln[3:].strip(); continue
        m = ROW.match(ln)
        if m and key:
            name = m.group(1).strip(); k = int(m.group(2)) if m.group(2) else 0; v = [int(x) for x in m.groups()[2:]]
            lab = (sec + " | " if sec else "") + name + (" | %d번" % k if k else "")
            out[key][lab] = [v[0:3], v[3:6], v[6:9]]                # 책상, 가슴, 안경 마다 (정확히, 옆자리까지, 후보 3자리)
    return out


def main():
    res = {"note": "정확히 / 옆자리까지 / 후보 3자리 (%). 책상 위, 가슴, 안경 순. run_pre.sh 와 run_room.sh 의 결과", "pre": parse_pre(os.path.join(ST, "log_pre.txt")), "room": {}}
    for tag, f in (("기본", "room_ura4x4p37.5h.json"), ("나쁜 조건", "room_ura4x4p37.5h_bad.json"), ("그대로 더함", "room_ura4x4p37.5h_sum.json"), ("평균을 빼고 더함", "room_ura4x4p37.5h_center.json")):
        p = os.path.join(ST, f)
        if os.path.exists(p):
            res["room"][tag] = json.load(open(p, encoding="utf-8"))
    json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("모음: 교실 %d 묶음, 방 %d 묶음" % (len(res["pre"]), len(res["room"])))
    for k, v in res["pre"].items():
        print("  %-28s %d줄" % (k, len(v)))


if __name__ == "__main__":
    if len(sys.argv) > 1: ST = sys.argv[1]
    main()
