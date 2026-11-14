"""3축 화면을 만든다: 이 폴더의 template.html 에 계산 코드(core3.js)와 데이터(data.json)를 넣어 HTML 한 장으로.
    python3 build3.py                   ->  ../dist/DeteCam_3축_화면.html   혼자 열리는 파일 (three.js 도 안에 넣는다. 인터넷 없이 열린다)
                                            저장소에서는 ../../docs/simulator/index.html 에도 같은 파일을 둔다 (깃허브 Pages 주소로 열린다)
    python3 build3.py --out 경로         ->  혼자 열리는 파일을 다른 곳에
    python3 build3.py --fragment 경로    ->  머리말 없는 조각 (three.js 는 CDN 에서 받는다). 웹에 올릴 때 쓴다
파이썬만 있으면 된다. 데이터(data.json)는 viewer_data3.py 가, 화면의 글과 그림은 template.html 이, 계산은 core3.js 가 맡는다."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2          # 처음 화면에 미리 돌려 두는 예시의 방 번호 (오차가 가운데값쯤 나오는 보통 경우)
THREE_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'


def read(name):
    return open(os.path.join(HERE, name), encoding="utf-8").read()


def fragment():
    html = read("template.html"); core = read("core3.js")
    blob = json.dumps(json.loads(read("data.json")), ensure_ascii=False, separators=(',', ':'))
    assert "</script" not in core and "</script" not in blob
    for k, v in {"/*__CORE__*/": core, "/*__DATA__*/null": blob, "/*__SEED__*/1": str(SEED)}.items():
        assert html.count(k) == 1, k
        html = html.replace(k, v)
    return html


def standalone(frag):
    """조각을 혼자 열리는 문서로 만든다. three.js 는 파일 안에 넣는다."""
    three = read("three.min.js")
    assert frag.count(THREE_TAG) == 1 and "</script" not in three
    body = frag.replace(THREE_TAG, "<script>" + three + "</script>")
    head = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>html{color-scheme:light}body{margin:0;font-size:14px}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n')
    return head + body + "\n</body>\n</html>\n"


if __name__ == "__main__":
    a = sys.argv[1:]; frag = fragment(); up = os.path.dirname(HERE); did = False
    out = os.path.join(up, "dist" if os.path.isdir(os.path.join(up, "dist")) else "", "DeteCam_3축_화면.html")       # 저장소에서는 dist/ 에
    if "--fragment" in a:
        p = a[a.index("--fragment") + 1]; open(p, "w", encoding="utf-8").write(frag); did = True
        print("조각 %.2f MB -> %s" % (len(frag.encode()) / 1e6, p))
    if "--out" in a:
        out = a[a.index("--out") + 1]; did = False
    if not did:
        s = standalone(frag); open(out, "w", encoding="utf-8").write(s)
        print("화면 %.2f MB -> %s" % (len(s.encode()) / 1e6, out))
        note = os.path.join(os.path.dirname(up), "docs", "simulator")            # 저장소에서는 문서 폴더에도 같은 파일을 둔다 (옵시디언 노트와 깃허브 Pages 가 이 파일을 보여 준다)
        if "--out" not in a and os.path.isdir(note):
            open(os.path.join(note, "index.html"), "w", encoding="utf-8").write(s); print("문서 폴더에도 ->", note)
