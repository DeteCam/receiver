"""파일을 어디에 쓰는지 한곳에 모았다.
   저장소 (receiver-cad/cad 안에서 실행):  STL -> ../stl, 그림 -> ../그림, 계산 결과(json) -> ./결과, 슬라이싱 결과 -> ../슬라이스
   작업용 배치 (옆에 out2 폴더가 있을 때):      out2/stl, out2/img, out2/*.json"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
if os.path.isdir(os.path.join(UP, "out2")):
    RES = os.path.join(UP, "out2"); STL = os.path.join(RES, "stl"); IMG = os.path.join(RES, "img"); SLICE = os.path.join(RES, "슬라이스")
else:
    RES = os.path.join(HERE, "결과"); STL = os.path.join(UP, "stl"); IMG = os.path.join(UP, "그림"); SLICE = os.path.join(UP, "슬라이스")
TMP = os.environ.get("DETECAM_TMP") or os.path.join(HERE, "_tmp")          # 임시 파일 (렌더 중간 그림)
for d in (RES, STL, IMG, TMP):
    os.makedirs(d, exist_ok=True)
