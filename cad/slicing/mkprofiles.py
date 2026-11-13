"""슬라이싱에 쓸 설정 파일을 만든다: Bambu Studio 기본 설정(편 것)에 이 수신기에 맞춘 값을 덧씌운다.
    python3 -I mkprofiles.py <편 기본 설정 폴더> <결과 폴더>"""
import json, os, sys
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MIDNIGHT = "#1B2A57"


def make(src, dst, extra):
    j = json.load(open(os.path.join(SRC, src), encoding="utf-8"))
    j.update(extra)
    json.dump(j, open(os.path.join(OUT, dst), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("%-34s <- %-40s 덧씌운 값 %s" % (dst, src, {k: v for k, v in extra.items()}))


# 프린터: X1 Carbon, 노즐 0.4. model_id 는 프린터 종류 표시(BL-P001 = X1 Carbon)
make("Bambu Lab X1 Carbon 0.4 nozzle.json", "machine_X1C_0.4.json", {"model_id": "BL-P001", "nozzle_volume_type": ["Standard"]})
# 출력 방식: 0.20mm Standard 에서 벽 3겹, 서포트 끔. 판 종류는 Textured PEI (PLA 55도)
COMMON = {"wall_loops": "3", "enable_support": "0", "curr_bed_type": "Textured PEI Plate"}
make("0.20mm Standard @BBL X1C.json", "process_0.20_벽3_채움15.json", dict(COMMON, sparse_infill_density="15%"))
# 채움 100% 에서는 격자(grid) 무늬를 쓸 수 없어서 줄무늬(zig-zag = Rectilinear)로 바꾼다
make("0.20mm Standard @BBL X1C.json", "process_0.20_벽3_채움100.json", dict(COMMON, sparse_infill_density="100%", sparse_infill_pattern="zig-zag"))
# 필라멘트: Generic PLA (노즐 220, 판 55). 색은 미리보기용으로 미드나잇 블루
make("Generic PLA.json", "filament_Generic_PLA_미드나잇블루.json", {"default_filament_colour": [MIDNIGHT]})
