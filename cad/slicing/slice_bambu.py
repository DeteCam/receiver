"""Bambu Studio(명령줄)로 출력판을 만들고 슬라이싱한다.

    python3 slice_bambu.py <STL 폴더> <결과 폴더> [--only 이름조각] [--parts | --parts-only]

판마다 Bambu Studio 를 두 번 돌린다. 둘 다 같은 STL, 같은 자리, 같은 설정을 쓴다.
  1) 프로젝트 파일(.3mf): 부품 + 놓는 자리 + 설정. Bambu Studio 에서 열어 고칠 수 있다.     -> 결과/프로젝트/
  2) 바로 출력 파일(.gcode.3mf): 슬라이싱까지 끝낸 것. 프린터로 바로 보낼 수 있다.           -> 결과/바로출력/
미리보기 그림은 Bambu Studio 가 그린 것을 꺼낸다.                                           -> 결과/미리보기/
--parts 를 주면 부품을 하나씩 따로 슬라이싱해서 부품별 시간·무게도 구한다 (부품표용).

놓는 자리는 자동 배치에 맡기지 않고 아래 PLATES 에 직접 적었다 (판 가장자리와 앞쪽 금지 구역을 피하려고).
슬라이싱한 뒤 G코드를 읽어서 '적은 자리에 실제로 찍히는지'까지 확인한다.
"""
import json, os, re, shutil, subprocess, sys, tempfile, zipfile, csv
import numpy as np
import trimesh
from shapely.geometry import MultiPoint, box as sbox

HERE = os.path.dirname(os.path.abspath(__file__))
BS = os.environ.get("BAMBU_STUDIO") or os.path.join(HERE, "bs.sh")       # Bambu Studio 실행 파일. 기본은 화면 없이 돌리는 bs.sh
SET = os.path.join(HERE, "settings")
MACHINE = os.path.join(SET, "machine_X1C_0.4.json")
PROC = {15: os.path.join(SET, "process_0.20_벽3_채움15.json"), 100: os.path.join(SET, "process_0.20_벽3_채움100.json")}
FIL = os.path.join(SET, "filament_Generic_PLA_미드나잇블루.json")
COLOUR = "#1B2A57"
BED = 256.0
Y_OFF = 2.0                    # X1 Carbon 은 G코드의 Y 가 모델 자리보다 2mm 작다 (extruder_offset = 0x2)
EDGE = 3.0                     # 판 가장자리에서 이만큼은 띄운다
FRONT = 20.0                   # 앞쪽 띠(y < 20)는 비운다: 프린터가 시작할 때 여기에 시험 줄을 긋는다 (x 18~240, y 0~14)
EXCL = (18.0, 28.0)            # 앞 왼쪽 모서리 금지 구역 (bed_exclude_area)
GAP = 4.0                      # 부품 사이 최소 틈

# 판 이름: (설명, 채움 %, [(STL 이름, 가운데 x, 가운데 y, 돌리는 각도)])
XS6 = (73.0, 95.0, 117.0, 139.0, 161.0, 183.0)
FIT6 = (1, 1, 2, 2, 3, 3)
PLATES = {
    "0판_시험조각": ("맨 먼저 뽑는다: 시험 조각, 시험용 돌기, 모터 2 맞춤 링, 핀 3종 × 2개, 작은 기어 3종 × 2개 (핀·기어는 왼쪽부터 맞춤 1, 1, 2, 2, 3, 3)", 15,
                [("90_test_coupon", 105.0, 158.0, 0), ("91_tenon_gauge", 160.0, 158.0, 0), ("16_m2_ring", 186.0, 158.0, 0)]
                + [("13_pin_fit%d" % f, x, 118.0, 0) for f, x in zip(FIT6, XS6)] + [("05_pinion_fit%d" % f, x, 88.0, 0) for f, x in zip(FIT6, XS6)]),
    "1판_받침_S": ("받침 통(작은 것), 받침 덮개, 고리, 랜선 홈 마개", 15, [("02_base_cover_S", 71.5, 90.5, 0), ("01_base_tub_S", 182.5, 182.5, 0), ("03_clip", 222.0, 52.0, 0), ("17_lan_cap", 176.0, 62.0, 0)]),
    "2판_요크": ("요크 (회전판 + 팔 + 전선 상자)", 15, [("04_yoke", 128.0, 130.0, 0)]),
    "3판_머리": ("머리 몸통, 머리 뚜껑, 모터 자리, 부채꼴 기어", 15, [("07_head_shell", 66.5, 86.0, 0), ("08_head_lid", 187.0, 86.0, 0), ("09_crown", 66.5, 184.0, 0), ("06_sector_gear", 187.0, 195.0, 0)]),
    "선택_받침통_L": ("큰 배터리용 받침 통 (지름 190)", 15, [("01_base_tub_L", 128.0, 132.0, 0)]),
    "선택_받침덮개_L": ("큰 받침의 덮개 (지름 186)", 15, [("02_base_cover_L", 128.0, 132.0, 0)]),
    "선택_판200": ("비교용 판 100×200 + 균형 덩어리. 채움 100%", 100, [("12_blade_100x200", 128.0, 90.0, 0), ("15_balance_200", 128.0, 175.0, 0)]),
}
for n in (1, 2, 3):
    PLATES["4판_회전부_맞춤%d" % n] = ("회전 막대(맞춤 %d) + 판 100×150 + 균형 조각 2장. 채움 100%%" % n, 100,
                                 [("11_blade_100x150", 128.0, 80.0, 0), ("10_rotor_bar_fit%d" % n, 128.0, 162.0, 0), ("14_trim_plate", 30.0, 162.0, 0), ("14_trim_plate", 226.0, 162.0, 0)])
ORDER = ["0판_시험조각", "1판_받침_S", "2판_요크", "3판_머리", "4판_회전부_맞춤1", "4판_회전부_맞춤2", "4판_회전부_맞춤3", "선택_받침통_L", "선택_받침덮개_L", "선택_판200"]
FULL = ("10_rotor_bar", "11_blade", "12_blade", "14_trim_plate", "15_balance_200")          # 따로 슬라이싱할 때 채움 100% 로 하는 부품 (판에 올릴 때와 같게)


def run(args, cwd, log):
    with open(log, "w") as f:
        r = subprocess.run([BS, "--debug", "2"] + args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    return r.returncode


def place(stl_dir, items, tmp):
    """STL 을 돌리고 옮겨서 임시 폴더에 다시 저장한다. (파일 목록, 위에서 본 모양 목록, 높이 목록, 첫 층 범위 목록)"""
    files, shapes, heights, firsts = [], [], [], []
    for i, (name, cx, cy, rot) in enumerate(items):
        m = trimesh.load(os.path.join(stl_dir, name + ".stl"))
        if rot:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot), [0, 0, 1]))
        b = m.bounds
        m.apply_translation([cx - (b[0][0] + b[1][0]) / 2, cy - (b[0][1] + b[1][1]) / 2, -b[0][2]])
        d = os.path.join(tmp, "%02d" % i); os.makedirs(d)
        f = os.path.join(d, name + ".stl"); m.export(f); files.append(f)
        shapes.append(MultiPoint(m.vertices[:, :2]).convex_hull); heights.append(float(m.bounds[1][2]))
        low = m.vertices[m.vertices[:, 2] < 0.01][:, :2]; firsts.append([float(low[:, 0].min()), float(low[:, 1].min()), float(low[:, 0].max()), float(low[:, 1].max())])
    return files, shapes, heights, firsts


def layout_problems(items, shapes):
    """적은 자리가 판 안에 있고, 금지 구역을 피하고, 서로 떨어져 있는지"""
    probs = []; gap_min = 1e9; edge_min = 1e9
    for (name, cx, cy, rot), s in zip(items, shapes):
        x0, y0, x1, y1 = s.bounds
        edge = min(x0, BED - x1, BED - y1); edge_min = min(edge_min, edge, y0)
        if edge < EDGE - 1e-6:
            probs.append("%s 가 판 가장자리에서 %.1fmm" % (name, edge))
        if y0 < FRONT - 1e-6:
            probs.append("%s 가 앞쪽 띠에 걸침 (y %.1f)" % (name, y0))
        if s.distance(sbox(0, 0, EXCL[0], EXCL[1])) < 3.0:
            probs.append("%s 가 앞 왼쪽 금지 구역에서 %.1fmm" % (name, s.distance(sbox(0, 0, EXCL[0], EXCL[1]))))
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            g = shapes[i].distance(shapes[j]); gap_min = min(gap_min, g)
            if g < GAP:
                probs.append("%s 와 %s 사이 %.1fmm" % (items[i][0], items[j][0], g))
    return probs, (None if gap_min > 1e8 else round(gap_min, 1)), round(edge_min, 1)


def gcode_facts(text):
    """G코드에서 확인할 값들을 뽑는다"""
    head = text[:6000]
    g = lambda pat, cast=float: (cast(m.group(1)) if (m := re.search(pat, head)) else None)
    d = dict(layers=g(r"total layer number: (\d+)", int), max_z=g(r"max_z_height: ([\d.]+)"), weight_g=g(r"total filament weight \[g\] : ([\d.]+)"),
             length_mm=g(r"total filament length \[mm\] : ([\d.]+)"))
    m = re.search(r"model printing time: (.+?); total estimated time: (.+)", head)
    d["model_time"], d["total_time"] = (m.group(1).strip(), m.group(2).strip()) if m else (None, None)
    feats = {}
    for f in re.findall(r"^; FEATURE: (.+)$", text, re.M):
        feats[f.strip()] = feats.get(f.strip(), 0) + 1
    d["features"] = feats
    d["support_lines"] = sum(v for k, v in feats.items() if "upport" in k)
    # 노즐이 재료를 내보내며 지나간 범위 (시작할 때 긋는 시험 줄은 빼고, 모델 좌표로 바꿔서: Y + 2)
    body = text[text.index("; MACHINE_START_GCODE_END"):] if "; MACHINE_START_GCODE_END" in text else text
    ext = [1e9, 1e9, -1e9, -1e9, 0.0]; x = y = None; z = 0.0
    num = re.compile(r" ([XYZE])(-?\d*\.?\d+)")
    for line in body.split("\n"):
        if not line.startswith(("G1 ", "G0 ", "G2 ", "G3 ")):
            continue
        v = dict(num.findall(line.split(";", 1)[0]))
        nx = float(v["X"]) if "X" in v else x; ny = float(v["Y"]) if "Y" in v else y
        if "Z" in v:
            z = float(v["Z"])
        if "E" in v and float(v["E"]) > 0 and ("X" in v or "Y" in v) and x is not None and y is not None:
            for px, py in ((x, y), (nx, ny)):
                ext[0] = min(ext[0], px); ext[1] = min(ext[1], py + Y_OFF); ext[2] = max(ext[2], px); ext[3] = max(ext[3], py + Y_OFF)
            ext[4] = max(ext[4], z)
        x, y = nx, ny
    d["extent"] = [round(v, 2) for v in ext]
    t = lambda key: (m.group(1).strip() if (m := re.search(r"^; %s = (.+)$" % re.escape(key), text, re.M)) else None)
    d["settings"] = {k: t(k) for k in ("layer_height", "initial_layer_print_height", "wall_loops", "sparse_infill_density", "sparse_infill_pattern", "top_shell_layers", "bottom_shell_layers",
                                       "enable_support", "nozzle_temperature", "nozzle_temperature_initial_layer", "textured_plate_temp", "curr_bed_type", "printer_model", "filament_type",
                                       "brim_type", "printer_settings_id", "print_settings_id", "filament_settings_id", "filament_max_volumetric_speed", "seam_position")}
    return d


def slice_plate(name, infill, items, stl_dir, out, keep=True):
    tmp = tempfile.mkdtemp(prefix="bs_", dir=os.path.join(out, "_tmp"))
    files, shapes, heights, firsts = place(stl_dir, items, tmp)
    probs, gap_min, edge_min = layout_problems(items, shapes)
    proj = os.path.join(tmp, "p.3mf"); gc = os.path.join(tmp, "g.gcode.3mf")
    common = ["--load-settings", "%s;%s" % (MACHINE, PROC[infill]), "--load-filaments", FIL, "--filament-colour", COLOUR]
    # 1단계: STL -> 프로젝트 (자동 배치 끔, 자동 눕히기 끔: 적은 자리 그대로)
    rc1 = run(common + ["--orient", "0", "--arrange", "0", "--export-3mf", proj] + files, tmp, os.path.join(tmp, "1.log"))
    if rc1 != 0 or not os.path.exists(proj):
        return dict(name=name, ok=False, error="프로젝트 만들기 실패 (%d) %s" % (rc1, probs), log=open(os.path.join(tmp, "1.log")).read()[-800:])
    # 2단계: 프로젝트 -> 슬라이싱. 설정을 다시 주는 것은 프린터 종류 표시(BL-P001)가 파일에 들어가게 하려는 것
    rc2 = run(common + ["--slice", "0", "--min-save", "--export-3mf", gc, proj], tmp, os.path.join(tmp, "2.log"))
    if rc2 != 0 or not os.path.exists(gc):
        return dict(name=name, ok=False, error="슬라이싱 실패 (%d) %s" % (rc2, probs), log=open(os.path.join(tmp, "2.log")).read()[-800:])
    res = json.load(open(os.path.join(tmp, "result.json")))
    z = zipfile.ZipFile(gc)
    info = z.read("Metadata/slice_info.config").decode()
    sps = res.get("sliced_plates", [])
    if len(sps) != 1:
        probs.append("판이 %d개로 나뉨" % len(sps))
    sp = sps[0]; i = sp["id"]
    f = gcode_facts(z.read("Metadata/plate_%d.gcode" % i).decode("utf-8", "replace"))
    listed = re.findall(r'<object identify_id="(\d+)" name="([^"]*)"', info)
    warns = re.findall(r'<warning msg="([^"]*)" level="(\d+)" error_code ?="([^"]*)"', info)
    model_id = (re.search(r'key="printer_model_id" value="([^"]*)"', info) or [None, ""])[1]
    support_used = re.findall(r'key="support_used" value="([^"]*)"', info)
    # 적은 자리와 슬라이서가 실제로 놓은 자리(첫 층 범위)를 견준다
    pj = json.loads(z.read("Metadata/plate_%d.json" % i).decode())
    todo = list(range(len(items))); placed = []
    for o in pj.get("bbox_objects", []):
        nm = o["name"].replace(".stl", ""); x0, y0, x1, y1 = o["bbox"]
        cand = [k for k in todo if items[k][0] == nm]
        if not cand:
            probs.append("모르는 개체 %s" % nm); continue
        k = min(cand, key=lambda k: abs(items[k][1] - (x0 + x1) / 2) + abs(items[k][2] - (y0 + y1) / 2)); todo.remove(k)
        dev = max(abs(a - b) for a, b in zip((x0, y0, x1, y1), firsts[k]))
        if dev > 0.8:
            probs.append("%s 자리가 %.1fmm 어긋남" % (nm, dev))
        placed.append(dict(name=nm, bbox=[round(v, 2) for v in (x0, y0, x1, y1)], dev=round(dev, 2)))
    if todo:
        probs.append("찍히지 않은 개체: %s" % [items[k][0] for k in todo])
    if len(listed) != len(items):
        probs.append("개체 목록 %d개 (올린 것 %d개)" % (len(listed), len(items)))
    # G코드가 실제로 지나가는 범위가 올린 부품들의 범위와 맞는가, 높이가 맞는가
    ux0 = min(s_.bounds[0] for s_ in shapes); uy0 = min(s_.bounds[1] for s_ in shapes); ux1 = max(s_.bounds[2] for s_ in shapes); uy1 = max(s_.bounds[3] for s_ in shapes)
    ex0, ey0, ex1, ey1, ez = f["extent"]
    over = max(ux0 - ex0, uy0 - ey0, ex1 - ux1, ey1 - uy1); under = max(ex0 - ux0, ey0 - uy0, ux1 - ex1, uy1 - ey1)
    if over > 0.3 or under > 1.2:
        probs.append("G코드 범위 x %.1f~%.1f y %.1f~%.1f 가 부품 범위 x %.1f~%.1f y %.1f~%.1f 와 다름" % (ex0, ex1, ey0, ey1, ux0, ux1, uy0, uy1))
    if abs(ez - max(heights)) > 0.25:
        probs.append("G코드 높이 %.2f (모델 %.2f)" % (ez, max(heights)))
    if min(ex0, ey0, BED - ex1, BED - ey1) < 1.0:
        probs.append("판 밖으로 나감")
    if f["support_lines"] or any(s != "false" for s in support_used):
        probs.append("서포트가 생김")
    if sp.get("warning_message"):
        probs.append("경고: " + sp["warning_message"])
    for msg, level, code in warns:
        if msg != "bed_temperature_too_high_than_filament":          # PLA 를 닫힌 프린터에서 뽑을 때 늘 뜨는 알림 (앞문을 열어 두라는 뜻). 읽어보기에 적어 둔다
            probs.append("경고: %s" % msg)
    if model_id != "BL-P001":
        probs.append("프린터 종류 표시가 비어 있음")
    st = f["settings"]
    want = dict(layer_height="0.2", wall_loops="3", sparse_infill_density="%d%%" % infill, enable_support="0", curr_bed_type="Textured PEI Plate", printer_model="Bambu Lab X1 Carbon", filament_type="PLA")
    for k, v in want.items():
        if st.get(k) != v:
            probs.append("설정 %s = %s (바란 값 %s)" % (k, st.get(k), v))
    row = dict(name=name, ok=bool(res.get("return_code") == 0), infill=infill, items=[list(it) for it in items], sec=round(sp["total_predication"]), model_sec=round(sp["main_predication"]),
               g=round(sum(x["total_used_g"] for x in sp["filaments"]), 2), m=round((f["length_mm"] or 0) / 1000.0, 2), layers=f["layers"], max_z=f["max_z"],
               features=f["features"], settings=st, placed=placed, extent=f["extent"], gap_min=gap_min, edge_min=edge_min, notices=[w[0] for w in warns], printer_model_id=model_id, problems=probs)
    if keep:
        shutil.copy(proj, os.path.join(out, "프로젝트", name + ".3mf")); shutil.copy(gc, os.path.join(out, "바로출력", name + ".gcode.3mf"))
        open(os.path.join(out, "미리보기", name + ".png"), "wb").write(z.read("Metadata/plate_%d.png" % i))
        try:
            open(os.path.join(out, "미리보기", name + "_위에서.png"), "wb").write(z.read("Metadata/top_%d.png" % i))
        except KeyError:
            pass
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def hm(s):
    return "%d시간 %02d분" % (s // 3600, (s % 3600) // 60) if s >= 3600 else "%d분" % round(s / 60.0)


if __name__ == "__main__":
    stl_dir, out = sys.argv[1], sys.argv[2]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for d in ("프로젝트", "바로출력", "미리보기", "_tmp"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    path = os.path.join(out, "요약.json")
    summary = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else dict(plates={}, parts={})
    bad = 0
    for name in ORDER:
        desc, infill, items = PLATES[name]
        if (only and only not in name) or "--parts-only" in sys.argv:
            continue
        row = slice_plate(name, infill, items, stl_dir, out); row["desc"] = desc
        summary["plates"][name] = row
        if not row["ok"]:
            bad += 1; print("%-18s 실패: %s\n%s" % (name, row["error"], row.get("log", "")), flush=True); continue
        bad += bool(row["problems"])
        print("%-18s %s  %6.1f g  개체 %2d  층 %3d  부품 사이 %s  가장자리 %s  %s" % (name, hm(row["sec"]), row["g"], len(row["placed"]), row["layers"], row["gap_min"], row["edge_min"],
              ("<== " + "; ".join(row["problems"])) if row["problems"] else "문제 없음"), flush=True)
        json.dump(summary, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if "--parts" in sys.argv or "--parts-only" in sys.argv:
        for n in sorted(f[:-4] for f in os.listdir(stl_dir) if f.endswith(".stl")):
            if only and only not in n:
                continue
            infill = 100 if n.startswith(FULL) else 15
            row = slice_plate(n, infill, [(n, 128.0, 132.0, 0)], stl_dir, out, keep=False)
            if not row["ok"]:
                bad += 1; print("%-22s 실패: %s" % (n, row["error"]), flush=True); continue
            bad += bool(row["problems"])
            summary["parts"][n] = dict(sec=row["sec"], model_sec=row["model_sec"], g=row["g"], m=row["m"], infill=infill, layers=row["layers"], features=row["features"], problems=row["problems"])
            print("%-22s %s (준비 빼고 %s)  %6.1f g  %s" % (n, hm(row["sec"]), hm(row["model_sec"]), row["g"], ("<== " + "; ".join(row["problems"])) if row["problems"] else ""), flush=True)
            json.dump(summary, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(out, "요약.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["판", "올리는 것", "채움 %", "출력 시간", "시간 (초)", "필라멘트 g", "필라멘트 m", "층 수", "서포트", "확인할 것"])
        for name in ORDER:
            row = summary["plates"].get(name)
            if row and row.get("ok"):
                w.writerow([name, row["desc"], row["infill"], hm(row["sec"]), row["sec"], row["g"], row["m"], row["layers"], "없음", "; ".join(row["problems"])])
    shutil.rmtree(os.path.join(out, "_tmp"), ignore_errors=True)
    print("확인할 것이 있는 판/부품: %d" % bad)
