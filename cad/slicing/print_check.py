"""출력 검증: 슬라이싱한 파일(.gcode.3mf)을 다시 읽어서, 프린터에 넣었을 때 문제없이 뽑힐지 본다.

    python3 print_check.py <STL 폴더> <슬라이스 결과 폴더> [--only 이름조각]      ->  결과 폴더의 print_check.json

Bambu Studio 가 만든 G코드를 한 줄씩 읽어 '실제로 노즐이 지나가며 찍는 자리'를 층마다 0.2mm 칸 그림으로 만들고, 모델(STL)의 단면과 견준다.
  1) 파일   프린터가 읽는 것이 다 들어 있는지, G코드가 깨지지 않았는지(md5), 기종·노즐·필라멘트·판 종류
  2) 범위   찍는 자리가 판(256 x 256) 안, 앞 왼쪽 금지 구역(18 x 28) 밖, 높이 250 안인지
  3) 빠짐   모델 가장자리 가운데 G코드가 찍지 않는 곳 (너무 얇아서 슬라이서가 버린 곳)
  4) 허공   아래층이 받쳐 주지 않는 곳. 양 끝이 받쳐진 '다리'인지 한쪽만 붙은 '외팔'인지, 얼마나 긴지. 받침(서포트) 없이 뽑으므로 가장 중요하다
  5) 바닥   첫 층이 판에 닿는 넓이가 작은데 키가 큰 것 (뽑다가 넘어질 수 있다)
  6) 속도   녹여 내보내는 양(mm3/s)이 필라멘트 한계를 넘지 않는지
  7) 경고   Bambu Studio 가 파일에 적어 둔 경고
실물 프린터에서만 알 수 있는 것(첫 층이 잘 붙는지, 필라멘트가 젖었는지, 노즐 막힘)은 여기서 볼 수 없다.
"""
import hashlib, json, os, re, sys, zipfile, time
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import manifold3d as m3

RES = 0.2                       # 칸 크기 mm
BED, ZMAX = 256.0, 250.0
EXCL = (18.0, 28.0)
Y_OFF = 2.0                     # X1 Carbon: G코드의 Y = 모델 자리 - 2mm
SUP = 0.4                       # 한 층(0.2mm)에 이만큼까지 밖으로 나가는 것은 '받쳐졌다'고 본다 (기울기 63도)
THIN = 0.5                      # 모델 가장자리에서 이 두께의 띠를 본다
FIL_AREA = np.pi * (1.75 / 2) ** 2


def parse(gcode):
    """G코드 -> 층 목록. 층 = dict(z, h, seg=[(x0, y0, x1, y1, 폭, 종류 번호, 물체 번호, 속도 mm/s, E)])"""
    layers = []; cur = None; x = y = 0.0; f = 0.0; feat = "?"; width = 0.42; obj = 0; wipe = False; feats = []
    fid = {}
    num = re.compile(r"([XYZEFIJ])(-?\d*\.?\d+)")
    started = False
    for line in gcode.split("\n"):
        if not line:
            continue
        if line[0] == ";":
            if line.startswith("; CHANGE_LAYER"):
                started = True; cur = dict(z=None, h=0.2, seg=[]); layers.append(cur)
            elif line.startswith("; Z_HEIGHT:") and cur is not None and cur["z"] is None:
                cur["z"] = float(line.split(":")[1])
            elif line.startswith("; LAYER_HEIGHT:") and cur is not None:
                cur["h"] = float(line.split(":")[1])
            elif line.startswith("; FEATURE:"):
                feat = line.split(":", 1)[1].strip()
                if feat not in fid:
                    fid[feat] = len(fid); feats.append(feat)
            elif line.startswith("; LINE_WIDTH:"):
                width = float(line.split(":")[1])
            elif line.startswith("; start printing object"):
                obj = int(line.rsplit(":", 1)[1])
            elif line.startswith("; stop printing object"):
                obj = 0
            elif line.startswith("; WIPE_START"):
                wipe = True
            elif line.startswith("; WIPE_END"):
                wipe = False
            elif line.startswith("; MACHINE_END_GCODE_START"):
                break
            continue
        c = line[:3]
        if c not in ("G1 ", "G2 ", "G3 ", "G0 "):
            continue
        v = dict((k, float(val)) for k, val in num.findall(line.split(";")[0]))
        if "F" in v:
            f = v["F"] / 60.0
        nx, ny = v.get("X", x), v.get("Y", y)
        e = v.get("E", 0.0)
        if started and cur is not None and e > 0 and not wipe and (nx != x or ny != y):
            k = fid.get(feat, -1)
            if c == "G1 " or c == "G0 ":
                cur["seg"].append((x, y, nx, ny, width, k, obj, f, e))
            else:                                   # 원호: 잘게 나눈다
                cx_, cy_ = x + v.get("I", 0.0), y + v.get("J", 0.0); r = np.hypot(x - cx_, y - cy_)
                a0 = np.arctan2(y - cy_, x - cx_); a1 = np.arctan2(ny - cy_, nx - cx_)
                if c == "G2 ":
                    if a1 >= a0: a1 -= 2 * np.pi
                else:
                    if a1 <= a0: a1 += 2 * np.pi
                n = max(2, int(abs(a1 - a0) * r / 0.4) + 1)
                aa = np.linspace(a0, a1, n + 1); px, py = cx_ + r * np.cos(aa), cy_ + r * np.sin(aa)
                for i in range(n):
                    cur["seg"].append((px[i], py[i], px[i + 1], py[i + 1], width, k, obj, f, e / n))
        x, y = nx, ny
    return [l for l in layers if l["z"] is not None], feats


def load_parts(stl_dir, items):
    """판에 놓인 부품들: (이름, manifold(판 좌표, G코드 기준), 높이)"""
    out = []
    for name, cx_, cy_, rot in items:
        m = trimesh.load(os.path.join(stl_dir, name + ".stl"))
        if rot:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot), [0, 0, 1]))
        b = m.bounds
        m.apply_translation([cx_ - (b[0, 0] + b[1, 0]) / 2, cy_ - Y_OFF - (b[0, 1] + b[1, 1]) / 2, -b[0, 2]])
        mesh = m3.Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32), tri_verts=np.asarray(m.faces, dtype=np.uint32))
        man = m3.Manifold(mesh)
        if man.is_empty():                           # STL 의 꼭짓점이 따로따로일 때: 합쳐서 다시
            m.merge_vertices(); man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32), tri_verts=np.asarray(m.faces, dtype=np.uint32)))
        out.append((name, man, float(m.bounds[1, 2]), m.bounds.copy()))
    return out


def hits_rect(S, x1, y1):
    """선분이 네모 [0, x1] x [0, y1] 안을 지나는가 (선분마다 참/거짓). 두 끝이 네모 밖이어도 가로질러 가면 참"""
    t0 = np.zeros(len(S)); t1 = np.ones(len(S))
    for p0, d, lim in ((S[:, 0], S[:, 2] - S[:, 0], x1), (S[:, 1], S[:, 3] - S[:, 1], y1)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (lim - p0) / d
        t1 = np.where(d > 0, np.minimum(t1, t), t1); t0 = np.where(d < 0, np.maximum(t0, t), t0)
        t0 = np.where((d == 0) & (p0 > lim), 2.0, t0)
    return t0 <= t1


class Raster(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0; self.w = int(np.ceil((x1 - x0) / RES)) + 1; self.h = int(np.ceil((y1 - y0) / RES)) + 1

    def px(self, x, y):
        return (x - self.x0) / RES, (y - self.y0) / RES

    def lines(self, segs, sel=None):
        im = Image.new("L", (self.w, self.h), 0); d = ImageDraw.Draw(im)
        for s in segs:
            if sel is not None and not sel(s):
                continue
            a = self.px(s[0], s[1]); b = self.px(s[2], s[3]); wpx = max(1, int(np.ceil(s[4] / RES - 0.05)))
            d.line([a, b], fill=1, width=wpx)
            if wpx > 1:
                rr = wpx / 2.0
                d.ellipse([b[0] - rr, b[1] - rr, b[0] + rr, b[1] + rr], fill=1)
        return np.asarray(im, dtype=bool)

    def polys(self, polygons):
        acc = np.zeros((self.h, self.w), dtype=bool)
        for pg in polygons:
            p = np.asarray(pg, dtype=float)
            if len(p) < 3:
                continue
            im = Image.new("L", (self.w, self.h), 0)
            ImageDraw.Draw(im).polygon([self.px(a, b) for a, b in p], fill=1)
            acc ^= np.asarray(im, dtype=bool)                       # 짝홀 규칙: 구멍은 지워지고 구멍 속 섬은 다시 채워진다
        return acc


def disk(r_mm):
    r = int(round(r_mm / RES)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def check_plate(path, stl_dir, items, quick=False):
    t0 = time.time(); z = zipfile.ZipFile(path); names = z.namelist(); res = dict(file=os.path.basename(path), problems=[], notes=[])
    # ---- 1) 파일
    need = ["Metadata/plate_1.gcode", "Metadata/plate_1.gcode.md5", "Metadata/slice_info.config", "Metadata/project_settings.config", "Metadata/model_settings.config", "3D/3dmodel.model", "[Content_Types].xml", "Metadata/plate_1.png"]
    miss = [n for n in need if n not in names]
    raw = z.read("Metadata/plate_1.gcode")
    md5_ok = hashlib.md5(raw).hexdigest().lower() == z.read("Metadata/plate_1.gcode.md5").decode().strip().lower()
    info = z.read("Metadata/slice_info.config").decode("utf-8")
    meta = dict(re.findall(r'<metadata key="([^"]+)" value="([^"]*)"', info))
    objs = {int(a): b for a, b in re.findall(r'<object identify_id="(\d+)" name="([^"]+)"', info)}
    warn = re.findall(r'<warning msg="([^"]+)"', info)
    fil = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<filament [^>]*>", info).group(0)))
    g = raw.decode("utf-8", "replace")
    cfg = dict(re.findall(r"^; (\w+) = (.*)$", g, re.M))
    head = g[:3000]
    res["file_check"] = dict(missing=miss, md5_ok=bool(md5_ok), printer_model_id=meta.get("printer_model_id"), nozzle=meta.get("nozzle_diameters"), support_used=meta.get("support_used"),
                             filament_type=fil.get("type"), used_g=float(fil.get("used_g", 0)), used_m=float(fil.get("used_m", 0)), prediction_s=int(meta.get("prediction", 0)),
                             printer=cfg.get("printer_model"), bed_type=cfg.get("curr_bed_type"), nozzle_temp=cfg.get("nozzle_temperature"), bed_temp=cfg.get("textured_plate_temp"),
                             layer_height=cfg.get("layer_height"), enable_support=cfg.get("enable_support"), brim=cfg.get("brim_type"), max_flow_setting=cfg.get("filament_max_volumetric_speed"),
                             gcode_bytes=len(raw), warnings=warn)
    if miss: res["problems"].append("파일에 빠진 것: %s" % miss)
    if not md5_ok: res["problems"].append("G코드의 md5 가 맞지 않는다")
    if meta.get("printer_model_id") != "BL-P001": res["problems"].append("프린터 기종이 X1 Carbon(BL-P001)이 아니다: %s" % meta.get("printer_model_id"))
    if meta.get("nozzle_diameters") != "0.4": res["problems"].append("노즐이 0.4 가 아니다")
    m = re.search(r"; total layer number: (\d+)", head); n_layers = int(m.group(1)) if m else None
    m = re.search(r"; max_z_height: ([\d.]+)", head); max_z = float(m.group(1)) if m else None
    # ---- G코드 읽기
    layers, feats = parse(g)
    if n_layers is not None and len(layers) != n_layers:
        res["problems"].append("층 수가 머리말(%d)과 다르다: %d" % (n_layers, len(layers)))
    S = np.array([s for l in layers for s in l["seg"]], dtype=float)
    xs = np.concatenate([S[:, 0], S[:, 2]]); ys = np.concatenate([S[:, 1], S[:, 3]])
    ext = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    # ---- 2) 범위
    in_excl = bool(hits_rect(S, EXCL[0], EXCL[1]).any())
    res["range"] = dict(x=[round(ext[0], 1), round(ext[2], 1)], y=[round(ext[1], 1), round(ext[3], 1)], z_max=max_z, edge_min=round(min(ext[0], ext[1], BED - ext[2], BED - ext[3]), 1), in_exclude=in_excl, layers=len(layers))
    if ext[0] < 0.5 or ext[1] < 0.5 or ext[2] > BED - 0.5 or ext[3] > BED - 0.5: res["problems"].append("판 밖으로 나간다")
    if in_excl: res["problems"].append("앞 왼쪽 금지 구역에 찍는다")
    if max_z and max_z > ZMAX: res["problems"].append("높이 250mm 를 넘는다")
    # ---- 6) 속도 (녹여 내보내는 양)
    seg_len = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]); ok = (seg_len > 0.3) & (S[:, 7] > 0)
    flow = np.zeros(len(S)); flow[ok] = S[ok, 8] * FIL_AREA / (seg_len[ok] / S[ok, 7])
    fl = {}
    for k, name in enumerate(feats):
        sel = ok & (S[:, 5] == k)
        if sel.any():
            fl[name] = dict(flow_p99=round(float(np.percentile(flow[sel], 99)), 1), speed_max=round(float(S[sel, 7].max()), 0), len_m=round(float(seg_len[sel].sum() / 1000), 1))
    res["flow"] = dict(max_p999=round(float(np.percentile(flow[ok], 99.9)), 1), by_feature=fl)
    lim = float(cfg.get("filament_max_volumetric_speed", "12").split(",")[0])
    if res["flow"]["max_p999"] > lim * 1.1: res["problems"].append("흘림 양이 한계(%.0f)를 넘는다: %.1f" % (lim, res["flow"]["max_p999"]))
    res["features_m"] = {k: v["len_m"] for k, v in fl.items()}
    if quick:
        return res
    # ---- 모델을 판 위에 놓는다
    parts = load_parts(stl_dir, items)
    exp_z = max(p[2] for p in parts)
    if max_z is not None and abs(max_z - exp_z) > 0.25:
        res["problems"].append("가장 높은 곳이 모델(%.1f)과 다르다: %.1f" % (exp_z, max_z))
    R = Raster(ext[0] - 3, ext[1] - 3, ext[2] + 3, ext[3] + 3)
    d_sup = disk(SUP); d_thin = disk(THIN); d_cov = disk(0.3)
    bridge_id = feats.index("Bridge") if "Bridge" in feats else -99
    # 물체 번호 -> 부품 번호 (자리로 맞춘다)
    lab = np.zeros((R.h, R.w), dtype=np.int16)
    for i, (name, man, hgt, b) in enumerate(parts):
        a = R.px(b[0, 0], b[0, 1]); c = R.px(b[1, 0], b[1, 1])
        lab[int(a[1]):int(c[1]) + 1, int(a[0]):int(c[0]) + 1] = i + 1
    prev = None; prev_miss = None; miss_tot = np.zeros(len(parts)); band_tot = np.zeros(len(parts)); un = []; miss_spots = []
    first = None; air_non_bridge = 0.0
    for li, L in enumerate(layers):
        zmid = L["z"] - L["h"] / 2.0 - 0.013          # 층 한가운데에서 살짝 아래: 모델의 수평면이 딱 층 한가운데에 걸리는 일을 피한다
        polys = []
        for name, man, hgt, b in parts:
            if zmid < hgt:
                polys += list(man.slice(float(zmid)).to_polygons())
        Mz = R.polys(polys)
        G = R.lines(L["seg"])
        if li == 0:
            first = (Mz.copy(), G.copy())
        # 3) 빠짐: 모델 가장자리 띠 가운데 G코드가 0.3mm 안에 없는 칸
        band = Mz & ~ndi.binary_erosion(Mz, d_thin)
        missing = band & ~ndi.binary_dilation(G, d_cov)
        missing = ndi.binary_opening(missing, np.ones((2, 2), dtype=bool))      # 한 칸 폭의 줄(첫 층을 조금 줄여 찍는 것, 칸 나눔의 오차)은 뺀다
        both = missing & ndi.binary_dilation(prev_miss, d_cov) if prev_miss is not None else np.zeros_like(missing)      # 두 층 이어서 빠진 곳만 센다 (한 층짜리는 층 나눔의 차이)
        prev_miss = missing
        missing = both
        if missing.any():
            lb, n = ndi.label(missing)
            for k in range(1, n + 1):
                a = (lb == k); area = a.sum() * RES * RES
                if area >= 0.6:
                    yy, xx = np.nonzero(a); pi_ = int(lab[int(yy.mean()), int(xx.mean())]) - 1
                    miss_spots.append(dict(z=round(L["z"], 2), area=round(float(area), 2), x=round(R.x0 + xx.mean() * RES, 1), y=round(R.y0 + yy.mean() * RES, 1), part=parts[pi_][0] if pi_ >= 0 else "?"))
        for i in range(len(parts)):
            sel = lab == i + 1
            miss_tot[i] += (missing & sel).sum() * RES * RES; band_tot[i] += (band & sel).sum() * RES * RES
        # 4) 허공: 이 층의 모델 가운데 아래층(조금 넓힌 것)이 받쳐 주지 않는 칸
        if prev is not None:
            sup = ndi.binary_dilation(prev, d_sup)
            U = Mz & ~sup
            if U.any():
                held = Mz & sup
                dist = ndi.distance_transform_edt(~held) * RES                         # 받쳐진 칸까지의 거리
                Gb = R.lines(L["seg"], lambda s: s[5] == bridge_id)
                air_non_bridge += float((G & U & ~ndi.binary_dilation(Gb, d_cov)).sum() * RES * RES)
                lb, n = ndi.label(U)
                for k in range(1, n + 1):
                    a = (lb == k); area = a.sum() * RES * RES
                    if area < 1.0:
                        continue
                    reach = float(dist[a].max()); yy, xx = np.nonzero(a)
                    if reach < 1.0:
                        continue
                    # 다리인가: 받쳐진 칸이 이 덩어리의 양쪽에 있는가 (덩어리를 조금 넓혀 닿는 받침 칸들의 퍼진 각으로 본다)
                    ring = ndi.binary_dilation(a, disk(0.6)) & held
                    kind = "섬"
                    if ring.any():
                        ry, rx = np.nonzero(ring); cy_, cx_ = yy.mean(), xx.mean()
                        ang = np.sort(np.arctan2(ry - cy_, rx - cx_)); gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
                        kind = "다리" if gaps.max() < np.radians(200.0) else "외팔"
                    bcov = float((Gb & a).sum()) / max(float((G & a).sum()), 1.0)
                    pi_ = int(lab[int(yy.mean()), int(xx.mean())]) - 1
                    un.append(dict(z=round(L["z"], 2), kind=kind, reach=round(reach, 1), area=round(float(area), 1), w=round(float((xx.max() - xx.min() + 1) * RES), 1), h=round(float((yy.max() - yy.min() + 1) * RES), 1),
                                   x=round(R.x0 + xx.mean() * RES, 1), y=round(R.y0 + yy.mean() * RES, 1), part=parts[pi_][0] if pi_ >= 0 else "?", bridge_share=round(bcov, 2)))
        prev = Mz
    # ---- 5) 바닥
    M1, G1 = first; lb, n = ndi.label(ndi.binary_dilation(G1, disk(0.3)))
    foot = []
    for i, (name, man, hgt, b) in enumerate(parts):
        sel = (lab == i + 1) & G1
        area = float(sel.sum() * RES * RES)
        if area <= 0:
            continue
        ids = np.unique(lb[sel]); ids = ids[ids > 0]
        small = min(float((lb == k).sum() * RES * RES) for k in ids) if len(ids) else 0.0
        foot.append(dict(part=name, contact_mm2=round(area, 0), islands=int(len(ids)), smallest_island_mm2=round(small, 0), height=round(hgt, 1), slender=round(hgt / max(np.sqrt(area), 1e-6), 2)))
    res["bed"] = foot
    for f_ in foot:
        if f_["slender"] > 2.5 and f_["contact_mm2"] < 150:
            res["notes"].append("%s: 바닥에 닿는 넓이 %.0fmm2 에 키 %.0fmm (가늘고 높다)" % (f_["part"], f_["contact_mm2"], f_["height"]))
    # ---- 정리
    res["missing"] = dict(by_part={parts[i][0]: dict(mm2=round(float(miss_tot[i]), 1), share_pct=round(100.0 * miss_tot[i] / max(band_tot[i], 1e-9), 2)) for i in range(len(parts))},
                          spots=sorted(miss_spots, key=lambda d: -d["area"])[:12], n_spots=len(miss_spots))
    un.sort(key=lambda d: -d["reach"])
    res["unsupported"] = dict(n=len(un), islands=[u for u in un if u["kind"] == "섬"][:10], cantilever=[u for u in un if u["kind"] == "외팔"][:12], bridge=[u for u in un if u["kind"] == "다리"][:12],
                              max_bridge_span=round(max([2 * u["reach"] for u in un if u["kind"] == "다리"] or [0.0]), 1), max_cantilever=round(max([u["reach"] for u in un if u["kind"] == "외팔"] or [0.0]), 1),
                              n_island=sum(u["kind"] == "섬" for u in un), n_cantilever=sum(u["kind"] == "외팔" for u in un), n_bridge=sum(u["kind"] == "다리" for u in un),
                              non_bridge_in_air_mm2=round(air_non_bridge, 1))
    res["sec"] = round(time.time() - t0, 0)
    return res


if __name__ == "__main__":
    stl_dir, out = sys.argv[1], sys.argv[2]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    summ = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    path = os.path.join(out, "print_check.json")
    allr = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for name, pl in summ["plates"].items():
        if only and only not in name:
            continue
        r = check_plate(os.path.join(out, "gcode", name + ".gcode.3mf"), stl_dir, [tuple(i) for i in pl["items"]])
        allr[name] = r
        u = r["unsupported"]; fc = r["file_check"]
        print("== %s  (%d초)  문제 %d개 %s" % (name, r["sec"], len(r["problems"]), r["problems"]))
        print("   파일: md5 %s, 기종 %s, 노즐 %s, %s, 판 %s %s도, 노즐 %s도, 받침 %s | %d층, 높이 %.1f, 가장자리까지 %.1fmm, 금지 구역 %s" % (
            "맞음" if fc["md5_ok"] else "틀림", fc["printer_model_id"], fc["nozzle"], fc["filament_type"], fc["bed_type"], fc["bed_temp"], fc["nozzle_temp"], fc["enable_support"],
            r["range"]["layers"], r["range"]["z_max"], r["range"]["edge_min"], "침범" if r["range"]["in_exclude"] else "밖"))
        print("   빠진 곳: %d군데 %s" % (r["missing"]["n_spots"], ", ".join("%s %.1fmm2 (z %.1f)" % (s["part"], s["area"], s["z"]) for s in r["missing"]["spots"][:5])))
        print("   허공: 섬 %d, 외팔 %d (가장 긴 것 %.1fmm), 다리 %d (가장 긴 것 %.1fmm). 다리가 아닌데 허공에 찍는 넓이 %.0fmm2" % (u["n_island"], u["n_cantilever"], u["max_cantilever"], u["n_bridge"], u["max_bridge_span"], u["non_bridge_in_air_mm2"]))
        for k in ("islands", "cantilever", "bridge"):
            for s in u[k][:4]:
                print("      %s %-18s z %6.2f  길이 %4.1f  넓이 %5.1f (%.1f x %.1f)  자리 (%.0f, %.0f)  다리로 찍는 비율 %.0f%%" % (s["kind"], s["part"], s["z"], s["reach"] * (2 if s["kind"] == "다리" else 1), s["area"], s["w"], s["h"], s["x"], s["y"], 100 * s["bridge_share"]))
        print("   흘림 양 최대 %.1f mm3/s (한계 %s) | 경고: %s | 메모: %s" % (r["flow"]["max_p999"], fc["max_flow_setting"], fc["warnings"], r["notes"]), flush=True)
        json.dump(allr, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
