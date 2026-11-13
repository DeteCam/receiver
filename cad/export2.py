"""출력 방향으로 눕혀 STL 로 내보내고, 저장한 파일을 다시 읽어 검사한다.  python3 export2.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import trimesh
import parts2 as P
from cadlib import *
from p2 import *

import paths
ROOT = paths.RES
OUT = paths.STL

# (파일 이름, 형상 함수, 출력 방향 회전, 수량, 한글 이름, 고르는 기준, 채움)
SPEC = [
    ("01_base_tub_S",    lambda: P.tub("S"),      [0, 0, 0],   1, "받침 통 (작은 것, 지름 140)", "배터리 106×71×18 이하", 15),
    ("01_base_tub_L",    lambda: P.tub("L"),      [0, 0, 0],   1, "받침 통 (큰 것, 지름 190)", "배터리 152×76×18 이하", 15),
    ("02_base_cover_S",  lambda: P.cover("S"),    [0, 0, 0],   1, "받침 덮개 (작은 것)", "받침 통과 같은 글자", 15),
    ("02_base_cover_L",  lambda: P.cover("L"),    [0, 0, 0],   1, "받침 덮개 (큰 것)", "받침 통과 같은 글자", 15),
    ("03_clip",          P.clip,                  [0, 0, 0],   1, "회전판 고리", "", 15),
    ("04_yoke",          P.yoke,                  [0, 0, 0],   1, "요크 (회전판 + 팔 + 전선 상자)", "", 15),
]
for i in (1, 2, 3):
    d, f, r = FITS[i]
    SPEC.append(("05_pinion_fit%d" % i, (lambda d=d, f=f: P.pinion(0.0, d, f)), [0, 0, 0], 2, "작은 기어 10T (맞춤 %d)" % i,
                 "셋 다 2개씩 뽑아서 모터 축에 맞는 것 2개를 쓴다", 15))
SPEC += [
    ("06_sector_gear",   P.sector_gear,           [90, 0, 0],  1, "부채꼴 기어", "", 15),
    ("07_head_shell",    P.head_shell,            [-90, 0, 0], 1, "머리 몸통", "", 15),
    ("08_head_lid",      P.head_lid,              [90, 0, 0],  1, "머리 뚜껑", "", 15),
    ("09_crown",         P.crown,                 [0, 0, 0],   1, "모터 자리", "", 15),
]
for i in (1, 2, 3):
    d, f, r = FITS[i]
    SPEC.append(("10_rotor_bar_fit%d" % i, (lambda d=d, f=f: P.rotor_bar(d, f)), [180, 0, 0], 1, "회전 막대 + 균형추 (맞춤 %d)" % i,
                 "시험 조각 %d번 구멍이 맞을 때. 채움 100%%" % i, 100))
SPEC += [
    ("11_blade_100x150", lambda: P.blade(150.0),  [0, 90, 0],  1, "판 100×150", "회전 막대와 같은 판에서 채움 100%로", 100),
    ("12_blade_100x200", lambda: P.blade(200.0),  [0, 90, 0],  1, "판 100×200 (비교용)", "15번 덩어리와 같이 쓴다. 채움 100%", 100),
]
for i in (1, 2, 3):
    SPEC.append(("13_pin_fit%d" % i, (lambda r=PIN_FITS[i]: P.pin(1, r)), [-90, 0, 0], 2, "끄덕임 축 핀 (지름 %.1f)" % (2 * PIN_FITS[i]),
                 "셋 다 2개씩 뽑아서 맞는 것 2개를 쓴다", 15))
SPEC += [
    ("14_trim_plate",    P.trim_plate,            [0, 90, 0],  2, "균형 조각", "회전부 균형이 안 맞을 때", 100),
    ("15_balance_200",   P.balance_200,           [0, 90, 0],  1, "판 200용 균형 덩어리", "판 100×200을 쓸 때. 채움 100%", 100),
    ("16_m2_ring",       P.m2_ring,               [90, 0, 0],  1, "모터 2 맞춤 링", "모터 2의 축 둘레 턱에 씌운 다음 작은 기어를 끼운다", 15),
    ("17_lan_cap",       P.lan_cap,               [0, 0, 0],   1, "랜선 홈 마개", "랜선을 받침 안으로 넣지 않을 때, 덮개를 덮기 전에 받침 뒤 홈에 끼운다", 15),
    ("90_test_coupon",   P.test_coupon,           [0, 0, 0],   1, "맞춤 시험 조각", "", 15),
    ("91_tenon_gauge",   P.tenon_gauge,           [0, 0, 0],   1, "시험용 돌기", "", 15),
]
DEFAULT = ("01_base_tub_S", "02_base_cover_S", "05_pinion_fit2", "10_rotor_bar_fit2", "13_pin_fit2")


def oriented(fn, rot):
    m = fn().rotate(rot)
    bb = m.bounding_box()
    return m.translate([-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]])


def clean32(m, digits=3):
    """좌표를 0.001mm 단위로 맞춰 거의 겹친 꼭짓점을 합치고, 선이나 점이 되어 버린 삼각형과 서로 맞붙어 접힌 삼각형 쌍을 지운다.
    이렇게 해야 어떤 프로그램이 파일을 읽어도 모든 모서리가 정확히 두 면에 닿는 닫힌 형상이 된다."""
    mesh = m.to_mesh()
    v = np.round(np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), digits)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    removed = 0
    for _ in range(4):
        uv, inv = np.unique(v, axis=0, return_inverse=True)
        f = inv.reshape(-1)[f]; v = uv
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        fs = np.sort(f, axis=1)                                   # 같은 세 점을 쓰는 삼각형이 둘이면(접힌 쌍) 둘 다 지운다
        _, idx, cnt = np.unique(fs, axis=0, return_inverse=True, return_counts=True)
        keep &= cnt[idx.reshape(-1)] == 1
        removed += int((~keep).sum())
        if keep.all():
            break
        f = f[keep]
    return trimesh.Trimesh(v, f, process=False), removed


def closed(tm):
    e = np.sort(tm.edges, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2)) and tm.is_winding_consistent


def overhang_report(tm, ang_deg=50.0):
    """바닥에 닿지 않으면서 아래를 향한 면의 넓이(mm2)"""
    n = tm.face_normals; a = tm.area_faces
    zc = tm.triangles_center[:, 2]
    down = -n[:, 2]
    mask = (down > np.sin(np.radians(ang_deg))) & (zc > 0.35)
    flat = (down > 0.985) & (zc > 0.35)
    return float(a[mask].sum()), float(a[flat].sum())


if __name__ == "__main__":
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    rows = []
    for name, fn, rot, qty, ko, pick, infill in SPEC:
        m = oriented(fn, rot)
        tm, nrem = clean32(m)
        bb = m.bounding_box()
        size = (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        oh, flat = overhang_report(tm)
        path = os.path.join(OUT, name + '.stl')
        tm.export(path)
        back = trimesh.load(path, process=True)
        rough = trimesh.load(path, process=False); rough.merge_vertices(digits_vertex=3)
        ok = (closed(back) and closed(rough) and back.is_watertight and len(back.faces) == len(tm.faces) and len(m.decompose()) == 1
              and max(size[:2]) <= 250 and size[2] <= 250 and abs(back.volume - m.volume()) < max(2.0, 1e-4 * m.volume()))
        rows.append(dict(file=name + '.stl', ko=ko, qty=qty, pick=pick, infill=infill, size=[round(s, 1) for s in size],
                         volume_cm3=round(m.volume() / 1000, 1), watertight=bool(back.is_watertight), faces=int(len(back.faces)),
                         bodies=len(m.decompose()), overhang_mm2=round(oh), flat_overhang_mm2=round(flat), ok=bool(ok)))
        print("%-20s %-26s x%d %6.1f x %6.1f x %6.1f %6.1f cm3 닫힘=%s 덩어리=%d 면 %5d 처짐면 %4d/%4d %s" % (
            name, ko, qty, size[0], size[1], size[2], m.volume() / 1000, closed(back) and back.is_watertight, len(m.decompose()),
            len(back.faces), oh, flat, "" if ok else "  <-- 확인 필요"))
    json.dump(rows, open(ROOT + '/parts.json', 'w'), ensure_ascii=False, indent=1)
    print("파일 %d개, 문제 %d개" % (len(rows), sum(not r["ok"] for r in rows)))
