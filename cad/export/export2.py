"""출력 방향으로 눕혀 STL 로 내보내고, 저장한 파일을 다시 읽어 검사한다.  python3 export2.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import trimesh
import panel as P
from cadlib import *

import paths
ROOT = paths.RES
OUT = paths.STL

# (파일 이름, 형상 함수, 출력 방향 회전, 수량, 한글 이름, 고르는 기준, 채움)
SPEC = [
    ("01_panel_tray",   P.tray,        [0, 0, 0],   2, "패널 통", "패널 하나에 1개. 뒷면을 바닥에, 받침 없이", 15),
    ("02_panel_bezel",  P.bezel,       [180, 0, 0], 2, "패널 앞덮개", "패널 하나에 1개. 겉면을 바닥에, 받침 없이", 15),
    ("03_mount_plate",  P.mount_plate, [180, 0, 0], 2, "자석 걸이판", "칠판·철문·사물함에 붙이거나 걸이에 거는 패널에. 통에 닿는 면을 바닥에", 15),
    ("04_foot",         P.foot,        [0, 90, 0],  1, "받침 발", "창턱·교탁·사물함 위에 세우거나 삼각대에 올리는 패널에 (걸이판 대신). 옆으로 눕혀서", 15),
    ("05_xiao_clip",    P.xiao_clip,   [180, 0, 0], 2, "ESP32 누름쇠", "패널 하나에 1개. 납작한 면을 바닥에", 15),
    ("06_hub_box",      P.hub_box,     [0, 0, 0],   1, "허브 통", "", 15),
    ("07_hub_deck",     P.hub_deck,    [0, 0, 0],   1, "허브 선반", "배터리 위에 얹고 파이를 올린다", 15),
    ("08_hub_lid",      P.hub_lid,     [180, 0, 0], 1, "허브 뚜껑", "겉면을 바닥에", 15),
    ("09_ant_plate",    P.ant_plate,   [180, 0, 0], 2, "안테나 받침판", "패널 하나에 1개. 앞면을 바닥에 (다리가 위로), 받침 없이", 15),
    ("10_turn_base",    P.turn_base,   [0, 0, 0],   1, "회전 받침 받침대", "숙소용. 밑면을 바닥에 (모터 칸이 위로)", 15),
    ("11_turn_top",     P.turn_top,    [0, 0, 0],   1, "회전 받침 돌림판", "숙소용. 밑면을 바닥에 (기운 받침이 위로)", 15),
    ("12_turn_hub",     P.turn_hub,    [0, 0, 0],   1, "회전 받침 축 이음쇠", "숙소용. 모터 축 ↔ 돌림판. 세워서", 15),
]
DEFAULT = tuple(n for n, *_ in SPEC)


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
