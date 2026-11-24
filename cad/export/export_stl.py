"""DeteCam 6.0 (Nano-2 AI) 6종 출력 부품 STL 내보내기 + 수밀성/크기/오버행 자동 검사."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "checks")]
import numpy as np
import trimesh
import nano2_node as P
import paths

OUT = paths.STL
RES = paths.RES

# (파일 이름, 형상 함수, 출력 방향 회전 [rx, ry, rz], 노드당 수량, 한글 이름, 설명, 채움 %)
SPEC = [
    ("01_rf_shell",    P.rf_shell,    [180, 0, 0], 1, "전면 RF 레이돔 쉘",     "96x96 포켓 전면 레이돔, 8.6mm RF 공극, 서포트 0", 15),
    ("02_sbc_chassis", P.sbc_chassis, [180, 0, 0], 1, "중앙 NanoPi·차폐 섀시",  "Ni-Cu 격벽 바닥 배치, 무개조 NanoPi R5C + AX210 베이, 서포트 0", 15),
    ("03_rear_cover",  P.rear_cover,  [180, 0, 0], 1, "후면 방열·틸트 커버",   "82x82 인셋 커버 + X축 틸트 귀, 서포트 0", 15),
    ("04_swivel_yoke", P.swivel_yoke, [0, 0, 0],   1, "2축 스위블 짐벌 요크",  "평탄 바닥 배치, 좌우 Pan±30° + 상하 Tilt-25°, 서포트 0", 15),
    ("05_wall_base",   P.wall_base,   [0, 0, 0],   1, "벽면 부착 마운트 판",   "66x66 벽면판, 자석4 + 나노겔 + 키홀, 서포트 0", 15),
    ("06_thumb_knob",  P.thumb_knob,  [0, 0, 0],   2, "M4 손조임 잠금 노브",   "공구 없이 팬/틸트 각도 고정 (노드당 2개), 서포트 0", 15),
]


def oriented(fn, rot):
    m = fn().rotate(rot)
    bb = m.bounding_box()
    return m.translate([-(bb[0] + bb[3]) / 2.0, -(bb[1] + bb[4]) / 2.0, -bb[2]])


def clean32(m, digits=3):
    mesh = m.to_mesh()
    v = np.round(np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), digits)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    for _ in range(4):
        uv, inv = np.unique(v, axis=0, return_inverse=True)
        f = inv.reshape(-1)[f]
        v = uv
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        fs = np.sort(f, axis=1)
        _, idx, cnt = np.unique(fs, axis=0, return_inverse=True, return_counts=True)
        keep &= cnt[idx.reshape(-1)] == 1
        if keep.all():
            break
        f = f[keep]
    return trimesh.Trimesh(v, f, process=False)


def closed(tm):
    e = np.sort(tm.edges, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2)) and tm.is_winding_consistent


def overhang_report(tm, ang_deg=50.0):
    n = tm.face_normals
    a = tm.area_faces
    zc = tm.triangles_center[:, 2]
    down = -n[:, 2]
    mask = (down > np.sin(np.radians(ang_deg))) & (zc > 0.35)
    flat = (down > 0.985) & (zc > 0.35)
    return float(a[mask].sum()), float(a[flat].sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    for fn in os.listdir(OUT):
        if fn.endswith(".stl"):
            os.remove(os.path.join(OUT, fn))
    rows = []
    for name, fn, rot, qty, ko, note, infill in SPEC:
        m = oriented(fn, rot)
        tm = clean32(m)
        bb = m.bounding_box()
        size = [float(bb[3] - bb[0]), float(bb[4] - bb[1]), float(bb[5] - bb[2])]
        oh, flat = overhang_report(tm)
        p = os.path.join(OUT, name + ".stl")
        tm.export(p)
        back = trimesh.load(p, process=True)
        ok = (
            closed(back)
            and back.is_watertight
            and len(m.decompose()) == 1
            and max(size[:2]) <= 240.0
            and size[2] <= 240.0
            and oh < 1.0
            and abs(back.volume - m.volume()) < max(2.0, 1e-3 * m.volume())
        )
        rows.append(dict(
            file=name + ".stl", ko=ko, qty_per_node=qty, qty_2nodes=qty * 2,
            note=note, infill=infill, size_mm=[round(s, 1) for s in size],
            volume_cm3=round(m.volume() / 1000.0, 1), watertight=bool(back.is_watertight),
            faces=int(len(back.faces)), bodies=len(m.decompose()),
            overhang_mm2=round(oh, 1), flat_overhang_mm2=round(flat, 1), ok=bool(ok),
        ))
        print("%-18s %-18s %5.1f x %5.1f x %4.1f mm  %5.1f cm3  닫힘=%-5s 덩어리=%d 처짐=%.1f/%.1f  %s" % (
            name, ko, size[0], size[1], size[2], m.volume() / 1000.0,
            back.is_watertight, len(m.decompose()), oh, flat, "OK" if ok else "<-- FAIL"
        ))
    with open(os.path.join(RES, "parts.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    bad = sum(not r["ok"] for r in rows)
    print(f"총 {len(rows)}종(노드당 7개) STL 생성 완료, 문제 {bad}개")
    sys.exit(bad)


if __name__ == "__main__":
    main()
