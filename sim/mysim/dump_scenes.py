"""Describe the 17 simulation rooms in words + top-view pictures (no ray tracing).

Re-runs the room builders with the same RNG seeds as the ray-traced data (mysim/scene_meta.py) and reads the receiver
spots / known reflectors from the ray-traced files in data_wall/.  Writes
  results/시뮬레이션-장면.md          rooms: size, shell materials, objects, people, receiver spots, map reflectors
  results/시뮬레이션-장면-좌표.md     seed-0 camera coordinates + desk / person centres
  results/scene_topviews.png        top views of all rooms (seed 0)

  PYTHONHASHSEED=0 python3 mysim/dump_scenes.py
"""
import os
import sys
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v1_ring"))
import scene_meta as S  # noqa: E402
import gen_wall as GW  # noqa: E402

DATA = os.path.join(ROOT, "data_wall")
OUT = os.path.join(ROOT, "results")

KIND = {  # room -> (Korean type, role)
    "classroom_a": ("교실", "시험실, 평가"), "classroom_b": ("교실", "시험실, 평가"), "classroom_c": ("교실", "시험실, 새 장면"),
    "lecture_a": ("강의실", "시험실, 평가"), "lecture_b": ("강의실(계단식)", "시험실, 평가"),
    "lodging_a": ("숙소", "시험실 밖, 개발"), "lodging_b": ("숙소", "시험실 밖, 평가"), "lodging_c": ("숙소(좁고 긴 방)", "시험실 밖, 새 장면"),
    "restroom_a": ("화장실", "시험실 밖, 개발"), "restroom_b": ("화장실", "시험실 밖, 평가"), "restroom_c": ("화장실(칸 5개)", "시험실 밖, 새 장면"),
    "bathhouse_a": ("목욕탕", "시험실 밖, 평가"), "bathhouse_b": ("목욕탕", "시험실 밖, 평가"),
    "locker_a": ("탈의실", "시험실 밖, 개발"), "locker_b": ("탈의실(가운데 사물함 섬)", "시험실 밖, 평가"),
    "office_a": ("사무실", "시험실 밖, 새 장면"), "worst_metal": ("금속 방(최악 장면)", "시험실 밖, 평가"),
}
TAG_KO = {
    "blackboard": "칠판", "whiteboard": "화이트보드", "window": "창", "winframe": "창틀", "door": "문", "podium": "교탁",
    "tv": "TV", "lockers": "사물함", "lockers_s": "사물함(남쪽 줄)", "lockers_n": "사물함(북쪽 줄)",
    "island_lockers": "가운데 사물함 섬", "ac_screen": "에어컨 · 스크린", "riser": "계단 단", "bath_part": "욕실 벽",
    "bath_part2": "욕실 벽", "bed": "침대", "wardrobe": "옷장", "desk": "책상", "fridge": "냉장고", "mirror": "거울",
    "bath_mirror": "욕실 거울", "full_mirror": "전신 거울", "shower_mirrors": "샤워대 줄 거울", "stall": "칸막이",
    "stall_front": "칸막이 앞판", "pipes": "배관", "sinks": "세면대", "bath_water": "탕(물)", "faucet": "수전",
    "sauna_door": "사우나 문", "bench": "벤치", "fan_dryer": "선풍기 · 드라이어", "metal_rack": "금속 선반",
    "aircon": "에어컨", "cabinets": "캐비닛", "monitor": "모니터", "partition": "파티션", "hand_dryer": "핸드 드라이어",
}
MAT_KO = {"body": "인체", "ceiling_board": "천장재", "concrete": "콘크리트", "glass": "유리", "metal": "금속",
          "plasterboard": "석고보드", "tile": "타일", "water": "물", "wood": "나무"}
CAM_KO = {"chest": "몸 앞쪽", "pencil_case": "필통", "glasses": "안경", "desk": "책상 위", "furniture": "가구",
          "wall_frame": "액자", "tv_charger": "TV · 충전기", "smoke_alarm": "화재경보기", "wall_fixture": "벽 부착물",
          "ceiling_vent": "천장 환풍구", "stall_tp": "칸 안 휴지걸이", "wall": "벽", "mirror_shelf": "거울 선반",
          "bench_wall": "벤치 쪽 벽", "locker_top": "사물함 위", "occluded": "가려진 곳", "general": "일반",
          "same_wall": "수신기와 같은 벽", "desk_charger": "책상 충전기", "wall_clock": "벽시계",
          "monitor_top": "모니터 위", "cabinet_top": "캐비닛 위", "partition_top": "파티션 위", "ceiling": "천장"}
SHELL_KO = {"floor": "바닥", "ceil": "천장", "wx0": "서벽(x=0)", "wxl": "동벽(x=L)", "wy0": "남벽(y=0)", "wyw": "북벽(y=W)"}
BODY = {"legs", "torso", "head"}


def f2(v):
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def people(boxes):
    heads = [(lo, hi) for lo, hi, m, t in boxes if t == "head"]
    legs = [(lo, hi) for lo, hi, m, t in boxes if t == "legs"]
    out = []
    for lo, hi in heads:
        c = (np.asarray(lo) + np.asarray(hi)) / 2
        standing = any(abs((l[0][0] + l[1][0]) / 2 - c[0]) < 0.02 and abs((l[0][1] + l[1][1]) / 2 - c[1]) < 0.02
                       for l in legs)
        out.append((c, standing))
    return out


def load_nodes(sc, sub):
    fp = os.path.join(DATA, sub, f"{sc}_2g4_s0.npy")
    if not os.path.exists(fp):
        return None
    D = np.load(fp, allow_pickle=True).item()
    return D


def room_md(sc, m, D, Dc, Df):
    L, W, H = m["dims"]
    kind, role = KIND[sc]
    lines = [f"### {sc} — {kind} ({role})", "",
             f"- 크기 L × W × H = {L:.1f} × {W:.1f} × {H:.1f} m (x: 0→L, y: 0→W, z: 바닥 0→천장)"]
    shell = {t: MAT_KO.get(mat, mat) for lo, hi, mat, t in m["boxes"] if t in SHELL_KO}
    lines.append("- 바닥 · 천장 · 벽 재질: " + ", ".join(f"{SHELL_KO[t]} {shell[t]}" for t in SHELL_KO if t in shell))
    groups = collections.OrderedDict()
    for lo, hi, mat, t in m["boxes"]:
        if t in SHELL_KO or t in BODY or t in ("deskleg",):
            continue
        groups.setdefault(t, []).append((np.asarray(lo), np.asarray(hi), mat))
    obj = []
    for t, items in groups.items():
        name = TAG_KO.get(t, t)
        mats = "/".join(sorted({MAT_KO.get(x[2], x[2]) for x in items}))
        if t == "desk" and len(items) > 3:
            sz = np.median([it[1] - it[0] for it in items], axis=0)
            lo = np.min([it[0] for it in items], axis=0)
            hi = np.max([it[1] for it in items], axis=0)
            obj.append(f"  - 책상 {len(items)}개({mats} 상판 + 금속 다리), 한 개 약 {sz[0]:.2f} × {sz[1]:.2f} m, "
                       f"높이 {np.median([it[1][2] for it in items]):.2f} m, 범위 x {lo[0]:.1f}~{hi[0]:.1f}, y {lo[1]:.1f}~{hi[1]:.1f} m "
                       f"(중심 좌표는 좌표 문서)")
        elif len(items) > 3:
            lo = np.min([it[0] for it in items], axis=0)
            hi = np.max([it[1] for it in items], axis=0)
            obj.append(f"  - {name} {len(items)}개({mats}), 범위 {f2(lo)}~{f2(hi)}")
        else:
            for lo, hi, mat in items:
                obj.append(f"  - {name}({MAT_KO.get(mat, mat)}): {f2(lo)}~{f2(hi)}")
    lines.append("- 물건(시드 0, 좌표는 최소점~최대점, m):")
    lines += obj
    ppl = people(m["boxes"])
    n_st = sum(1 for _, s in ppl if s)
    if ppl:
        st = "; ".join(f"({c[0]:.1f}, {c[1]:.1f})" for c, s in ppl if s)
        lines.append(f"- 사람(`base` 변형): {len(ppl)}명 = 앉음 {len(ppl) - n_st} · 서 있음 {n_st}"
                     + (f". 서 있는 사람 머리 중심 x, y: {st}" if st else "")
                     + ". 시드마다 자리가 바뀐다. `nop` 변형에서는 모두 뺀다.")
    else:
        lines.append("- 사람: 없음")
    cnt = collections.Counter(m["cam_types"])
    lines.append(f"- 카메라 40곳(시드 0): " + ", ".join(f"{CAM_KO.get(str(k), str(k))} {v}" for k, v in cnt.most_common())
                 + ". 좌표는 좌표 문서.")
    if D is not None:
        refl = D["user_reflectors"]
        if refl:
            rs = []
            for r in refl:
                axis, coord, lo, hi = r[0], r[1], r[2], r[3]
                rs.append(f"{'x' if axis == 0 else 'y'}={coord:.2f} 면 {f2(lo)}~{f2(hi)}")
            lines.append("- 추정기가 아는 반사면(지도, 큰 금속 · 거울 · 칠판): " + "; ".join(rs))
        nl = []
        for lab, p, note in zip(D["node_labels"], D["nodes"], D["node_notes"]):
            nl.append(f"{lab} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" + (f" [{note}]" if note else ""))
        lines.append("- 벽 수신 자리(벽에서 0.10 m, A = 2.0 m, B = 1.5 m; c = 모서리 옆 0.6 m, m = 벽 가운데): " + "; ".join(nl))
    if Dc is not None:
        lines.append("- 모서리 수신 자리(k, 두 벽에서 0.12 m): " + "; ".join(
            f"{lab} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" + (f" [{n}]" if n else "")
            for lab, p, n in zip(Dc["node_labels"], Dc["nodes"], Dc["node_notes"]) if lab.endswith("_A")))
    if Df is not None:
        lines.append("- 방 안 자유 자리(f, 돌판 비교 전용 · 큰 스탠드라 R8로 우리 수신기에는 금지): " + "; ".join(
            f"{lab} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" + (f" [{n}]" if n else "")
            for lab, p, n in zip(Df["node_labels"], Df["nodes"], Df["node_notes"])))
    lines.append("")
    return "\n".join(lines)


def coords_md(sc, m):
    L, W, H = m["dims"]
    lines = [f"### {sc}", "", "카메라(시드 0) 번호: 종류 x, y, z", ""]
    rows = [f"{i:02d}: {CAM_KO.get(str(t), str(t))} {c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}"
            for i, (c, t) in enumerate(zip(m["cams"], m["cam_types"]))]
    lines.append("```")
    lines += rows
    lines.append("```")
    if m["desks"]:
        lines.append("")
        lines.append(f"책상 중심 x, y(시드 0, {len(m['desks'])}개, 상판 높이 {m['desks'][0][4]:.2f} m):")
        lines.append("```")
        cs = [f"({(d[0] + d[1]) / 2:.2f}, {(d[2] + d[3]) / 2:.2f})" for d in m["desks"]]
        for k in range(0, len(cs), 8):
            lines.append(" ".join(cs[k:k + 8]))
        lines.append("```")
    ppl = people(m["boxes"])
    if ppl:
        lines.append("")
        lines.append(f"사람 머리 중심 x, y(시드 0, `base`; s = 서 있음):")
        lines.append("```")
        cs = [f"({c[0]:.2f}, {c[1]:.2f}{', s' if s else ''})" for c, s in ppl]
        for k in range(0, len(cs), 8):
            lines.append(" ".join(cs[k:k + 8]))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def draw(rooms, metas, Ds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    col = {"metal": "#888888", "glass": "#7fb3d5", "wood": "#c8a165", "tile": "#d7d7d7", "concrete": "#e6e6e6",
           "plasterboard": "#e8e0d0", "water": "#5dade2", "body": "#f1948a", "ceiling_board": "#eeeeee"}
    n = len(rooms)
    cols = 4
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 5, rws * 4.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, sc in zip(axes.flat, rooms):
        m = metas[sc]
        L, W, H = m["dims"]
        ax.axis("on")
        ax.add_patch(Rectangle((0, 0), L, W, fill=False, lw=1.5))
        for lo, hi, mat, t in m["boxes"]:
            if t in SHELL_KO or t in ("deskleg", "legs", "torso"):
                continue
            z = 3 if mat == "body" else 2
            ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fc=col.get(mat, "#cccccc"),
                                   ec="k" if mat in ("metal", "glass") else "none", lw=0.4, zorder=z, alpha=0.9))
        c = m["cams"]
        ax.scatter(c[:, 0], c[:, 1], s=9, marker="x", c="crimson", zorder=4)
        D = Ds.get(sc)
        if D is not None:
            for lab, p in zip(D["node_labels"], D["nodes"]):
                if lab.endswith("_A"):
                    ax.scatter(p[0], p[1], s=30, marker="s", c="navy", zorder=5)
                    ax.text(p[0], p[1], lab[:2], fontsize=6, color="navy", zorder=6)
        ax.set_xlim(-0.3, L + 0.3)
        ax.set_ylim(-0.3, W + 0.3)
        ax.set_aspect("equal")
        ax.set_title(f"{sc}  {L:.1f}x{W:.1f}x{H:.1f} m", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.suptitle("DeteCam simulation rooms, top view, seed 0 (x: camera, square: wall receiver spot at 2.0 m;"
                 " grey=metal, blue=glass/mirror, brown=wood, pink=people)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110)


def main():
    rooms = ["classroom_a", "classroom_b", "classroom_c", "lecture_a", "lecture_b",
             "lodging_a", "lodging_b", "lodging_c", "restroom_a", "restroom_b", "restroom_c",
             "bathhouse_a", "bathhouse_b", "locker_a", "locker_b", "office_a", "worst_metal"]
    metas, Ds = {}, {}
    md = [
        "# DeteCam 시뮬레이션 장면 (방 17개)",
        "",
        "11-24~26 시뮬레이션에 쓴 방을 코드(`sim/v1_ring/gen_scenes.py`, `mysim/fresh_scenes.py`, `mysim/gen_wall.py`)에서 "
        "그대로 다시 만들어 적은 것이다(`PYTHONHASHSEED=0 python3 mysim/dump_scenes.py`). 광선 추적 데이터의 카메라 좌표와 "
        "모두 같음을 확인했다. 모든 숫자는 시뮬레이션 설정값이며 실제 방을 잰 것이 아니다.",
        "",
        "## 공통",
        "",
        "- 좌표: 방의 한 모서리가 원점. x는 길이 L 방향(서 → 동), y는 폭 W 방향(남 → 북), z는 높이. 단위 m.",
        "- 모든 물건은 직육면체다. 재질은 ITU-R P.2040 값(Sionna RT ITU 재질: 콘크리트 · 석고보드 · 나무 · 유리 · 금속 · 천장재)이다. "
        "타일은 콘크리트 값을 쓴다. 인체는 2.4 GHz εr 52.7 · σ 1.74 S/m, 5 GHz 49.3 · 4.5 S/m(두께 0.20 m), 물은 2.4 GHz 78 · 1.5, "
        "5 GHz 73 · 5.2다. 칠판 · 거울 · 사물함 · 책상 다리 · 문은 금속이다.",
        "- 사람: 앉은 사람은 몸통 + 머리, 서 있는 사람은 다리 + 몸통 + 머리 상자. 변형 `base`는 사람 포함(시험실 기본), "
        "`nop`은 사람을 모두 뺀 것(시험실 밖 기본), `crowd`는 서 있는 사람 3명(작은 방) 또는 8명을 더한 것, "
        "`bare`는 사람과 옮길 수 있는 가구를 뺀 것이다.",
        "- 시드(0 · 1 · 2)마다 바뀌는 것: 카메라 40곳, 사람 자리, 시험실 책상 위치의 작은 흔들림. 방 크기 · 벽 · 고정 "
        "가구 · 거울 · 칠판은 같다. 난수는 `np.random.default_rng(2000 + seed*101 + hash(scene)%97)`라서 "
        "`PYTHONHASHSEED=0`이 꼭 필요하다.",
        "- 카메라는 실제로 숨기는 자리에 둔다(종류는 방마다 적음). 카메라 안테나 방향은 무작위다.",
        "- 수신 자리: 벽 8곳(모서리 옆 c0~c3, 벽 가운데 m0~m3) × 높이 2.0 m(A) · 1.5 m(B), 벽에서 0.10 m. 가구가 막으면 "
        "가구 앞으로 나오거나 옆으로 미끄러지거나 가구 위로 올라간다([ ] 안 메모). 모서리 자리 k0~k3, 방 안 자유 자리 "
        "f0~f4(돌판 비교 전용)도 있다. 순서 c0 남벽 서쪽, c1 동벽 남쪽, c2 북벽 동쪽, c3 서벽 북쪽, m0 남벽 가운데, "
        "m1 동벽 가운데, m2 북벽 가운데, m3 서벽 가운데.",
        "- 개발용 방(조절값을 정한 방): classroom_a, lodging_a, restroom_a, locker_a. 새 장면(classroom_c, lodging_c, "
        "office_a, restroom_c)은 나중에 따로 만든 처음 보는 방이다.",
        "- 위에서 본 그림: `results/scene_topviews.png`(시드 0). 좌표 목록: 「시뮬레이션-장면-좌표」.",
        "",
        "## 방별",
        "",
    ]
    cmd = ["# DeteCam 시뮬레이션 장면 좌표 (시드 0)", "",
           "「시뮬레이션-장면」의 방 17개에서 시드 0의 카메라 40곳, 책상 중심, 사람 머리 중심을 적은 것이다. "
           "시뮬레이션 설정값이며 실측 아님. 좌표 기준은 장면 문서의 공통 절과 같다.", ""]
    for sc in rooms:
        m = S.meta_for(sc, 0)
        D = load_nodes(sc, "base")
        assert D is None or np.allclose(D["cams"], m["cams"]), sc
        metas[sc], Ds[sc] = m, D
        md.append(room_md(sc, m, D, load_nodes(sc, "corner_base"), load_nodes(sc, "free_nop")))
        cmd.append(coords_md(sc, m))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "시뮬레이션-장면.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(OUT, "시뮬레이션-장면-좌표.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(cmd))
    draw(rooms, metas, Ds, os.path.join(OUT, "scene_topviews.png"))
    print("ok")


if __name__ == "__main__":
    main()
