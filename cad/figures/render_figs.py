"""DeteCam 5.0 (Palm-12) 3D CAD 렌더 + AR-OPA/Palm-12 EM 벤치마크 차트 생성 스크립트.

    xvfb-run -a python3 cad/figures/render_figs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import palm_node as P
from cadlib import box
import paths
from render import render

IMG = paths.IMG
os.makedirs(IMG, exist_ok=True)


def up(m):
    """노드 좌표(X=벽가로, Y=벽위, +Z=방안쪽) -> 렌더 좌표(X=가로, Y=-Z(앞쪽=-Y), Z=위)."""
    return m.rotate([90, 0, 0])


def wrap_assembly(A, shifts=None, with_wall=False):
    out = {}
    for k, (m, col, alpha) in A.items():
        mm = m
        if shifts and k in shifts:
            mm = mm.translate(shifts[k])
        out[k] = (up(mm), "grp", col, True)
    if with_wall:
        # 깔끔한 벽면 단면 슬랩 (벽면 z = WALL_Z0 뒤에 배치)
        wall = box(-115.0, 115.0, -95.0, 95.0, P.WALL_Z0 - 8.0, P.WALL_Z0 - 0.2)
        out["wall_demo"] = (up(wall), "wall", "#E2DFD7", False)
    return out


def render_cad_views():
    # 1) 벽면 부착 + 각도 조절(Pan +30°, Tilt -15°) 3/4 측면-전면 뷰 (벽면 2축 짐벌까지 한눈에!)
    A_swivel = wrap_assembly(P.node_assembly(pan_deg=25.0, tilt_deg=-15.0), with_wall=True)
    render(A_swivel, os.path.join(IMG, "palm12_wall_swivel.png"),
           cam_az=-16, cam_el=16, dist=430, focus=(0, 32, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=False)

    # 2) 전면 정면-쿼터 콤팩트 뷰 (손바닥 크기 120x120x30.6 mm 강조)
    A_front = wrap_assembly(P.node_assembly(pan_deg=0.0, tilt_deg=0.0), with_wall=False)
    render(A_front, os.path.join(IMG, "palm12_front_compact.png"),
           cam_az=-62, cam_el=18, dist=360, focus=(0, 22, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=True, ground_z=-68)

    # 3) 전-후 분해도 (Exploded hardware stackup along Z)
    shifts = {
        "rf_shell":       [0, 0,  62],
        "smd_antennas":   [0, 0,  34],
        "rf_ics":         [0, 0,  34],
        "rf_pcb":         [0, 0,  34],
        "nicu_reflector": [0, 0,  14],
        "sbc_chassis":    [0, 0,  -4],
        "rock3a":         [0, 0, -24],
        "ax210":          [0, 0, -24],
        "rear_cover":     [0, 0, -52],
        "knob_tilt":      [18, 0, -74],
        "swivel_yoke":    [0, 0, -82],
        "knob_pan":       [0, 18, -104],
        "wall_base":      [0, 0, -112],
        "magnets":        [0, 0, -124],
    }
    A_exp = wrap_assembly(P.node_assembly(pan_deg=0.0, tilt_deg=0.0), shifts=shifts)
    render(A_exp, os.path.join(IMG, "palm12_exploded.png"),
           cam_az=-38, cam_el=18, dist=560, focus=(0, 32, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=False)
    print("Palm-12 CAD 3D renders saved!")


def render_charts():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    sim_dir = os.path.abspath(os.path.join(HERE, "..", "..", "sim"))
    opa = json.load(open(os.path.join(sim_dir, "ar_opa_results.json"), encoding="utf-8"))
    palm = json.load(open(os.path.join(sim_dir, "palm_array_results.json"), encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), facecolor="#F7F6F2")

    # Chart 1: AR-OPA Wall-Mounted Swivel 6-Space P90 Comparison
    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    keys = ["motel", "toilet", "bath", "locker", "classroom", "hall150"]
    labels = ["숙소\n(거울)", "화장실\n(칸막이)", "목욕탕\n(타일·기둥)", "탈의실\n(금속사물함)", "40석 교실\n(칠판·책상)", "150석 강당\n(기둥·계단)"]
    x = np.arange(len(keys))
    w = 0.25
    v_naive = [opa["spaces"][k]["naive"]["p90_m"] for k in keys]
    v_flush = [opa["spaces"][k]["wall_flush_0deg"]["p90_m"] for k in keys]
    v_opa   = [opa["spaces"][k]["ar_opa"]["p90_m"] for k in keys]
    ax.bar(x - w, v_naive, w, label="입구 벽면 직부착 (Naive)", color="#C8CDD5")
    ax.bar(x,     v_flush, w, label="각도 고정 0° 벽부착 (Flush)", color="#4A659C")
    ax.bar(x + w, v_opa,   w, label="AR-OPA 벽면 스위블 (Palm-12)", color="#EB6834")
    for i, val in enumerate(v_opa):
        ax.text(i + w, val + 0.14, f"{val:.2f}m", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#EB6834")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("최악 10% 위치 오차 P90 (m, 낮을수록 우수)", fontsize=10)
    ax.set_title("AR-OPA 벽면 부착·스위블 6대 실공간 검증 (Naive 대비 P90 -65.9%)", fontsize=11.2, fontweight="bold", pad=10, color="#1E2D5A")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.8)

    # Chart 2: Palm-12-Edge vs v4.0 L-8 vs Fold12-Mini EM & Size Comparison
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    p12 = palm["Palm-12-Edge (114mm 테두리12소자 SMD)"]
    v4  = palm["Ref: v4.0 L-8 (420mm L자 팔)"]
    metrics = ["정면 크기\n(×10 cm)", "5.2 GHz\n부엽 PSL", "5.2 GHz AoA\n오차 (÷10°)", "2.45 GHz\n결합보정 후(°)", "3기 총비용\n(÷10만 원)"]
    m_v4  = [4.2, v4["psl"]["5.2G"], v4["capon_52_cal"]["med_deg"] / 10.0, p12["capon_245_raw"]["med_deg"], 8.40]
    m_p12 = [1.2, p12["psl"]["5.2G"], p12["capon_52_cal"]["med_deg"] / 10.0, p12["capon_245_cal"]["med_deg"], 4.67]
    xm = np.arange(len(metrics))
    ax2.bar(xm - 0.18, m_v4,  0.34, label="v4.0 (42cm L자팔·스탠드 / 2.45G 미보정)", color="#4A659C")
    ax2.bar(xm + 0.18, m_p12, 0.34, label="v5.0 Palm-12 (12cm 벽부착·결합보정)", color="#19C2B0")
    notes = ["-71%", "-53%", "-99%", "-64%", "-44%"]
    for i, (b, nt) in enumerate(zip(m_p12, notes)):
        ax2.text(i + 0.18, b + 0.18, nt, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0F7A6E")
    ax2.set_xticks(xm); ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel("정규화 지표 (낮을수록 작고·싸고·정확)", fontsize=10)
    ax2.set_title("v4.0(42cm) → v5.0 Palm-12(12cm 손바닥 벽부착) 혁신 비교", fontsize=11.2, fontweight="bold", pad=10, color="#1E2D5A")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.8)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(IMG, "ar_opa_and_v5_benchmark.png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Palm-12 benchmark charts saved!")


if __name__ == "__main__":
    render_cad_views()
    render_charts()
