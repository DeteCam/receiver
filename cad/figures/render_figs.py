"""DeteCam 6.0 (Nano-2 AI) 3D CAD 렌더 + 10대 실공간 3D 광선추적·PyTorch AI 벤치마크 & 빔패턴 차트 생성기.

    xvfb-run -a python3 cad/figures/render_figs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "model"),
                os.path.join(os.path.dirname(HERE), "export"),
                os.path.join(ROOT, "sim"), os.path.join(ROOT, "firmware")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import nano2_node as P
from cadlib import box
import paths
from render import render
from raytrace_engine import NANO12_EDGE_M, mutual_coupling_matrix, C0, build_10_diverse_spaces, trace_multipath_rays, synthesize_wideband_csi
from core.ai_engine import Nano2AIEngine
from core.trif_v12 import rclpa_repair_phase_flips, apply_coupling_compensation

IMG = paths.IMG
os.makedirs(IMG, exist_ok=True)


def up(m):
    return m.rotate([90, 0, 0])


def wrap_assembly(A, shifts=None, with_wall=False):
    out = {}
    for k, (m, col, alpha) in A.items():
        mm = m
        if shifts and k in shifts:
            mm = mm.translate(shifts[k])
        out[k] = (up(mm), "grp", col, True)
    if with_wall:
        wall = box(-100.0, 100.0, -85.0, 85.0, P.WALL_Z0 - 8.0, P.WALL_Z0 - 0.2)
        out["wall_demo"] = (up(wall), "wall", "#E2DFD7", False)
    return out


def render_cad_views():
    A_swivel = wrap_assembly(P.node_assembly(pan_deg=25.0, tilt_deg=-15.0), with_wall=True)
    render(A_swivel, os.path.join(IMG, "nano2_wall_swivel.png"),
           cam_az=-18, cam_el=16, dist=390, focus=(0, 34, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=False)

    A_front = wrap_assembly(P.node_assembly(pan_deg=0.0, tilt_deg=0.0), with_wall=False)
    render(A_front, os.path.join(IMG, "nano2_front_compact.png"),
           cam_az=-60, cam_el=18, dist=330, focus=(0, 26, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=True, ground_z=-56)

    shifts = {
        "rf_shell":       [0, 0,  54],
        "smd_antennas":   [0, 0,  30],
        "rf_ics":         [0, 0,  30],
        "rf_pcb":         [0, 0,  30],
        "nicu_reflector": [0, 0,  12],
        "sbc_chassis":    [0, 0,  -4],
        "ax210":          [0, 0, -24],
        "nanopi_r5c":     [0, 0, -32],
        "rear_cover":     [0, 0, -60],
        "knob_tilt":      [18, 0, -78],
        "swivel_yoke":    [0, 0, -86],
        "knob_pan":       [0, 18, -106],
        "wall_base":      [0, 0, -114],
        "magnets":        [0, 0, -126],
    }
    A_exp = wrap_assembly(P.node_assembly(pan_deg=0.0, tilt_deg=0.0), shifts=shifts)
    render(A_exp, os.path.join(IMG, "nano2_exploded.png"),
           cam_az=-38, cam_el=18, dist=520, focus=(0, 34, 0), size=(1400, 1000),
           bg="#F7F6F2", fov=28, ground=False)
    print("Nano-2 CAD 3D renders saved!")


def setup_font():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def render_charts():
    setup_font()
    sim_dir = os.path.join(ROOT, "sim")
    res10 = json.load(open(os.path.join(sim_dir, "nano2_10space_results.json"), encoding="utf-8"))
    ai_res = json.load(open(os.path.join(sim_dir, "ai_training_results.json"), encoding="utf-8"))

    # ── Chart 1: 10-Space 3D Ray-Tracing P90 & <=1.0m Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), facecolor="#F7F6F2")

    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    skeys = list(res10["per_space"].keys())
    short = ["숙소\n원룸", "L자\n코너", "화장실\n3칸", "목욕탕\n기둥", "금속\n탈의실",
             "피팅룸\n부스", "*40석\n교실", "*150석\n강당", "*통유리\n오피스", "*과밀\n기숙사"]
    x = np.arange(len(skeys))
    w = 0.25
    v_p3  = [min(3.5, res10["per_space"][k]["bands"]["5.22G"]["v5_Palm12_3Node_Classic"]["p90_m"]) for k in skeys]
    v_n2c = [min(3.5, res10["per_space"][k]["bands"]["5.22G"]["Nano2_2Node_Classic"]["p90_m"]) for k in skeys]
    v_n2a = [min(3.5, res10["per_space"][k]["bands"]["5.22G"]["Nano2_2Node_AI"]["p90_m"]) for k in skeys]

    ax.bar(x - w, v_p3,  w, label="v5.0 3기 Palm-12 (120mm, 고전)", color="#94A3B8")
    ax.bar(x,     v_n2c, w, label="2기 Nano-2 (96mm, AI 미적용)", color="#4A659C")
    ax.bar(x + w, v_n2a, w, label="v6.0 단 2기 Nano-2 AI (ReconCovNet+MA-PINN)", color="#0EA5A4")
    for i, k in enumerate(skeys):
        raw_p3 = res10["per_space"][k]["bands"]["5.22G"]["v5_Palm12_3Node_Classic"]["p90_m"]
        raw_ai = res10["per_space"][k]["bands"]["5.22G"]["Nano2_2Node_AI"]["p90_m"]
        if raw_p3 > 3.5:
            ax.text(i - w, 3.53, f"{raw_p3:.1f}m", ha="center", va="bottom", fontsize=7.5, color="#64748B", fontweight="bold")
        ax.text(i + w, min(3.5, raw_ai) + 0.06, f"{raw_ai:.2f}", ha="center", va="bottom", fontsize=7.6, color="#0F766E", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.4)
    ax.set_ylim(0, 3.95)
    ax.set_ylabel("5.22 GHz 최악 10% 위치 오차 P90 (m, 낮을수록 우수)", fontsize=9.8)
    ax.set_title("10대 실공간 3D 광선추적 (*미학습 공간 4종 포함): 단 2기 Nano-2 AI가 3기 Palm-12 압도!", fontsize=10.6, fontweight="bold", pad=10, color="#1E2D5A")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    # Right panel: Aggregate <=1.0m accuracy across All 10 / Unseen 4 / NLoS Blind Spots (Both 2.45G & 5.22G)
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    groups = ["전체 10공간\n(5.22 GHz)", "미학습 4공간\n(5.22 GHz)", "사각지대 NLoS\n(5.22 GHz)", "전체 10공간\n(2.45 GHz)", "사각지대 NLoS\n(2.45 GHz)"]
    gkeys = [("all_10_spaces", "5.22G"), ("unseen_4_spaces", "5.22G"), ("nlos_blindspot_cases", "5.22G"),
             ("all_10_spaces", "2.45G"), ("nlos_blindspot_cases", "2.45G")]
    xg = np.arange(len(groups))
    wb = 0.19
    a_1n  = [res10["aggregate"][g][b]["Nano1_1Node_AI"]["within_1m_pct"] for g, b in gkeys]
    a_3p  = [res10["aggregate"][g][b]["v5_Palm12_3Node_Classic"]["within_1m_pct"] for g, b in gkeys]
    a_2ai = [res10["aggregate"][g][b]["Nano2_2Node_AI"]["within_1m_pct"] for g, b in gkeys]
    a_2ph = [res10["aggregate"][g][b]["Nano2_2Node_AI_Phone"]["within_1m_pct"] for g, b in gkeys]

    ax2.bar(xg - 1.5*wb, a_1n,  wb, label="1기 Nano-1 AI (단일 한계)", color="#CBD5E1")
    ax2.bar(xg - 0.5*wb, a_3p,  wb, label="v5.0 3기 Palm-12 (고전)", color="#4A659C")
    ax2.bar(xg + 0.5*wb, a_2ai, wb, label="v6.0 단 2기 Nano-2 AI", color="#0EA5A4")
    ax2.bar(xg + 1.5*wb, a_2ph, wb, label="2기 Nano-2 AI + 0원 폰보행", color="#EB6834")
    for i, (v1, v2) in enumerate(zip(a_2ai, a_2ph)):
        ax2.text(i + 0.5*wb, v1 + 1.0, f"{v1:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#0F766E", fontweight="bold")
        ax2.text(i + 1.5*wb, v2 + 1.0, f"{v2:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#C2410C", fontweight="bold")
    ax2.set_xticks(xg); ax2.set_xticklabels(groups, fontsize=8.8)
    ax2.set_ylim(25, 118)
    ax2.set_ylabel("1.0 m 이내 탐지 성공률 (%, 높을수록 우수)", fontsize=9.8)
    ax2.set_title("패널 수·AI 효과 종합 검증 (2기+AI가 3기 초과 · 1기 대비 +35%p 도약)", fontsize=10.6, fontweight="bold", pad=10, color="#1E2D5A")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.2, loc="upper center", ncol=2)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(IMG, "nano2_10space_ai_benchmark.png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Chart 2: REAL ReconCovNet-v4 2D Beampattern Super-Resolution + CamGuard-Net Ablation ──
    engine = Nano2AIEngine()
    spaces = build_10_diverse_spaces(rng_seed=2026)
    sp = spaces["classroom_40"]  # UNSEEN room!
    rx = np.array([0.06, 3.6, 1.95])
    bore = np.array([1.0, 0.0, -0.25]); bore /= np.linalg.norm(bore)
    tx = np.array([4.2, 4.1, 1.25])
    rays = trace_multipath_rays(tx, rx, bore, sp)
    fc = 2.45e9
    C_nom = mutual_coupling_matrix(NANO12_EDGE_M, fc)
    C_inv = np.linalg.inv(C_nom)
    rng = np.random.default_rng(77)
    pjit = np.exp(1j * rng.normal(0.0, np.radians(5.0), 12))
    C_true = np.diag(pjit) @ C_nom + 0.08 * (rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12))) / np.sqrt(24.0)
    H_raw = synthesize_wideband_csi(rays, bore, NANO12_EDGE_M, fc, 40e6, n_sub=32, snr_db=13.0, flip_prob=0.15, C_mat=C_true, rng=rng)
    H_fix = rclpa_repair_phase_flips(H_raw, NANO12_EDGE_M, fc, bore=bore, C_mat=C_nom)
    H_cal = apply_coupling_compensation(H_fix, C_inv)
    R_ai, R_cl = engine.reconstruct_covariance(H_cal, fc_hz=fc)

    from raytrace_engine import panel_axes
    ex, ey, ez = panel_axes(bore)
    u_w = (tx - rx) / np.linalg.norm(tx - rx)
    true_az = np.degrees(np.arctan2(np.dot(u_w, ex), np.dot(u_w, ey)))
    true_el = np.degrees(np.arcsin(np.clip(np.dot(u_w, ez), -1.0, 1.0)))

    azs = np.linspace(-60, 60, 91)
    els = np.linspace(-40, 40, 61)
    AZ, EL = np.meshgrid(np.radians(azs), np.radians(els))
    U_loc = np.stack([np.cos(EL) * np.sin(AZ), np.cos(EL) * np.cos(AZ), np.sin(EL)], axis=-1).reshape(-1, 3)
    A_loc = np.exp(2j * np.pi * (U_loc @ NANO12_EDGE_M.T) / (C0 / fc))

    def spec_db(R, load):
        Ri = np.linalg.inv(R + load * (np.real(np.trace(R)) / 12.0) * np.eye(12))
        p = 1.0 / np.maximum(np.real(np.einsum("nm,mk,nk->n", A_loc.conj(), Ri, A_loc)), 1e-12)
        db = 10.0 * np.log10(p / np.max(p) + 1e-6)
        return db.reshape(len(els), len(azs))

    S_cl = spec_db(R_cl, 0.10)
    S_ai = spec_db(R_ai, 0.08)

    fig2 = plt.figure(figsize=(16.5, 5.0), facecolor="#F7F6F2")
    gs = fig2.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.055, 0.22, 1.15], wspace=0.22, left=0.045, right=0.985, top=0.88, bottom=0.13)
    a0 = fig2.add_subplot(gs[0, 0])
    a1 = fig2.add_subplot(gs[0, 1])
    cax = fig2.add_subplot(gs[0, 2])
    ac = fig2.add_subplot(gs[0, 4])

    for idx_a, (a, S, ttl) in enumerate([
        (a0, S_cl, "AI 적용 전: 8.4 cm 고전 2.45 GHz 빔 (흐릿·번짐)"),
        (a1, S_ai, "AI 적용 후: ReconCovNet-v4 초해상도 (미학습 교실)"),
    ]):
        im = a.imshow(S, extent=[-60, 60, -40, 40], origin="lower", aspect="auto", cmap="turbo", vmin=-14, vmax=0)
        a.scatter([true_az], [true_el], s=145, marker="*", c="white", edgecolors="black", lw=1.0, label="실제 카메라 방향")
        a.set_title(ttl, fontsize=9.8, fontweight="bold", color="#1E2D5A", pad=8)
        a.set_xlabel("좌우 방위각 Azimuth (°)", fontsize=9)
        if idx_a == 0:
            a.set_ylabel("상하 고도각 Elevation (°)", fontsize=9)
        a.legend(loc="upper right", fontsize=8.2, framealpha=0.88)
    cb = fig2.colorbar(im, cax=cax)
    cb.set_label("정규화 전파 세기 (dB)", fontsize=9)

    ac.set_facecolor("#FFFFFF")
    cg = ai_res["camguard_net"]
    cats = ["F1 점수", "정밀도\n(Precision)", "재현율\n(Recall)", "오경보율\n(낮을수록↓)"]
    vp = [cg["passive_only"]["f1"]*100, cg["passive_only"]["precision"]*100, cg["passive_only"]["recall"]*100, cg["passive_only"]["false_alarm_pct"]]
    vc = [cg["full_with_causal"]["f1"]*100, cg["full_with_causal"]["precision"]*100, cg["full_with_causal"]["recall"]*100, cg["full_with_causal"]["false_alarm_pct"]]
    xc = np.arange(len(cats))
    ac.bar(xc - 0.18, vp, 0.34, label="수동 분석만 (Passive)", color="#94A3B8")
    ac.bar(xc + 0.18, vc, 0.34, label="CamGuard-Net + 5초 인과자극", color="#EB6834")
    for i, (a1v, a2v) in enumerate(zip(vp, vc)):
        ac.text(i - 0.18, a1v + 1.8, f"{a1v:.1f}%", ha="center", va="bottom", fontsize=8.2, color="#475569")
        ac.text(i + 0.18, a2v + 1.8, f"{a2v:.1f}%", ha="center", va="bottom", fontsize=8.4, color="#C2410C", fontweight="bold")
    ac.set_xticks(xc); ac.set_xticklabels(cats, fontsize=8.8)
    ac.set_ylim(0, 124)
    ac.set_ylabel("성능 (%)", fontsize=9.5)
    ac.set_title("CamGuard-Net: 라이브 방송 등 하드 네거티브 분리 (800대)", fontsize=9.8, fontweight="bold", color="#1E2D5A", pad=8)
    ac.grid(axis="y", alpha=0.25, linestyle="--")
    ac.spines[["top", "right"]].set_visible(False)
    ac.legend(frameon=False, fontsize=8.4, loc="upper center", ncol=2)

    fig2.savefig(os.path.join(IMG, "nano2_ai_superres_and_camguard.png"), dpi=180, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print("Nano-2 scientific AI & 10-space charts saved!")


if __name__ == "__main__":
    render_cad_views()
    render_charts()
