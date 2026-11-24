# -*- coding: utf-8 -*-
"""DeteCam 6.0 (Nano-2 AI) 핵심 펌웨어·PyTorch Edge-AI·AR-OPA-2 통합 검증 테스트."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.trif_v12 import (
    NANO12_EDGE_M,
    coherent_ratio,
    mutual_coupling_matrix,
    rclpa_repair_phase_flips,
    apply_coupling_compensation,
    capon_score,
    raps_conformal_calibrate,
    raps_predict_set,
    steering_matrix,
)
from core.ai_engine import Nano2AIEngine
from core.ar_opa import ObstacleBox, recommend_placement, auto_verify_nodes


def test_nano12_sub_half_wavelength_5ghz():
    """Nano-2 (84x84 mm PCB) 12소자 최소 간격이 5.85 GHz 반파장(25.6 mm) 미만이어야 그레이팅 로브 0%."""
    dists = []
    for i in range(len(NANO12_EDGE_M)):
        for j in range(i + 1, len(NANO12_EDGE_M)):
            dists.append(float(np.linalg.norm(NANO12_EDGE_M[i] - NANO12_EDGE_M[j])))
    d_min_mm = min(dists) * 1000.0
    assert d_min_mm < 25.6, f"5GHz grating-lobe bound violated: d_min={d_min_mm:.1f} mm"
    assert d_min_mm > 18.0, f"Elements too close for SMD clearance: d_min={d_min_mm:.1f} mm"


def test_cm_rclpa_and_reconcovnet_v4_psd():
    """실제 학습된 PyTorch ReconCovNet-v4 가중치가 정확한 Hermitian PSD 공분산을 복원하는지 검증."""
    rng = np.random.default_rng(7)
    fc = 5.22e9
    bore = np.array([0.0, 1.0, 0.0])
    C = mutual_coupling_matrix(NANO12_EDGE_M, fc)
    C_inv = np.linalg.inv(C)

    tx = np.array([[0.8, 3.2, 0.3]])
    a = steering_matrix(tx, np.zeros(3), NANO12_EDGE_M, bore, fc)[0]
    a /= np.linalg.norm(a)
    H_clean = np.outer(C @ a, np.exp(1j * rng.uniform(0, 2 * np.pi, 32)))
    # 15% AX210 180도 위상 반전 주입
    H_flip = H_clean.copy()
    H_flip[[2, 9]] *= -1.0

    H_fix = rclpa_repair_phase_flips(H_flip, NANO12_EDGE_M, fc, bore=bore, C_mat=C)
    H_cal = apply_coupling_compensation(H_fix, C_inv)

    engine = Nano2AIEngine()
    R_ai, _ = engine.reconstruct_covariance(H_cal, fc_hz=fc)
    assert R_ai.shape == (12, 12)
    # Hermitian 대칭성 및 양반정치(PSD) 수학적 보장 검증
    assert np.allclose(R_ai, R_ai.conj().T, atol=1e-5)
    eigvals = np.linalg.eigvalsh(R_ai)
    assert float(np.min(eigvals)) >= -1e-4, f"Non-PSD eigenvalue: {np.min(eigvals)}"

    # Capon 정점 확인
    pts = np.array([[0.8, 3.2, 0.3], [-1.5, 3.2, 0.3], [2.2, 3.2, 0.3]])
    score, cv = capon_score(H_flip, np.zeros(3), bore, pts, fc, elem_local_m=NANO12_EDGE_M, C_inv=C_inv)
    assert int(np.argmax(score)) == 0
    assert cv < 0.08


def test_camguard_net_causal_separation():
    """실제 학습된 PyTorch CamGuard-Net 가중치가 라이브 방송 하드 네거티브와 몰래카메라를 분리하는지 검증."""
    engine = Nano2AIEngine()
    rng = np.random.default_rng(2026)
    T = 60
    stim = np.zeros(T); stim[20:40] = 1.0

    # 1) 몰래카메라: 5초 조명/움직임 자극에 비트레이트 +55% 인과 반응
    up_cam = 550.0 * (1.0 + 0.55 * stim + rng.normal(0, 0.06, T))
    dn_cam = rng.uniform(20, 60, T)
    seq_cam = np.stack([np.log1p(up_cam) / 8.0, np.log1p(dn_cam) / 8.0, np.log1p(up_cam) / 8.0, np.full(T, 0.85)])
    r_cam = float(np.corrcoef(up_cam, stim)[0, 1])
    is_cam, p_cam = engine.classify_camera(seq_cam, [r_cam, 1.0, 1.0])
    assert is_cam and p_cam > 0.85, f"Covert camera missed: p={p_cam}"

    # 2) 하드 네거티브 (스마트폰 유튜브 라이브 업로드): 업링크는 비슷하지만 실내 조명 자극에 무반응!
    up_neg = 550.0 * (1.0 + rng.normal(0, 0.08, T))
    dn_neg = rng.uniform(50, 120, T)
    seq_neg = np.stack([np.log1p(up_neg) / 8.0, np.log1p(dn_neg) / 8.0, np.log1p(up_neg) / 8.0, np.full(T, 0.82)])
    r_neg = float(np.corrcoef(up_neg, stim)[0, 1])
    is_neg, p_neg = engine.classify_camera(seq_neg, [r_neg, 0.0, 0.0])
    assert (not is_neg) and p_neg < 0.30, f"Live-stream hard negative false alarm: p={p_neg}"


def test_ar_opa_2node_lshape_virtual_mirror():
    """AR-OPA-2 가 L자형 코너 사각지대 방에서 단 2노드 + 가상 거울 가시성으로 최적 벽면 배치 반환."""
    obs = [
        ObstacleBox(3.1, 6.0, 2.4, 4.8, 0.0, 2.6, kind="wall", loss_db=14.0, refl=0.6),
        ObstacleBox(0.0, 0.06, 1.0, 2.6, 0.8, 2.0, kind="mirror", loss_db=18.0, refl=0.95),
    ]
    roi = [[x, y, 1.1] for x in np.linspace(0.6, 5.4, 5) for y in np.linspace(0.6, 4.2, 4)
           if not (x > 3.0 and y > 2.3)]
    res = recommend_placement((6.0, 4.8, 2.6), obs, roi, k=2, mirror_aware=True)
    assert len(res["nodes"]) == 2
    assert res["virtual_mirror_coverage_pct"] >= 85.0
    for nd in res["nodes"]:
        assert -30 <= nd["pan_deg"] <= 30
        assert nd["tilt_deg"] in (0, -15, -25)

    ver = auto_verify_nodes([2.1, 3.0], [0.025, 0.031])
    assert ver["ok"] and ver["placement_score"] >= 80


def test_raps_conformal_90pct_guarantee():
    """RAPS Conformal Prediction 이 임의 분포에서 통계적 90% 커버리지를 보장하는지 검증."""
    rng = np.random.default_rng(123)
    n_cal, n_test, n_cls = 300, 300, 40
    logits = rng.normal(0, 1.4, (n_cal + n_test, n_cls))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(n_cls, p=probs[i]) for i in range(n_cal + n_test)])

    q_hat = raps_conformal_calibrate(probs[:n_cal], y[:n_cal], alpha=0.10)
    covered = sum(int(y[n_cal + i] in raps_predict_set(probs[n_cal + i], q_hat)) for i in range(n_test))
    assert covered / n_test >= 0.87, f"Empirical coverage too low: {covered / n_test:.3f}"


def test_coherent_ratio_cancels_common_phase():
    """이중 체인 비(H_B/H_A)가 AX210 공통 위상 오차(CFO/SFO/PDD)를 완벽히 제거하는지 검증."""
    rng = np.random.default_rng(9)
    phi_common = rng.uniform(-np.pi, np.pi, 32)
    h_a = np.exp(1j * phi_common)
    h_b = np.exp(1j * (phi_common + 0.75))
    ratio = coherent_ratio(h_b, h_a)
    assert np.allclose(np.angle(ratio), 0.75, atol=1e-6)
