# -*- coding: utf-8 -*-
"""DeteCam 5.0 (Palm-12) 핵심 엔진 자동 검증 테스트 (TriF v12 + C⁻¹ 결합보정 + RAPS + 3단계 카메라 판별 + 벽면 스위블 AR-OPA)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import trif_v12 as T
import score as S
import ar_opa as O


def test_palm12_mutual_coupling_and_rclpa_capon():
    """Palm-12(112x112 mm) 소형 배열에서 상호결합(C) + 15% 180° 위상반전이 동시에 발생해도
    C⁻¹ 보정 + RCLPA + High-DOF Capon 이 2.45 GHz & 5.2 GHz 모두 정확히 목표를 찾는지 검증."""
    node = np.array([0.0, 0.0, 1.95])
    bore = np.array([0.0, 1.0, -0.25]); bore /= np.linalg.norm(bore)
    gt = np.array([1.1, 3.2, 1.05])
    pts = np.array([[x, 3.2, z] for x in np.linspace(-1.5, 2.5, 17) for z in np.linspace(0.5, 1.8, 7)])
    gt_idx = int(np.argmin(np.linalg.norm(pts - gt, axis=1)))

    for ch in (6, 44):  # 2.437 GHz & 5.220 GHz
        fc = T.channel_freq_hz(ch)
        C_true = T.mutual_coupling_matrix(T.PALM12_EDGE_M, fc)
        C_inv = np.linalg.inv(C_true)
        a_true = T.steering_matrix(gt[None, :], node, T.PALM12_EDGE_M, bore, fc)[0]
        # 상호결합이 섞인 수신 CSI 생성 + 2개 소자(16.7%) 180° 위상 반전 주입
        h_coupled = C_true @ a_true
        rng = np.random.default_rng(100 + ch)
        H = h_coupled[:, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, (1, 30)))
        H += 0.04 * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        H_clean = H.copy()
        # 위상 반전은 보정된 도메인 또는 AX210 포트에서 발생 -> 복구 검증
        H_flip = H_clean.copy()
        H_flip[2] *= -1.0
        H_flip[9] *= -1.0

        log_s, cv_f = T.capon_score(H_flip, node, bore, pts, fc, C_inv=C_inv, repair_flips=True)
        est = pts[int(np.argmax(log_s))]
        err_m = float(np.linalg.norm(est - pts[gt_idx]))
        assert err_m < 0.35, f"ch={ch} Palm-12 오차 과다: {err_m:.2f} m"
        assert cv_f < 0.08


def test_virtual_mirror_and_raps_coverage():
    """10면 가상 거울 앵커 생성 + RAPS 90% 통계 보장 Conformal 보정 검증."""
    anchors = T.virtual_mirror_anchors_10plane([0.06, 2.0, 1.95], [1.0, 0.0, -0.2], (8.5, 7.5, 2.8))
    assert len(anchors) == 10
    assert T.should_activate_mirror_anchors(14.0, [0.06, 0.08, 0.07]) is True
    assert T.should_activate_mirror_anchors(4.0, [0.02, 0.03, 0.04]) is False

    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 1.2, (400, 12))
    true_idx = rng.integers(0, 12, 400)
    logits[np.arange(400), true_idx] += 2.6
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    q_hat = T.raps_conformal_calibrate(probs[:200], true_idx[:200], alpha=0.10)
    hits = [true_idx[i] in T.raps_predict_set(probs[i], q_hat) for i in range(200, 400)]
    assert np.mean(hits) >= 0.87


def test_camera_classifier_distinguishes_camera_vs_phone_laptop_iot():
    """3단계 카메라 판별기가 IP 카메라만 강하게 포착하고 스마트폰/노트북/방밖 기기를 기각하는지 검증."""
    stim_cam = S.causal_stimulus_score([180, 190], [420, 460], [195, 185])
    assert stim_cam["triggered"] is True
    cam_res = S.score_wifi_device(dict(
        mac="00:12:34:aa:bb:cc", uplink_kbps=450, downlink_kbps=25, mean_len=1180,
        pattern="steady", ssid="NAX_88412", bssid="02:11:22:33:44:55",
        single_stream_legacy=True, causal_stimulus_hit=stim_cam["triggered"], pos_3d_m=[2.2, 2.5, 1.1],
    ), room_dims=(4.5, 3.8, 2.6))
    assert cam_res["level"] == "camera" and cam_res["score"] >= 8

    # 일반 스마트폰 (다운링크 우세) -> 기각
    ph_res = S.score_wifi_device(dict(
        mac="aa:bb:cc:11:22:33", uplink_kbps=40, downlink_kbps=1800, mean_len=900,
        pattern="quiet", ssid="School_WiFi", bssid="00:1a:2b:3c:4d:5e", pos_3d_m=[2.0, 2.0, 0.9],
    ), room_dims=(4.5, 3.8, 2.6))
    assert ph_res["level"] == "ignore"

    # 옆방 카메라 (방 경계 밖) -> outside 차단
    out_res = S.score_wifi_device(dict(
        mac="00:12:34:ff:ee:dd", uplink_kbps=500, downlink_kbps=20, mean_len=1200,
        pattern="steady", ssid="IPC-2026", bssid="02:aa:bb:cc:dd:ee", pos_3d_m=[6.2, 2.0, 1.2],
    ), room_dims=(4.5, 3.8, 2.6))
    assert out_res["level"] == "outside" and out_res["inside_room"] is False


def test_ad8317_3node_cross_subtraction_and_ble_zone():
    """AD8317 x 3대 교차 차감으로 Wi-Fi 제거 및 LTE/5G 카메라 최근접 노드 + 무료 BLE 구역 추정 검증."""
    V = np.full((3, 20), 1.55)
    wifi_mask = np.zeros(20, dtype=bool)
    wifi_mask[3:6] = True
    V[:, 3:6] = 1.20       # Wi-Fi 구간 (교차 차감으로 무시되어야 함)
    V[0, 12:15] = 1.10     # 노드 1 근접 LTE 버스트 (+20.5 dB, 비-Wi-Fi 구간!)
    V[1, 12:15] = 1.44
    V[2, 12:15] = 1.48
    res = S.detect_cellular_burst_ad8317(V, wifi_mask)
    assert res["detected"] is True and res["closest_node"] == 1

    ble = S.ble_zone_estimate([[0.06, 1.5, 1.95], [8.44, 3.8, 1.95], [4.25, 7.44, 1.10]], [-48.0, -72.0, -74.0])
    assert ble["strongest_node"] == 1


def test_ar_opa_wall_swivel_detents_and_autoverify():
    """Palm-12 벽면 AR-OPA 가 2.0m 이하 손 닿는 벽면 위치와 유효한 15° 클릭 각도를 반환하는지 검증."""
    obs = [O.ObstacleBox(2.45, 6.05, 0.0, 0.08, 0.9, 2.1, "metal", 20.0, 0.98)]
    roi = np.array([[x, y, 0.9] for x in np.linspace(1.2, 7.3, 5) for y in np.linspace(2.0, 6.5, 4)])
    rec = O.recommend_placement((8.5, 7.5, 2.8), obs, roi, k=3)
    assert len(rec["nodes"]) == 3
    for n in rec["nodes"]:
        assert n["pos_m"][2] <= 2.05, f"2.5m 스탠드 없이 손 닿는 높이여야 함: {n['pos_m']}"
        assert n["pan_deg"] in O.PAN_CLICKS_DEG
        assert n["tilt_deg"] in O.TILT_CLICKS_DEG
        assert "→" in n["guide_ko"]

    ok_v = O.auto_verify_nodes([2.1, 3.0, 2.4], [0.025, 0.031, 0.028])
    bad_v = O.auto_verify_nodes([2.1, 12.5, 2.4], [0.025, 0.085, 0.028])
    assert ok_v["ok"] is True and bad_v["ok"] is False
