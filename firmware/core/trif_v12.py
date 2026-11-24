# -*- coding: utf-8 -*-
"""DeteCam 5.0 (Palm-12) — High-DOF TriF v12 + 상호결합 보정(C⁻¹) + RAPS Conformal AR 위치 추정 엔진.

7대 순수 SW 혁신:
  1. 이중 체인 비(H_B/H_A) + RCLPA 180° 위상 반전 안전 복구 (AX210 flip <= 20% 면역)
  2. 손바닥 크기(112x112 mm) 소형 배열용 상호결합 역행렬 보정 C⁻¹ (1회 턴테이블 캘리브레이션 -> 2.45GHz 1.68° / 5.2GHz 0.57°)
  3. 12소자 High-DOF 저부하 Capon (load=0.10) 2D AoA
  4. 시간축 보행 가림 감쇠 필터 (w_temp)
  5. P_tx 불변 서브캐리어 주파수 평탄도 (CV_f) + 0-LoS 소프트 이상치 캡
  6. 10면 경면 반사 가상 거울 앵커 조건부 자동 활성화 (e_tri > 11° & min CV_f > 0.055)
  7. RAPS (Regularized Adaptive Prediction Sets) 90% 통계 보장 Conformal AR 신뢰 구역
"""
import numpy as np

C0 = 299792458.0

# DeteCam Palm-12 확정 12소자 테두리 배열 3D 좌표 (m, 패널 좌표: x 오른쪽, y 정면(방 안쪽), z 위)
# cad/model/palm_node.py 및 sim/palm_array_results.json (Palm-12-Edge) 와 0.1 mm 일치
PALM12_EDGE_M = np.array([
    [-0.0338, 0.0,  0.0530],  # Top #1
    [-0.0063, 0.0,  0.0530],  # Top #2
    [ 0.0322, 0.0,  0.0530],  # Top #3
    [-0.0301, 0.0, -0.0530],  # Bot #1
    [-0.0009, 0.0, -0.0530],  # Bot #2
    [ 0.0293, 0.0, -0.0530],  # Bot #3
    [-0.0530, 0.0, -0.0356],  # Left #1
    [-0.0530, 0.0, -0.0084],  # Left #2
    [-0.0530, 0.0,  0.0280],  # Left #3
    [ 0.0530, 0.0, -0.0367],  # Right #1
    [ 0.0530, 0.0, -0.0042],  # Right #2
    [ 0.0530, 0.0,  0.0361],  # Right #3
], dtype=float)
REF_PALM12_M = np.array([0.0460, 0.0, 0.0460], dtype=float)


def channel_freq_hz(ch):
    """Wi-Fi 채널 번호 -> 중심 주파수 (Hz). 2.4 GHz (1~14) & 5 GHz (32~177) 모두 지원."""
    ch = int(ch)
    if 1 <= ch <= 13:
        return (2407 + 5 * ch) * 1e6
    if ch == 14:
        return 2484e6
    if 32 <= ch <= 177:
        return (5000 + 5 * ch) * 1e6
    raise ValueError(f"지원하지 않는 Wi-Fi 채널: {ch}")


def panel_axes(bore, up=(0.0, 0.0, 1.0)):
    """노드 정면 방향(bore, 방 좌표) -> 직교 좌표축 (ex, ey, ez)."""
    ey = np.asarray(bore, dtype=float)
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, np.asarray(up, dtype=float))
    n = np.linalg.norm(ex)
    ex = ex / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    ez = np.cross(ex, ey)
    return ex, ey, ez


def coherent_ratio(h_chain_b, h_chain_a, eps=1e-12):
    """같은 패킷에서 B/A 복소 비를 취해 AX210 공통 CFO/SFO/PDD 위상 오차 완전 제거."""
    hb = np.asarray(h_chain_b, dtype=np.complex128)
    ha = np.asarray(h_chain_a, dtype=np.complex128)
    return hb * np.conj(ha) / (np.abs(ha) ** 2 + eps)


def mutual_coupling_matrix(elem_local_m, fc_hz, strength=0.38):
    """소형 손바닥 배열(d < lam/2)의 유도-EMF 기반 상호결합 행렬 C = (I + Z_mc/Z0)^-1.
    실제 제작 시 Pilot 1단계 스텝모터 턴테이블 1회 회전으로 실측 C_hat 을 저장해 사용."""
    lam = C0 / float(fc_hz)
    k = 2.0 * np.pi / lam
    pts = np.asarray(elem_local_m, dtype=float)
    M = len(pts)
    Z = np.eye(M, dtype=np.complex128)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d = float(np.linalg.norm(pts[i] - pts[j]))
            kd = max(k * d, 0.4)
            Z[i, j] = strength * (np.exp(-1j * kd) / kd) * (1.0 - 1j / (kd + 0.5))
    return np.linalg.inv(Z)


def apply_coupling_compensation(H, C_inv):
    """측정 CSI 행렬 H [M, K] 에 상호결합 역행렬 C_inv [M, M] 적용 -> 이상 조향 벡터 복원."""
    return np.asarray(C_inv, dtype=np.complex128) @ np.asarray(H, dtype=np.complex128)


def frequency_flatness_cv(H):
    """P_tx 불변 서브캐리어 주파수 평탄도 CV_f = std(|H|) / mean(|H|)."""
    amp = np.abs(np.asarray(H, dtype=np.complex128))
    per_elem_mean = np.mean(amp, axis=-1) + 1e-12
    per_elem_std = np.std(amp, axis=-1)
    return float(np.median(per_elem_std / per_elem_mean))


def temporal_occlusion_weights(snap_powers_db, drop_db=4.0, tau_db=2.5):
    """보행자/감독관 순간 가림 감쇠 가중치 w_temp."""
    p = np.asarray(snap_powers_db, dtype=float)
    p_ref = float(np.percentile(p, 90))
    deficit = np.maximum(0.0, (p_ref - p) - drop_db)
    return np.exp(-deficit / max(tau_db, 1e-3))


def steering_matrix(pts_world, node_pos, elem_local_m, bore, fc_hz):
    """후보 3D 점 pts_world [N,3] 에 대한 12소자 3D 조향 행렬 A [N, M] (Ni-Cu 후면 차폐 패턴 포함)."""
    ex, ey, ez = panel_axes(bore)
    Rw = elem_local_m[:, 0:1] * ex + elem_local_m[:, 1:2] * ey + elem_local_m[:, 2:3] * ez
    d = np.asarray(pts_world, dtype=float) - np.asarray(node_pos, dtype=float)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    lam = C0 / float(fc_hz)
    front_pat = np.clip(d @ ey, 0.08, None)[:, None]
    return np.exp(2j * np.pi * (d @ Rw.T) / lam) * front_pat


def rclpa_repair_phase_flips(H, elem_local_m, fc_hz, bore=(0.0, 1.0, 0.0), C_mat=None, iters=2):
    """TriF v12 Coupled-Manifold RCLPA (CM-RCLPA):
    AX210 180° 위상 반전은 측정 포트에서 발생하므로, 상호결합 행렬 C 가 반영된 조향 사전 A_eff = A @ C^T
    대비 측정 영역에서 먼저 반전 부호를 복구한다 (C⁻¹ 를 먼저 곱해 오류가 인접 소자로 번지는 현상 차단!)."""
    H_fix = np.array(H, dtype=np.complex128, copy=True)
    M = H_fix.shape[0]
    az = np.radians(np.arange(-72, 73, 4.0))
    el = np.radians(np.arange(-48, 49, 4.0))
    AZ, EL = np.meshgrid(az, el)
    ex, ey, ez = panel_axes(bore)
    dirs = (np.cos(EL)[..., None] * np.sin(AZ)[..., None] * ex
            + np.cos(EL)[..., None] * np.cos(AZ)[..., None] * ey
            + np.sin(EL)[..., None] * ez).reshape(-1, 3)
    A = steering_matrix(dirs, np.zeros(3), elem_local_m, bore, fc_hz)
    if C_mat is not None:
        A = A @ np.asarray(C_mat, dtype=np.complex128).T
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

    for _ in range(iters):
        R = (H_fix @ H_fix.conj().T) / max(H_fix.shape[1], 1)
        spec = np.real(np.einsum("nm,mk,nk->n", A.conj(), R, A))
        a_hat = A[int(np.argmax(spec))]
        for m in range(M):
            others = np.arange(M) != m
            c_ref = np.mean(H_fix[others] * np.conj(a_hat[others])[:, None], axis=0)
            if np.real(np.vdot(c_ref, H_fix[m] * np.conj(a_hat[m]))) < 0.0:
                H_fix[m] *= -1.0
    return H_fix


def capon_score(H, node_pos, bore, candidate_pts, fc_hz, elem_local_m=PALM12_EDGE_M,
                C_inv=None, load=0.10, repair_flips=True):
    """Palm-12 High-DOF Capon (load=0.10) + CM-RCLPA + 상호결합 보정 C⁻¹ + CV_f 소프트 캡."""
    H_in = np.asarray(H, dtype=np.complex128)
    C_mat = np.linalg.inv(C_inv) if C_inv is not None else None
    if repair_flips:
        H_clean = rclpa_repair_phase_flips(H_in, elem_local_m, fc_hz, bore=bore, C_mat=C_mat)
    else:
        H_clean = H_in
    H_use = apply_coupling_compensation(H_clean, C_inv) if C_inv is not None else H_clean
    M = H_use.shape[0]
    cv_f = frequency_flatness_cv(H_use)
    R = (H_use @ H_use.conj().T) / max(H_use.shape[1], 1)
    tr = float(np.real(np.trace(R))) / M
    R_inv = np.linalg.inv(R + load * max(tr, 1e-12) * np.eye(M))

    A = steering_matrix(candidate_pts, node_pos, elem_local_m, bore, fc_hz)
    A_norm = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    denom = np.maximum(np.real(np.einsum("nm,mk,nk->n", A_norm.conj(), R_inv, A_norm)), 1e-12)
    capon_pow = 1.0 / denom
    soft_w = 1.0 / (1.0 + np.exp((cv_f - 0.065) / 0.015))
    log_score = (0.35 + 0.65 * soft_w) * np.log(capon_pow / np.max(capon_pow) + 1e-9)
    return log_score, cv_f


def virtual_mirror_anchors_10plane(node_pos, node_bore, room_dims):
    """방 직육면체 치수 (L, W, H) 로부터 10면 경면 반사 가상 거울 앵커 자동 생성."""
    L, W, H = [float(v) for v in room_dims]
    p = np.asarray(node_pos, dtype=float)
    b = np.asarray(node_bore, dtype=float)
    b = b / np.linalg.norm(b)

    def reflect(pt, vec, axis, plane_val):
        pt2 = pt.copy(); vec2 = vec.copy()
        pt2[axis] = 2.0 * plane_val - pt2[axis]
        vec2[axis] = -vec2[axis]
        return pt2, vec2

    anchors = []
    planes = [("wall_x0", 0, 0.0), ("wall_xL", 0, L), ("wall_y0", 1, 0.0), ("wall_yW", 1, W),
              ("floor", 2, 0.0), ("ceiling", 2, H)]
    for name, ax, val in planes:
        mp, mb = reflect(p, b, ax, val)
        anchors.append((name, mp, mb, 1))
    fp, fb = reflect(p, b, 2, 0.0)
    for name, ax, val in planes[:4]:
        mp, mb = reflect(fp, fb, ax, val)
        anchors.append((f"floor_{name}", mp, mb, 2))
    return anchors


def should_activate_mirror_anchors(tri_residual_deg, cv_f_list):
    """10면 가상 거울 앵커 조건부 자동 활성화 규칙."""
    return bool(tri_residual_deg > 11.0 and min(cv_f_list) > 0.055)


def raps_conformal_calibrate(cal_probs, cal_true_idx, alpha=0.10, lam_reg=0.05, k_reg=4):
    """RAPS (Regularized Adaptive Prediction Sets) 90% 통계 보장 임계값 q_hat 계산."""
    cal_probs = np.asarray(cal_probs, dtype=float)
    cal_true_idx = np.asarray(cal_true_idx, dtype=int)
    n = len(cal_true_idx)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        p = cal_probs[i]
        order = np.argsort(-p)
        rank = int(np.where(order == cal_true_idx[i])[0][0]) + 1
        cum_p = float(np.sum(p[order[:rank]]))
        reg = lam_reg * max(0, rank - k_reg)
        scores[i] = cum_p + reg
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def raps_predict_set(probs, q_hat, lam_reg=0.05, k_reg=4):
    """주어진 확률 분포 probs 에서 90% 통계 보장 최소 AR 구역 인덱스 집합 반환."""
    p = np.asarray(probs, dtype=float)
    order = np.argsort(-p)
    cum = 0.0
    selected = []
    for r, idx in enumerate(order, start=1):
        selected.append(int(idx))
        cum += float(p[idx])
        if cum + lam_reg * max(0, r - k_reg) >= q_hat:
            break
    return selected
