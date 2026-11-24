# -*- coding: utf-8 -*-
"""DeteCam 5.0 (Palm-12) — AR-OPA 벽면 부착·스위블 각도 최적 배치 엔진 + Auto-Verify.

대형 2.5m 스탠드 없이 손이 닿는 벽면(1.95m / 1.10m)에 자석·재사용 나노겔로 붙이고
직교 2축 짐벌의 좌우 Pan(±30°, 15° 클릭) · 상하 Tilt(0°/-15°/-25°) 각도만 딸깍 맞춰
비전문가도 20초 안에 수학적 최적 배치를 완성하도록 안내:
  1. Mixed-LoS/NLoS 3D AoA Fisher Information Matrix (arXiv:2604.00863) 서브모듈러 + 1-Swap 최적화
  2. Palm-12 11.4mm 후면 Ni-Cu 벽면 차폐 + 측면 거울/금속 스침 반사 페널티 (arXiv:2412.13208)
  3. 3초 상호 전파 교정(Auto-Verify): 숨은 벽 내부 철근/배관 반사 감지 시 옆으로 40cm 이동 안내
"""
import numpy as np

PAN_CLICKS_DEG = (-30, -15, 0, 15, 30)
TILT_CLICKS_DEG = (0, -15, -25)


class ObstacleBox:
    """축 정렬 3D 장애물/반사체 [x0, x1, y0, y1, z0, z1]."""
    def __init__(self, x0, x1, y0, y1, z0, z1, kind="furniture", loss_db=6.0, refl=0.3):
        self.b = np.array([x0, x1, y0, y1, z0, z1], dtype=float)
        self.kind = kind
        self.loss_db = float(loss_db)
        self.refl = float(refl)


def _hits_box(p0, p1, box):
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        lo, hi = box.b[2 * i], box.b[2 * i + 1]
        if abs(d[i]) < 1e-9:
            if p0[i] < lo or p0[i] > hi:
                return False
        else:
            t1, t2 = (lo - p0[i]) / d[i], (hi - p0[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return False
    return tmax > 0.02 and tmin < 0.98


def _dist_to_box(p, box):
    q = np.array([np.clip(p[i], box.b[2 * i], box.b[2 * i + 1]) for i in range(3)])
    return float(np.linalg.norm(p - q))


def swivel_boresight(wall_normal, pan_deg, tilt_deg):
    """벽면 법선과 직교 짐벌 (pan_deg, tilt_deg) 로부터 3D 수신 정면 단위 벡터 계산."""
    n = np.asarray(wall_normal, dtype=float)
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(n, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    ap, at = np.radians(pan_deg), np.radians(tilt_deg)
    horiz = np.cos(ap) * n + np.sin(ap) * right
    bore = np.cos(at) * horiz + np.sin(at) * up
    return bore / np.linalg.norm(bore)


def snap_to_detent(wall_normal, target_dir):
    """목표 방향과 가장 가까운 15° 클릭 스위블 각도(Pan ±30°, Tilt 0°/-15°/-25°) 반환."""
    td = np.asarray(target_dir, dtype=float)
    td = td / max(np.linalg.norm(td), 1e-9)
    best_dot = -2.0
    best = (0, 0, swivel_boresight(wall_normal, 0, 0))
    for pan in PAN_CLICKS_DEG:
        for tilt in TILT_CLICKS_DEG:
            b = swivel_boresight(wall_normal, pan, tilt)
            d = float(np.dot(b, td))
            if d > best_dot:
                best_dot = d
                best = (pan, tilt, b)
    return best


def _aoa_fim(p_node, bore, p_target, obstacles):
    v = p_target - p_node
    r = max(float(np.linalg.norm(v)), 0.3)
    u = v / r
    cos_a = float(np.dot(u, bore))
    if cos_a < 0.15:
        sig = np.radians(55.0)
        return (np.eye(3) - np.outer(u, u)) / ((r * sig) ** 2), False
    sig_deg = 2.6 / max(cos_a, 0.28) * (1.0 + 0.035 * r)
    loss_db = sum(ob.loss_db for ob in obstacles if _hits_box(p_node, p_target, ob))
    is_los = loss_db < 3.0
    if not is_los:
        sig_deg *= (1.0 + 0.28 * loss_db)
    sig = np.radians(min(sig_deg, 55.0))
    return (np.eye(3) - np.outer(u, u)) / ((r * sig) ** 2), is_los


def _guide_text(wall_ko, z_m, pan_deg, tilt_deg):
    pan_txt = "정면(0°)" if pan_deg == 0 else (f"오른쪽 {abs(pan_deg)//15}칸(+{pan_deg}°)" if pan_deg > 0 else f"왼쪽 {abs(pan_deg)//15}칸({pan_deg}°)")
    tilt_txt = "수평(0°)" if tilt_deg == 0 else f"아래 {1 if tilt_deg == -15 else 2}칸({tilt_deg}°)"
    mount_txt = "자석·나노겔 벽부착" if wall_ko != "교탁" else "요크 접어 교탁 거치"
    return f"{wall_ko} {z_m:.1f}m ({mount_txt}) → 좌우 {pan_txt}, 상하 {tilt_txt}"


def recommend_placement(room_dims, obstacles, roi_pts, k=2, mirror_aware=True):
    """DeteCam 6.0 AR-OPA-2: AR 스캔 방·장애물로부터 최적 2노드(k=2) 벽면 부착 위치 + Pan/Tilt 클릭 각도 계산.
    직접 가시선(LoS)뿐 아니라 L자 코너·중앙 기둥·칸막이 너머 1차 가상 거울 반사 가시성까지 동시 최적화!"""
    L, W, H = [float(v) for v in room_dims]
    roi = np.asarray(roi_pts, dtype=float)
    center = roi.mean(axis=0)
    z_high = min(2.00, H - 0.35)
    z_mid = 1.10
    off = 0.06

    xs = np.linspace(0.45, L - 0.45, max(4, min(7, int(L / 1.4))))
    ys = np.linspace(0.45, W - 0.45, max(4, min(7, int(W / 1.4))))
    specs = [(x, off, (0.0, 1.0, 0.0), "앞벽") for x in xs]
    specs += [(x, W - off, (0.0, -1.0, 0.0), "뒷벽") for x in xs]
    specs += [(off, y, (1.0, 0.0, 0.0), "왼벽") for y in ys[1:-1]]
    specs += [(L - off, y, (-1.0, 0.0, 0.0), "오른벽") for y in ys[1:-1]]
    if W >= 6.5:
        specs.append((L / 2.0, 1.15, (0.0, 1.0, 0.0), "교탁"))

    cands = []
    for x, y, normal, wall_ko in specs:
        for z in (z_high, z_mid):
            if wall_ko == "교탁" and z > 1.3:
                continue
            p = np.array([x, y, z], dtype=float)
            if any(ob.b[0] - 0.03 <= x <= ob.b[1] + 0.03 and ob.b[2] - 0.03 <= y <= ob.b[3] + 0.03
                   and ob.b[4] - 0.03 <= z <= ob.b[5] + 0.03 for ob in obstacles):
                continue
            pan, tilt, bore = snap_to_detent(normal, center - p)
            refl_pen = 0.0
            for ob in obstacles:
                if ob.kind in ("metal", "mirror"):
                    d_ob = _dist_to_box(p, ob)
                    if d_ob < 0.45:
                        ob_c = np.array([0.5*(ob.b[0]+ob.b[1]), 0.5*(ob.b[2]+ob.b[3]), 0.5*(ob.b[4]+ob.b[5])])
                        rel = ob_c - p
                        behind = float(np.dot(rel / max(np.linalg.norm(rel), 1e-6), bore)) < -0.35
                        refl_pen += (0.20 if behind else 0.70) * ob.refl * ((0.45 - d_ob) / 0.45) ** 2
            cands.append((p, bore, refl_pen, wall_ko, pan, tilt))

    nc, nr = len(cands), len(roi)
    F = np.zeros((nc, nr, 3, 3))
    LoS = np.zeros((nc, nr), dtype=bool)
    Vis = np.zeros((nc, nr), dtype=bool)
    planes = [(0, 0.0), (0, L), (1, 0.0), (1, W)]

    for i, c in enumerate(cands):
        p, bore = c[0], c[1]
        for j, q in enumerate(roi):
            F[i, j], LoS[i, j] = _aoa_fim(p, bore, q, obstacles)
            Vis[i, j] = LoS[i, j]
            if mirror_aware and not LoS[i, j]:
                # 1차 가상 거울 반사 경로가 사각지대(L자 코너·기둥 뒤)를 비추는지 검사!
                for ax, pval in planes:
                    q_img = q.copy(); q_img[ax] = 2.0 * pval - q_img[ax]
                    den = q_img[ax] - p[ax]
                    if abs(den) < 1e-5:
                        continue
                    t = (pval - p[ax]) / den
                    if not (0.02 < t < 0.98):
                        continue
                    hit = p + t * (q_img - p)
                    if not (0.0 <= hit[0] <= L and 0.0 <= hit[1] <= W and 0.0 <= hit[2] <= H):
                        continue
                    um = (hit - p) / max(float(np.linalg.norm(hit - p)), 1e-6)
                    cm = float(np.dot(um, bore))
                    if cm < 0.20:
                        continue
                    mloss = sum(ob.loss_db for ob in obstacles if _hits_box(p, hit, ob) or _hits_box(hit, q, ob))
                    if mloss < 4.5:
                        rm = max(float(np.linalg.norm(q_img - p)), 0.4)
                        sig_m = np.radians(4.0 / max(cm, 0.30) * (1.0 + 0.03 * rm))
                        um_r = um.copy(); um_r[ax] = -um_r[ax]
                        F[i, j] += 0.45 * (np.eye(3) - np.outer(um_r, um_r)) / ((rm * sig_m) ** 2)
                        Vis[i, j] = True
                        break

    def eval_set(idxs):
        Fs = F[idxs].sum(axis=0) + 1e-4 * np.eye(3)[None, :, :]
        _, ld = np.linalg.slogdet(Fs)
        invF = np.linalg.inv(Fs)
        crlb = np.sqrt(np.clip(np.trace(invF, axis1=1, axis2=2), 1e-9, 100.0))
        p90 = float(np.percentile(crlb, 90))
        vis_cov = float(np.mean(Vis[idxs].sum(axis=0) >= min(2, len(idxs))))
        vis_any = float(np.mean(Vis[idxs].any(axis=0)))
        rpen = sum(cands[i][2] for i in idxs)
        min_d = min(
            (float(np.linalg.norm(cands[idxs[a]][0] - cands[idxs[b]][0]))
             for a in range(len(idxs)) for b in range(a + 1, len(idxs))),
            default=2.0,
        )
        z_div = 0.35 if (len(idxs) >= 2 and max(cands[i][0][2] for i in idxs) - min(cands[i][0][2] for i in idxs) >= 0.65) else 0.0
        wall_div = 0.35 * len(set(cands[i][3] for i in idxs))
        return float(np.mean(ld)) - 1.4 * p90 + 1.6 * vis_cov + 1.8 * vis_any - 1.0 * rpen - 3.0 * max(0.0, 1.5 - min_d) + z_div + wall_div, p90, vis_any

    chosen = []
    for _ in range(k):
        best_i = max((i for i in range(nc) if i not in chosen), key=lambda i: eval_set(chosen + [i])[0])
        chosen.append(best_i)

    best_val, p90, vis_any = eval_set(chosen)
    improved = True
    while improved:
        improved = False
        for pos in range(k):
            for i in range(nc):
                if i in chosen:
                    continue
                trial = chosen.copy()
                trial[pos] = i
                v, p90_t, vis_t = eval_set(trial)
                if v > best_val + 1e-4:
                    best_val, p90, vis_any, chosen, improved = v, p90_t, vis_t, trial, True

    return dict(
        nodes=[
            dict(
                id=idx + 1,
                wall=cands[c][3],
                pos_m=[round(float(v), 2) for v in cands[c][0]],
                pan_deg=int(cands[c][4]),
                tilt_deg=int(cands[c][5]),
                aim=[round(float(v), 2) for v in cands[c][1]],
                mount="벽면 자석·나노겔 스위블" if cands[c][3] != "교탁" else "요크 탁상 거치",
                guide_ko=_guide_text(cands[c][3], float(cands[c][0][2]), int(cands[c][4]), int(cands[c][5])),
            )
            for idx, c in enumerate(chosen)
        ],
        expected_p90_crlb_m=round(p90, 3),
        virtual_mirror_coverage_pct=round(100.0 * vis_any, 1),
    )


def auto_verify_nodes(mutual_aoa_err_deg, ambient_cv_f, max_err_deg=8.0, max_cv_f=0.06):
    """배치 직후 3초 상호 전파 교정(Auto-Verify): 숨은 벽 내부 철근/배관 반사 감지 및 40cm 이동 안내."""
    warnings = []
    for i, (err, cv) in enumerate(zip(mutual_aoa_err_deg, ambient_cv_f), start=1):
        if err > max_err_deg or cv > max_cv_f:
            warnings.append(
                f"노드 {i}번 주변 숨은 금속 반사 강함 (상반오차 {err:.1f}°, CV_f={cv:.3f}) -> 옆으로 40 cm 이동 권장"
            )
    score = max(0, min(100, int(100 - 3.5 * np.mean(mutual_aoa_err_deg) - 250 * max(0.0, np.mean(ambient_cv_f) - 0.03))))
    return dict(ok=len(warnings) == 0, placement_score=score, warnings=warnings)
