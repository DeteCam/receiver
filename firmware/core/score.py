# -*- coding: utf-8 -*-
"""DeteCam 5.0 — 3단계 캐스케이드 카메라 판별기 + LTE/5G(AD8317×3)·BLE 확장 엔진.

교수님 질문 1("수많은 전자기기 중에 카메라만 어떻게 찾아낼 것인가?")에 대한 3단계 해답:
  1단계 (수동 MAC/PHY 핑거프린팅): 암호화 해제 없이 업링크/다운링크 비대칭, 프레임 크기,
        영상 연속/사진 버스트 박자, 핫스팟·숨김 SSID 짝, 카메라형 SSID 패턴으로 선별
  2단계 (인과적 장면 자극 확인 — 사적 공간용): CamLoPA(IEEE S&P 2025) / SnoopDog 원리 적용.
        5초 조명 On/Off 또는 보행-정지 시 화소 변화에 연동해 비트레이트가 동시 급변하는 기기만 확정
  3단계 (3D 방 경계 필터 + 크로스-스펙트럼 확장): 방 밖 송신원 자동 제외 +
        3노드 AD8317(1MHz~10GHz) 로그 검파기에서 동시각 Wi-Fi/BLE 로그를 교차 차감해
        유심(LTE/5G) 몰래카메라 업링크 버스트 탐지 및 3노드 구역 경보
"""
import re
import numpy as np

CAM_SSID_RE = [
    r"^NAX_", r"^FTYC", r"^BATC", r"^DDD-", r"^A9_", r"HDWIFICAM",
    r"HIDVCAM", r"^IPC[-_]", r"^CAM[-_]", r"MINI[-_ ]?CAM", r"^V380", r"^LOOKCAM",
]
HOTSPOT_RE = [r"IPHONE", r"GALAXY", r"ANDROIDAP", r"HOTSPOT", r"핫스팟", r"^SM-"]


def _match_any(pats, text):
    t = (text or "").upper()
    return any(re.search(p, t) for p in pats)


def is_random_mac(mac):
    try:
        return bool(int(mac.split(":")[0], 16) & 2)
    except Exception:
        return False


def causal_stimulus_score(rate_before, rate_during, rate_after):
    """2단계 인과적 자극 반응 검정 (조명 점멸 또는 보행-정지 시 VBR 코덱 비트레이트 급변 비율).
    일반 노트북/스마트폰/TV/공유기는 방 조명이나 보행과 전송량이 연동되지 않음.
    """
    b = max(float(np.mean(rate_before)), 1.0)
    d = float(np.mean(rate_during))
    a = max(float(np.mean(rate_after)), 1.0)
    ratio_up = d / b
    ratio_drop = d / a
    triggered = (ratio_up >= 1.45 and ratio_drop >= 1.30) or (b / max(d, 1.0) >= 1.45)
    return dict(triggered=bool(triggered), ratio_up=round(ratio_up, 2), ratio_drop=round(ratio_drop, 2))


def score_wifi_device(dev, room_dims=None):
    """단일 Wi-Fi 기기(MAC)의 카메라 의심 점수와 사람 친화적 근거 산출."""
    if dev.get("mine"):
        return dict(mac=dev["mac"], score=0, level="mine", inside_room=True, reasons=["내 기기로 등록되어 제외됨"])

    score = 0
    reasons = []

    up_bps = float(dev.get("uplink_kbps", 0.0))
    dn_bps = float(dev.get("downlink_kbps", 0.0))
    mean_len = int(dev.get("mean_len", 0))
    if up_bps > 120.0 and up_bps > 1.8 * max(dn_bps, 10.0) and mean_len >= 300:
        score += 2
        reasons.append(f"업링크 우세 (송신 {up_bps:.0f} kbps vs 수신 {dn_bps:.0f} kbps, 평균 {mean_len} B)")

    pat = dev.get("pattern", "quiet")
    if pat in ("steady", "periodic"):
        score += 2
        reasons.append("영상 스트리밍 또는 일정 주기 사진 송출 패턴")
    elif pat == "burst" and dev.get("photo_bursts", 0) >= 1:
        score += 1
        reasons.append(f"사진 크기 단발 업링크 버스트 ({dev.get('photo_bursts', 1)}회)")

    ssid = dev.get("ssid", "")
    bssid = dev.get("bssid", "")
    if dev.get("hidden_ssid") or is_random_mac(bssid) or _match_any(HOTSPOT_RE, ssid):
        score += 2 if up_bps > 80.0 else 1
        reasons.append("시설 공유기가 아닌 개인 핫스팟·숨김 Wi-Fi에 연결됨")

    if _match_any(CAM_SSID_RE, ssid):
        score += 3
        reasons.append(f"초소형 카메라 앱 특유의 Wi-Fi 이름 패턴({ssid})")

    if dev.get("single_stream_legacy"):
        score += 1
        reasons.append("초소형 카메라 칩 특징 (1×1 단일 안테나 구형 규격)")

    if dev.get("causal_stimulus_hit"):
        score += 3
        reasons.append("조명/움직임 변화에 전송량이 즉각 연동됨 (인과 자극 확인)")

    # 3단계: 3D 방 경계 물리 필터 (방 밖이면 경보 자동 차단 -> 옆 교실·옆 방 오경보 제거)
    inside = True
    pos = dev.get("pos_3d_m")
    if pos is not None and room_dims is not None:
        L, W, H = room_dims
        margin = 0.25
        inside = (-margin <= pos[0] <= L + margin and -margin <= pos[1] <= W + margin and -margin <= pos[2] <= H + margin)
        if not inside:
            reasons.append("3D 위치 추정 결과 방 경계 밖으로 판정되어 경보 제외")
            return dict(mac=dev["mac"], score=score, level="outside", inside_room=False, reasons=reasons)

    level = "camera" if score >= 5 else ("suspect" if score >= 3 else "ignore")
    return dict(mac=dev["mac"], score=score, level=level, inside_room=inside, reasons=reasons)


def detect_cellular_burst_ad8317(voltages_3nodes, wifi_active_mask, v_baseline=1.55, slope_mv_db=-22.0, thresh_db=10.0):
    """3노드 AD8317 (1MHz~10GHz, -22mV/dB 음의 기울기) 교차 차감 LTE/5G 버스트 탐지 + 구역 비교.
    voltages_3nodes: [3, T] 각 노드 RP2040/ADC 전압 시계열 (V)
    wifi_active_mask: [T] 동시각 AX210/BT 패킷 수신 여부 (bool) -> True 인 구간은 차감!
    """
    V = np.asarray(voltages_3nodes, dtype=float)
    mask = np.asarray(wifi_active_mask, dtype=bool)
    # 전압 강하량 -> 상대 전력 상승(dB) 환산
    power_rise_db = (v_baseline - V) * 1000.0 / abs(slope_mv_db)
    non_wifi = ~mask
    if not np.any(non_wifi):
        return dict(detected=False, max_rise_db=0.0, closest_node=None, node_peaks_db=[0.0, 0.0, 0.0])

    peaks = [round(float(np.max(power_rise_db[k, non_wifi])), 1) for k in range(V.shape[0])]
    best_node = int(np.argmax(peaks))
    max_rise = peaks[best_node]
    return dict(
        detected=bool(max_rise >= thresh_db),
        max_rise_db=max_rise,
        closest_node=best_node + 1 if max_rise >= thresh_db else None,
        node_peaks_db=peaks,
    )


def ble_zone_estimate(node_positions, rssi_3nodes, path_loss_n=2.4):
    """3노드 내장 Bluetooth 5.x RSSI 가중 중심 기반 무료 BLE 카메라 대략 구역 추정."""
    pos = np.asarray(node_positions, dtype=float)
    rssi = np.asarray(rssi_3nodes, dtype=float)
    # 상대 거리 역제곱 가중치
    w = 10.0 ** ((rssi - np.max(rssi)) / (10.0 * path_loss_n))
    w = w / np.sum(w)
    est = np.sum(pos * w[:, None], axis=0)
    return dict(
        zone_center_m=[round(float(v), 2) for v in est],
        strongest_node=int(np.argmax(rssi)) + 1,
        weights=[round(float(v), 2) for v in w],
    )
