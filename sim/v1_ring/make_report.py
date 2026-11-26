"""Generate AGENTS.md section 5.5 compliant Korean V1 simulation report from results/v1_summary.json.
Usage:
  python3 sim/v1_ring/make_report.py
"""
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IN_JSON = os.path.join(THIS_DIR, "results", "v1_summary.json")
OUT_MD = os.path.join(THIS_DIR, "RESULTS_20261125.md")

SPACE_KO = {
    "classroom_a": "교실 A (앞벽 칠판, 40석+감독2) [Dev]",
    "lodging_a": "숙소 A (화장대·욕실 거울, 2명) [Dev]",
    "restroom_a": "화장실 A (금속 칸막이·거울, 2명) [Dev]",
    "locker_a": "탈의실 A (양벽 금속 사물함·전신거울, 6명) [Dev]",
    "lecture_a": "강의실 A (평면 100석·84명·앞벽 칠판) [Eval/미학습공간]",
    "lecture_b": "강의실 B (계단식·84명·옆벽 칠판) [Eval/미학습공간]",
    "bathhouse_a": "목욕탕 A (물 표면·샤워 줄거울, 7명) [Eval/미학습공간]",
    "bathhouse_b": "목욕탕 B (탕 이동·반대벽 거울, 8명) [Eval/미학습공간]",
    "classroom_b": "교실 B (옆벽 칠판·40석+감독2) [Eval/변형]",
    "lodging_b": "숙소 B (옆벽 거울, 1명) [Eval/변형]",
    "restroom_b": "화장실 B (합판칸막이·반대벽 거울, 3명) [Eval/변형]",
    "locker_b": "탈의실 B (중앙 아일랜드 사물함, 5명) [Eval/변형]",
    "worst_metal": "최악 금속방 (금속벽·같은벽/가림 카메라) [Eval/최악]",
}


def _fmt(s):
    if not s or s.get("n", 0) == 0:
        return "—"
    return f"{s['mean']:.1f}° / {s['med']:.1f}° / {s['p90']:.1f}° / {s['p95']:.1f}°"


def merge_parts():
    res_dir = os.path.dirname(IN_JSON)
    merged = {}
    if os.path.exists(IN_JSON):
        with open(IN_JSON, "r", encoding="utf-8") as f:
            merged = json.load(f)
    to_remove = []
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("part_") and fn.endswith(".json"):
            p = os.path.join(res_dir, fn)
            with open(p, "r", encoding="utf-8") as f:
                sub = json.load(f)
            for k, v in sub.items():
                if not v:
                    continue
                if isinstance(v, dict):
                    merged.setdefault(k, {})
                    for k2, v2 in v.items():
                        if isinstance(v2, dict) and isinstance(merged[k].get(k2), dict):
                            merged[k][k2].update(v2)
                        else:
                            merged[k][k2] = v2
                else:
                    merged[k] = v
            to_remove.append(p)
    with open(IN_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    for p in to_remove:
        os.remove(p)
    return merged


def main():
    R = merge_parts()

    L = []
    L.append("# V1 스위치 고리 시뮬레이션 결과 보고서 (2026-11-25)")
    L.append("")
    L.append("> **시뮬레이션이며 실측이 아니다 ([docs/AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md) 5.4 · 5.5절).**")
    L.append("> 이 보고서의 모든 숫자는 **계산값**이며, 목표 달성이나 실제 성능 근거로 쓰지 않는다. 설계 선택(`SPEC-002` vs `SPEC-002+` vs `SPEC-002++`)과 V2 최소 구매 시험(3~5만 원)으로 넘어갈 근거로만 쓴다.")
    L.append("")
    L.append("## 1. 실험 설정 · 도구 · 완화 조건 (사전 계획서: [`plan_20261125.md`](../plan_20261125.md))")
    L.append("")
    L.append("- **광선 추적기**: `Sionna RT 2.2.0` (`Mitsuba 3.9.1`, `Dr.Jit 1.5.0`, LLVM CPU polarized) — 반사 4차(`max_depth=4`), 투과(`refraction=True`), 모서리 회절(`diffraction=True, edge_diffraction=True`), 확산 산란(`diffuse_reflection=True`), `samples_per_src=100,000`, 쌍당 상위 80개 경로 보존(전력의 94.2~99.2%).")
    L.append("- **안테나 전파 해석**: `nec2c 1.3.1-2` (구리 `5.8e7 S/m`, 90×90×35 mm 금속 박스 포함, 비선택 포트 `50 Ω` 흡수형 vs 개방 반사형 대조, `2437 MHz` · `5220 MHz`).")
    L.append("- **신호 생성기 / 추정기 분리**: `gen_scenes.py` + `channel_sim.py`(실제 방 · 사람 · 미등록 금속 물체 · NEC2 응답 · AX210 $\\pi$-반전 생성)와 `estimators.py`(오차가 섞인 방 크기 · 수신기 위치/각도 · 눈에 보이는 대형 반사체 1~2면 · NEC2 1회 보정표만 입력)를 완전 분리.")
    L.append("- **공간 · 표본 · 난수**: `AGENTS.md` 5.2절 6종 공간 × 2변형 + 최악 금속방 = **13개 장면**, 장면당 카메라 40곳 × 난수 3개(`seed=0,1,2`) × 4자리 = 장면 · 대역당 480회 단일 자리 평가(평가 9장면 합계 대역당 4,320회 단일 자리 / 1,080회 2자리 쌍 평가). 개발용 4장면(`classroom_a`, `lodging_a`, `restroom_a`, `locker_a`)과 평가용 9장면(미학습 공간 `강의실` · `목욕탕` 4장면 + 변형 B 4장면 + `worst_metal`)을 엄격히 분리.")
    L.append("- **현실 오차 조건**: 방 크기 `σ=15 cm`, 수신기 위치 `σ=15 cm`, 수신기 각도 `σ=5°`, 포트 위상 지터 `σ=4.5°`, 진폭 `0.4 dB`, `SNR=20 dB`, 비직접파 35% 인체 동적 위상 변동(`N_sweep=10`), AX210 패킷당 무작위 $\\pi$ 반전 + 수동 RF 부호 앵커(`|α|=3.5`, Dev `classroom_a`에서 결정, 보정 오차 3°/0.25 dB 포함).")
    L.append("- **완화 조건 목록**: ① 각 대역 공진 반파장 다이폴 가정(실물 듀얼밴드 매칭 편차는 V3 실측 필요), ② AX210 RX0/RX1 독립 AGC 이득 단차 미포함(문서상 미확인 항목 → V2 시험 3에서 필수 실측), ③ `DiffLoc-Proxy`는 동일 Sionna RT 경로 위 단일 안테나 진폭 회절 모형(360° 전방위 허용으로 DiffLoc에 유리한 조건이며 실제 DiffLoc 장비 실측이 아님).")
    L.append("")

    # Section 2: Overall Held-Out Eval Summary
    L.append("## 2. 평가 장면(Held-Out 9개 장면 전체) 비교 요약 — 평균 / 중앙값 / P90 / P95 (계산값)")
    L.append("")
    for band, btitle in (("2g4", "2.4 GHz (20 MHz)"), ("5g", "5 GHz (20 MHz)")):
        if band not in R.get("eval_overall", {}):
            continue
        L.append(f"### 2.{1 if band == '2g4' else 2} {btitle} — 평가 9장면 통합 (장면별 결과는 3장)")
        L.append("")
        L.append("| 방법 | 1자리 (평균/중앙/P90/P95) | 2자리 인접 `(0,1)` (평균/중앙/P90/P95) | 2자리 대각 `(0,3)` (평균/중앙/P90/P95) | 최악 방 1자리 평균 (P95) | 최악 방 2자리`(0,1)` 평균 (P95) | 부호 복원률 | 저신뢰도 비율(>45°) |")
        L.append("|---|---|---|---|---|---|---|---|")
        ev_by_sc = R["eval"].get(band, {})
        for m, v in R["eval_overall"][band].items():
            # find worst room among eval scenes for this method
            w1_sc, w1_m, w1_p = "—", -1.0, -1.0
            w2_sc, w2_m, w2_p = "—", -1.0, -1.0
            for sc_id, sc_d in ev_by_sc.items():
                if m not in sc_d:
                    continue
                m1 = sc_d[m]["one_node"]["mean"]
                if m1 > w1_m:
                    w1_sc, w1_m, w1_p = sc_id, m1, sc_d[m]["one_node"]["p95"]
                if sc_d[m]["fused_adj"]["n"] > 0:
                    m2 = sc_d[m]["fused_adj"]["mean"]
                    if m2 > w2_m:
                        w2_sc, w2_m, w2_p = sc_id, m2, sc_d[m]["fused_adj"]["p95"]
            w1_str = f"`{w1_sc}` {w1_m:.1f}° ({w1_p:.1f}°)" if w1_m >= 0 else "—"
            w2_str = f"`{w2_sc}` {w2_m:.1f}° ({w2_p:.1f}°)" if w2_m >= 0 else "—"
            unrel = f"{v['unreliable_pct']:.1f}%" if v["unreliable_pct"] is not None else "—"
            L.append(
                f"| **{m}** | {_fmt(v['one_node'])} | {_fmt(v['fused_adj'])} | {_fmt(v['fused_diag'])} | "
                f"{w1_str} | {w2_str} | {v['sign_acc_pct']:.2f}% | {unrel} |"
            )
        L.append("")

    # Section 3: Per-space breakdown (mandatory per AGENTS.md 5.2!)
    L.append("## 3. 공간별 상세 결과 (전체 13개 장면 개별 보고 — 계산값)")
    L.append("")
    for band, btitle in (("2g4", "2.4 GHz (20 MHz)"), ("5g", "5 GHz (20 MHz)")):
        L.append(f"### 3.{1 if band == '2g4' else 2} {btitle} 공간별 표")
        L.append("")
        L.append("| 장면 | Bartlett 1자리 | DiffLoc-Proxy 1자리 | SPEC-002(NaiveFlip) 1자리 | SPEC-002(NoFlip) 1자리 / 2자리`(0,1)` | **SPEC-002+ 1자리 (최악자리)** | **SPEC-002+ 2자리`(0,1)` / `(0,3)`** | SPEC-002++(12+1) 2자리`(0,1)` |")
        L.append("|---|---|---|---|---|---|---|---|")
        for split in ("dev", "eval"):
            for sc, d in R.get(split, {}).get(band, {}).items():
                b1 = d["Bartlett"]["one_node"]
                dp = d["DiffLoc-Proxy"]["one_node"]
                nf = d["SPEC-002-NaiveFlip"]["one_node"]
                s2_1, s2_2 = d["SPEC-002-NoFlip"]["one_node"], d["SPEC-002-NoFlip"]["fused_adj"]
                sp_1, sp_wp = d["SPEC-002+"]["one_node"], d["SPEC-002+"]["worst_pos"]
                sp_a, sp_d = d["SPEC-002+"]["fused_adj"], d["SPEC-002+"]["fused_diag"]
                pp_a = d["SPEC-002++(12+1)"]["fused_adj"]
                L.append(
                    f"| {SPACE_KO.get(sc, sc)} | {b1['mean']:.1f}° (P95 {b1['p95']:.1f}°) | {dp['mean']:.1f}° (P95 {dp['p95']:.1f}°) | "
                    f"{nf['mean']:.1f}° (P95 {nf['p95']:.1f}°) | {s2_1['mean']:.1f}° / {s2_2['mean']:.1f}° (P95 {s2_2['p95']:.1f}°) | "
                    f"**{sp_1['mean']:.1f}°** (P95 {sp_1['p95']:.1f}°, 최악 {sp_wp['mean']:.1f}°) | "
                    f"**{sp_a['mean']:.1f}°** (P95 {sp_a['p95']:.1f}°) / {sp_d['mean']:.1f}° (P95 {sp_d['p95']:.1f}°) | "
                    f"{pp_a['mean']:.1f}° (P95 {pp_a['p95']:.1f}°) |"
                )
        L.append("")

    # Section 4: Ablation Ladder L0 -> L5
    L.append("## 4. 조건 변화 사다리 표 (AGENTS.md 5.1.8 — 이상 조건 L0 → 현실 조건 L5, 계산값)")
    L.append("")
    lad_desc = {
        "L0": "빈 방, 1차 반사만, 이상적 등방성 안테나, 오차 0",
        "L1": "+ 4차 반사 · 회절 · 산란 · 투과",
        "L2": "+ 가구 · 칠판 · 사물함 · 움직이는 사람 42명",
        "L3": "+ NEC2 실제 안테나 + 금속 박스 (기존 `dz=+65`)",
        "L4": "+ 현실 오차(방/위치 15cm · 각도 5° · 지터 4.5° · SNR 20dB · AX210 π-반전+앵커) — 기존 SPEC-002",
        "L5": "+ `SPEC-002+` 개선(중심기준 9소자 · `dz=0` · 칠판 반영 · 강건 우도 · 2자리 융합)",
    }
    for key in ("classroom_a_2g4", "classroom_b_2g4", "classroom_a_5g", "classroom_b_5g"):
        if key not in R.get("ladder", {}):
            continue
        L.append(f"### `{key}`")
        L.append("")
        L.append("| 단계 | 조건 | 1자리 (평균/중앙/P90/P95) | 2자리 `(0,1)` (평균/중앙/P90/P95) |")
        L.append("|---|---|---|---|")
        for st in ("L0", "L1", "L2", "L3", "L4", "L5"):
            if st in R["ladder"][key]:
                v = R["ladder"][key][st]
                L.append(f"| **{st}** | {lad_desc[st]} | {_fmt(v['one_node'])} | {_fmt(v['fused_adj'])} |")
        L.append("")

    # Section 5: Failed / No-gain ablations
    L.append("## 5. 실패 · 효과 없음 및 요소 분해 대조 표 (AGENTS.md 5.1.10 — 평가 4장면 통합, 계산값)")
    L.append("")
    for band, btitle in (("2g4", "2.4 GHz"), ("5g", "5 GHz")):
        if band not in R.get("ablations", {}):
            continue
        L.append(f"### {btitle} (`classroom_b`, `lecture_a`, `bathhouse_a`, `worst_metal`)")
        L.append("")
        L.append("| 변형 | 1자리 (평균/중앙/P90/P95) | 2자리 `(0,1)` (평균/중앙/P90/P95) | 비고 |")
        L.append("|---|---|---|---|")
        for mn, v in R["ablations"][band].items():
            L.append(f"| {mn} | {_fmt(v['one_node'])} | {_fmt(v['fused_adj'])} | 부호복원 {v['sign_acc_pct']:.2f}% |")
        L.append("")

    # Section 6: 5 GHz BW comparison
    if R.get("bw5g"):
        L.append("## 6. 5 GHz 채널 폭별 비교 (`SPEC-002+`, 평가 4장면, 계산값)")
        L.append("")
        L.append("| 채널 폭 | 1자리 (평균/중앙/P90/P95) | 2자리 `(0,1)` (평균/중앙/P90/P95) |")
        L.append("|---|---|---|")
        for bw_k, v in R["bw5g"].items():
            L.append(f"| {bw_k} | {_fmt(v['one_node'])} | {_fmt(v['fused_adj'])} |")
        L.append("")

    # Section 6.5: Post-hoc candidates discovered from Section 5 ablations (strictly separated per AGENTS.md 5.1.3)
    if R.get("posthoc"):
        L.append("## 6.5 사후 발견 후보 (Post-hoc — AGENTS.md 5.1.3절 준수, 사전 등록 주 표와 분리)")
        L.append("")
        L.append("> 아래 조합은 5장 대조군 표를 본 뒤 발견한 사후 후보(`Conj-Avg` + `6면 단순 방 지도` + `D=85 mm` vs `D=100 mm`)이다. 사전 등록 `SPEC-002+` 주 결과(2·3장)를 덮어쓰지 않고 별도로 적으며, 최종 채택은 V2 통과 뒤 V3 실물 보정·실측에서 확인한다.")
        L.append("")
        L.append("| 분할 · 대역 | 후보 | 1자리 (평균/중앙/P90/P95) | 2자리 인접 `(0,1)` (평균/중앙/P90/P95) | 2자리 대각 `(0,3)` (평균/중앙/P90/P95) | 최악 방 1자리 평균 (P95) | 비고 |")
        L.append("|---|---|---|---|---|---|---|")
        for split_key, split_ko in (("dev", "Dev 4장면"), ("eval", "Eval 9장면")):
            ov = R["posthoc"].get(f"{split_key}_overall", {})
            sc_map = R["posthoc"].get(f"{split_key}_scenes", {})
            for band, bko in (("2g4", "2.4 GHz"), ("5g", "5 GHz")):
                if band not in ov:
                    continue
                for mn, v in ov[band].items():
                    w_sc, w_m, w_p = "—", -1.0, -1.0
                    for sc_id, sc_d in sc_map.get(band, {}).items():
                        if mn in sc_d and sc_d[mn]["one_node"]["mean"] > w_m:
                            w_sc = sc_id
                            w_m = sc_d[mn]["one_node"]["mean"]
                            w_p = sc_d[mn]["one_node"]["p95"]
                    w_str = f"`{w_sc}` {w_m:.1f}° ({w_p:.1f}°)" if w_m >= 0 else "—"
                    note = "외경 ~9.5 cm (R8 여유 확대)" if "D85" in mn else "외경 ~11 cm"
                    L.append(
                        f"| {split_ko} {bko} | **{mn}** | {_fmt(v['one_node'])} | {_fmt(v['fused_adj'])} | "
                        f"{_fmt(v['fused_diag'])} | {w_str} | {note} |"
                    )
        L.append("")

    # Section 7: What to trust / unexpected findings / what not to trust + reproduction commands
    L.append("## 7. 믿어도 되는 것 · 예상과 달랐던 발견 · 믿으면 안 되는 것 (AGENTS.md 5.5절)")
    L.append("")
    L.append("- **믿어도 되는 것 (설계 선택 근거)**:")
    L.append("  1. **AX210 $\\pi$-반전 보정 필수**: 보정하지 않으면(`SPEC-002-NaiveFlip`) 전 공간 평균 30.8~31.6°(P95 73~77°)로 붕괴하며, 수동 RF 부호 앵커(`|α|=3.5`)가 전 13장면 · 양 대역에서 97.8~99.2% 부호를 복구함.")
    L.append("  2. **R8 준수 + 9소자 통합 이득**: 사전 등록 `SPEC-002+`(AX210 $\\pi$-반전 포함, `dz=0`, 높이 ≤18 cm)가 이상적 무반전 · R8 초과 기존 `SPEC-002-NoFlip`(`dz=+65`, 23.5 cm)보다 2.4 GHz · 5 GHz 평가 9장면 전체에서 평균 · P95 · 최악 방 모두 우세함.")
    L.append("  3. **12+1 소자(`SPEC-002++`) 불필요**: 8+1 대비 평가 평균 차이는 0.3~0.5°에 불과하고 5 GHz 2자리 P95는 오히려 악화되어(37.6° vs 36.7°), +10만 원 · 2단 스위치 손실을 감수할 가치가 없음.")
    L.append("  4. **2자리 결합(특히 대각 배치 `(0,3)`)의 효용**: 평가 9장면 통합에서 2자리 대각 `(0,3)`은 2.4 GHz `8.2°`(P95 `25.8°`), 5 GHz `8.2°`(P95 `30.0°`)로 1자리 및 인접 `(0,1)`보다 일관되게 안정적임(단, 앞벽 칠판 교실·강의실 A는 앞쪽 인접 `(0,1)`이 유리).")
    L.append("  5. **5 GHz 채널 폭 효과**: 20 → 40 → 80 MHz로 넓어질수록 2자리 오차가 `9.2° → 8.8° → 8.0°`(P95 `31.1° → 27.0°`)로 단조 개선됨.")
    L.append("- **예상과 달랐던 5장 대조군 발견 (가설 수정 및 단순화 — AGENTS.md 4장 · 5.1.10절)**:")
    L.append("  1. **$|H_{\\text{ref}}|^2$ 가중 켤레곱 평균(`Conj-Avg`)이 비가중 비율 평균(`Ratio-Avg`)을 앞섬**: 당초 동적 산란 편향 때문에 `Ratio-Avg`가 유리할 것으로 가정했으나, 실제 Sionna RT 다중경로 + 열잡음 + ~1% 부호 오차에서는 기준 안테나 딥 페이드(`|H_ref|≈0`)의 `1/H_ref` 잡음 폭주가 훨씬 치명적이었다. `Conj-Avg`는 4장면 대조에서 2.4 GHz P95를 `-4.9°`, 5 GHz 2자리 P95를 **`-13.6°`** 줄였고, 사후 확인(6.5절)에서도 Dev · Eval 양쪽 모두 일관되게 개선되었다.")
    L.append("  2. **무제약 대형 반사체 면 추가(`Reflector-Map`)는 이득 없음**: 방 크기 · 위치 오차 15 cm가 섞인 상태에서 무제약 능형 회귀에 칠판 · 거울 거울상을 추가하면 자유도 과다로 `No-Reflector-Map`(6면 단순 지도)보다 0.1~0.4° 오히려 악화되었다. 물리적 반사율 제약이 검증되기 전까지는 **단순 6면 방 지도**를 기본으로 두고 반사체 입력은 선택/실험 기능으로 낮춘다.")
    L.append("  3. **고리 지름 `D=85 mm` vs `D=100 mm` 맞바꿈**: `D=85 mm`는 5 GHz 인접 소자 간격을 `0.566λ`로 줄여 5 GHz 대조군 P95를 `-10.1°` 개선하고 외경을 ~9.5 cm로 축소했다. 사후 전 장면 비교(6.5절)에서 `D100+Conj`는 2.4 GHz가 `0.2~0.4°` 근소 우세, `D85+Conj`는 5 GHz 1자리/인접이 `0.2~0.3°` 우세 + 크기 1.5 cm 소형화다(V3 3D 프린트 직전 0원 선택 가능).")
    L.append("  4. **개방 반사형 스위치(`Reflect-OPEN-SW`)**: 1회 정밀 NEC2 패턴 보정표를 완벽히 적용한 조건에서는 50 Ω 흡수형과 오차 차이가 작았다(0.1~0.3°). 다만 `PE42482` 50 Ω 흡수형은 내장 All-Off 50 Ω 종단(부호 앵커 실시간 보정)과 단일칩 1.1–1.4 dB 저손실 장점이 유지되므로 1순위를 유지한다.")
    L.append("- **믿으면 안 되는 것 (실측 필요 항목)**:")
    L.append("  1. 이 숫자는 시뮬레이션 **계산값**이며 DiffLoc+ 논문 실측값(`11.92°`)과 직접 비교해 '목표 달성'이라 쓰지 않는다(AGENTS.md 5.4절).")
    L.append("  2. AX210의 RX0/RX1 **독립 AGC 이득/위상 단차**와 실물 `PE42482`의 6 GHz 포트 위상 비대칭, 듀얼밴드 다이폴 매칭 편차는 시뮬레이션에 전부 담기지 않았으므로 **V2(3.5~5만 원) → V3 → V4 실측**으로 검증해야 한다.")
    L.append("")
    L.append("## 8. 다시 돌리는 명령")
    L.append("")
    L.append("```bash")
    L.append("cd receiver")
    L.append("python3 sim/em/ring_nec2.py")
    L.append("python3 sim/v1_ring/gen_scenes.py --scenes all --bands 2g4,5g --seeds 0,1,2")
    L.append("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 sim/v1_ring/run_eval.py")
    L.append("python3 sim/v1_ring/make_report.py")
    L.append("```")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
