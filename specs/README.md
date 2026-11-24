# 스펙 목록 (`receiver`)

DeteCam 5.0 (`Palm-12`) 최신 단일 스펙입니다. 이전 세대(v1~v4) 스펙은 정리했으며 초기 회전 반사판 구조는 [receiver-legacy](https://github.com/DeteCam/receiver-legacy)에 보관되어 있습니다.

| 번호 | 스펙 | 상태 | 요약 | 핵심 구현 |
|---|---|---|---|---|
| 001 | [`001-palm-receiver`](001-palm-receiver/spec.md) | 설계·시뮬레이션·슬라이싱 완료 | 손바닥 크기(`120×120×30.6 mm`) 12+1소자 통합 RF PCB + 무타공 벽면 2축 스위블 + `TriF v12` + `AR-OPA` + Bambu X1C 단 1판 G코드 | `cad/model/palm_node.py`, `firmware/core/*.py`, `sim/*.py` |
