# 스펙 목록 (`receiver`)

DeteCam 6.0 (`Nano-2 AI`) 최신 단일 스펙입니다. 이전 세대 스펙은 정리했으며 초기 구형 구조는 [receiver-legacy](https://github.com/DeteCam/receiver-legacy)에 보관되어 있습니다.

| 번호 | 스펙 | 상태 | 요약 | 핵심 구현 |
|---|---|---|---|---|
| 001 | [`001-nano2-receiver`](001-nano2-receiver/spec.md) | 설계·AI학습·10공간검증·슬라이싱 완료 | 포켓 크기(`96×96×36.0 mm`) 단 2기 벽면 스위블 + 무개조 `NanoPi R5C` + `ReconCovNet-v4` + `CamGuard-Net` + Bayesian `MA-PINN` + `AR-OPA-2` + Bambu X1C 단 1판 G코드 | `cad/model/nano2_node.py`, `firmware/core/ai_engine.py`, `sim/*.py` |
