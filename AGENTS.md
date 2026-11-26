# AGENTS.md (receiver)

**작업 전에 [DeteCam/docs의 AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md)를 먼저 끝까지 읽는다.** 목표, 요구사항, 시뮬레이션 규칙, 검증 · 조사 · 구매 규칙은 그 파일이 기준이며 여기에 다시 적지 않는다. 지금 고른 설계는 [docs/현재_방향.md](https://github.com/DeteCam/docs/blob/main/현재_방향.md)에 있다.

이 저장소에만 해당하는 규칙:

1. 지금 설계는 [`specs/002-switch-ring-receiver`](specs/002-switch-ring-receiver/spec.md)다. 바꾸려면 스펙부터 고친다.
2. 시뮬레이션은 `sim/` 아래에 둔다. 새 계산을 돌리기 전에 `sim/plan_YYYYMMDD_이름.md`에 수신기 · 카메라 자리, 조건, 평가 장면, 판정 기준을 먼저 적는다. 결과 보고는 docs AGENTS.md 5.5절 형식을 따르고, 공간(5.2절)과 도구(5.3절) 규칙을 지킨다. 계산 목록 · 결과 · 다시 돌리는 법은 [`sim/README.md`](sim/README.md)에 모은다.
3. 신호를 만드는 코드(`sim/mysim/gen_*.py`, `my_eval.py`의 합성 부분, `risk_synth.py`)와 추정기는 서로의 내부 값을 읽지 않는다. 추정기는 사용자가 실제로 알 수 있는 방 정보만 받는다.
4. 큰 데이터(`sim/data_*/`, `sim/results/*.pkl`, `*.npy`)는 올리지 않는다. 다시 만드는 명령을 README에 적는다.
5. `sim/setup_paths.sh`로 바꾼 경로는 커밋하지 않는다. 스크립트에는 자리표시 경로(`/opt/detecam/...`)만 둔다.
6. v6 코드와 숫자는 브랜치 `archive/v6-nano2`에만 있다. 재현 말고는 쓰지 않는다.
7. 커밋 · PR은 [CONTRIBUTING](https://github.com/DeteCam/.github/blob/main/CONTRIBUTING.md) 규칙을 따르고 AI 도구 흔적을 남기지 않는다.
