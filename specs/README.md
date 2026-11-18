# 스펙 목록 (receiver)

스펙 주도 개발: `spec.md`(무엇을·왜) → `plan.md`(어떻게) → `tasks.md`(할 일) 순서로 쓰고 나서 만든다. 규칙은 [CONTRIBUTING](https://github.com/DeteCam/.github/blob/main/CONTRIBUTING.md), 시스템 전체 명세는 [docs/specs/000-system](https://github.com/DeteCam/docs/tree/main/specs/000-system).

| 번호 | 스펙 | 상태 | 한 줄 | 코드 |
|---|---|---|---|---|
| 004 | [detection](004-detection/spec.md) | 구현 | 둘레를 엿들어 의심 기기를 고르고, 찔러서 응답을 받는다 | `firmware/pi/detect.py` `firmware/core/score.py` `firmware/core/identify.py` |
| 006 | [panel-receiver](006-panel-receiver/spec.md) | 구현 | 모터 없이 안테나 배열로 방향을 잰다. 패널 둘을 걸어 두고 잰 것을 모은다. 노트북 없이 폰으로 조종한다 | `cad/model/panel.py` `firmware/core/panel.py` `sim/rt/rt_study.py` |
| 007 | [phone-uplink-alarm](007-phone-uplink-alarm/spec.md) | 설계 (권장) | 허브에 RTL-SDR 을 꽂아 교실 안 휴대폰 송신을 알린다. 와이파이를 끊어 LTE 로 우회하는 경우의 일부. 자리는 재지 않는다 | (아직) |

상태: 설계(스펙 작성 중) → 구현(코드가 있고 계산으로 확인) → 검증(실물로 확인 중) → 완료.

001 direction-finding(판을 돌려 방향 읽기), 002 position-estimation(여러 곳을 겹쳐 자리 내기), 003 mechanics(관절 셋과 판 틀), 005 standalone-control(혼자 동작과 모터 조종)은 돌판 수신기의 스펙이라 코드와 함께 [receiver-legacy](https://github.com/DeteCam/receiver-legacy) 로 옮겼다. 002 의 자리 내기와 005 의 혼자 동작은 006 이 이어받는다 (006 의 FR-3, FR-9). 번호는 다시 쓰지 않는다.
