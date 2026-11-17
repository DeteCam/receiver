# 스펙 목록 (receiver)

스펙 주도 개발: `spec.md`(무엇을·왜) → `plan.md`(어떻게) → `tasks.md`(할 일) 순서로 쓰고 나서 만든다. 규칙은 [CONTRIBUTING](https://github.com/DeteCam/.github/blob/main/CONTRIBUTING.md), 시스템 전체 명세는 [docs/specs/000-system](https://github.com/DeteCam/docs/tree/main/specs/000-system).

| 번호 | 스펙 | 상태 | 한 줄 | 코드 |
|---|---|---|---|---|
| 001 | [direction-finding](001-direction-finding/spec.md) | 006 으로 바뀜 | 판을 돌려 얻은 신호 곡선으로 카메라 방향을 읽는다 | `firmware/core/analyze3.py` `estimator.py` `fusion.py` |
| 002 | [position-estimation](002-position-estimation/spec.md) | 구현 | 여러 곳에서 잰 것을 겹쳐 구역과 자리 후보를 낸다 | `firmware/core/locate.py` |
| 003 | [mechanics](003-mechanics/spec.md) | 006 으로 바뀜 | 관절 셋, 판 틀, 받침, 스탠드 받이. 출력해서 나사로 조립한다 | `cad/` `print/` |
| 004 | [detection](004-detection/spec.md) | 구현 | 둘레를 엿들어 의심 기기를 고르고, 찔러서 응답을 받는다 | `firmware/pi/detect.py` `firmware/core/score.py` |
| 005 | [standalone-control](005-standalone-control/spec.md) | 구현 | 노트북 없이 켜지고 폰으로 조종한다 | `firmware/pi/node.py` `web.html` |
| 006 | [panel-receiver](006-panel-receiver/spec.md) | 구현 | 모터 없이 안테나 배열로 방향을 잰다. 패널 둘을 걸어 두고 잰 것을 모은다 | `cad/model/panel.py` `firmware/core/panel.py` `sim/rt/rt_study.py` |

상태: 설계(스펙 작성 중) → 구현(코드가 있고 계산으로 확인) → 검증(실물로 확인 중) → 완료. 001 과 003(돌판 수신기)은 태그 `plate-v2` 에 두고 006 으로 바꿨다. 002 의 자리 내기는 006 이 이어받는다.
