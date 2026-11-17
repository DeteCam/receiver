<div align="center">

# DeteCam 수신기

**숨은 Wi-Fi 카메라의 방향과 자리를 찾는 수신기. 걸어 두는 패널 둘과 허브 하나, 모터 없음**

### [▶ 동작 영상](https://detecam.github.io/receiver/videos/) &nbsp;·&nbsp; [▶ 조립 영상](https://detecam.github.io/receiver/videos/) &nbsp;·&nbsp; [왜 이렇게 만들었나](docs/설계%20검토.md)

![패널 모습](docs/images/패널_모습.png)

</div>

## 어떻게 찾나

패널에는 안테나 16개가 달린 15 cm 판이 있습니다. 전자 스위치로 안테나를 하나씩 골라 같은 전파를 16자리에서 받으면, 자리마다의 위상 차이에서 전파가 온 방향이 나옵니다. 한 번 재는 데 0.3초입니다. 돌아가는 것이 없어 소리가 나지 않습니다.

ESP32 는 수신 길이 하나라 안테나 사이의 위상을 직접 못 잽니다. 그래서 기준 안테나와 합친 세기를, 기준 길의 위상을 네 번 바꿔 가며 잽니다. 세기 네 개에서 위상이 나옵니다 (홀로그램을 찍는 방법과 같습니다).

패널 둘을 마주 보는 두 벽의 2 m 높이에 걸어 둡니다. 두 방향이 만나는 곳이 카메라의 자리입니다. 걸어 둔 채 계속 재므로, 사람들이 자세를 바꾸는 사이 가려졌던 카메라가 보이는 순간을 모읍니다.

![교실 동작](docs/images/교실_동작.png)

| 탐지 | 위치 측정 | 표시 |
|---|---|---|
| 허브가 둘레의 Wi-Fi 를 엿들어 의심 기기를 고른다 | 패널 둘이 그 기기가 보낼 때마다 방향을 잰다 (0.3초) | 허브가 자리 후보 셋을 폰에 보여 준다 |

## 숫자로 보면 (계산, 실물은 아직)

40명이 앉은 교실을 전파 경로 계산(Sionna RT)으로 풀었습니다 ([설계 검토](docs/설계%20검토.md), [패널 연구](sim/rt/패널%20연구.md)). 패널 둘을 창 유리와 복도 쪽 창턱의 2 m 높이에 걸고, 자리를 정확히 / 옆자리까지 맞힌 비율(%)입니다.

| 카메라가 보낸 횟수 | 책상 위 | 가슴 | 안경 |
|---|---|---|---|
| 1번 | 70 / 90 | 56 / 81 | 70 / 93 |
| 4번 | 80 / 95 | 80 / 94 | 88 / 99 |
| 8번 | 84 / 97 | 87 / 97 | 90 / 100 |

![전송을 모을수록](docs/images/정확도_전송횟수.png)

| 항목 | 값 |
|---|---|
| 방향 오차 (한 번, 직접 보일 때) | 가운데값 3.6도. 몸에 가리면 24도 |
| 한 번 재는 시간 | 0.3초 (패킷 64개) |
| 사진만 가끔 보내는 카메라 | 전송 한 번(0.5초)마다 한 번 잰다 |
| 강의실 72석 | 패널 둘, 여덟 번: 81 · 80 · 87%. 패널 셋: 89 · 94 · 99% |
| 크기와 무게 | 패널 165 × 225 × 38 mm, 약 0.6 kg. 허브 165 × 87 × 59 mm |
| 출력 | 다섯 판(1·2판은 두 번), 약 22시간, 필라멘트 0.8 kg, 받침 없이 (Bambu X1 Carbon, PLA) |
| 부품 값 | 약 53만 원 (어림, 탐지용 동글과 필라멘트까지). 먼저 하는 실험에는 24만 원 |

앞선 구조(판을 돌리는 수신기)는 같은 조건에서 책상 위 21~55% 였습니다. 여섯 곳을 2.4 m 스탠드로 옮겨 가며 재야 90% 가 나왔습니다. 그 설계는 태그 [`plate-v2`](https://github.com/DeteCam/receiver/tree/plate-v2)에 있습니다.

![돌판과 패널](docs/images/돌판과_패널.png)

## 폴더

```
docs/       문서와 그림 (옵시디언 볼트의 DeteCam/receiver 폴더)
cad/        3D 모델 코드: model/panel.py (치수와 부품), export/ checks/ figures/ slicing/ results/
stl/        출력할 부품 8가지      print/      출력 파일 (G코드 5판)      marker/     AR 마커
firmware/   수신기 코드: core/panel.py (패널 계산), pi/ (허브), tests/
sim/        시뮬레이션: rt/ (교실 전파 경로 계산, 패널 연구), direction/ location/ em/ (돌판 때의 계산)
specs/      스펙: 기능마다 spec → plan → tasks (스펙 주도 개발)
console/    돌판 수신기의 시뮬레이터
```

파일마다의 설명은 **[폴더와 파일](docs/폴더와%20파일.md)** 에 있습니다.

## 처음이면 이 순서로

1. [설계 검토](docs/설계%20검토.md) — 왜 돌판에서 패널로 바꿨나, 얼마나 맞나, 무엇이 아직 위험한가
2. [체크리스트](docs/체크리스트.md) — 부품을 다 사기 전에 하는 실험 넷
3. [부품표](docs/부품표.md) · [부품 검토](docs/부품%20검토.md) — 살 것과 뽑을 것
4. [조립 안내](docs/조립%20안내.md) — 패널과 허브 조립, 거는 법

## 지금 상태 (2026-11-17)

- **출력 전**입니다. 계산, 슬라이싱, 슬라이싱한 파일의 검증까지 했고 실물은 없습니다.
- 패널 구조는 두 가지가 실물에서 되어야 섭니다: ESP32-C5 가 ACK 의 CSI 를 내는가, 세기만으로 위상이 되찾아지는가. 안 되면 돌판(`plate-v2`)으로 돌아갑니다.
- 장치 코드는 패널의 계산(`firmware/core/panel.py`)과 검사까지 있습니다. ESP32 에 올릴 코드와 허브의 패널 연결은 아직 없습니다 ([스펙 006](specs/006-panel-receiver/tasks.md)). `firmware/pi/` 의 모터 조종과 `console/` 의 시뮬레이터는 돌판 때의 것입니다.
- 블루투스로만 보내는 기기(일부 AI 안경)와 5 GHz 로만 보내는 카메라의 방향은 재지 못합니다.
- 저장소는 넷입니다: receiver(여기) · [app](https://github.com/DeteCam/app) · [ai](https://github.com/DeteCam/ai) · [docs](https://github.com/DeteCam/docs). 규칙은 [CONTRIBUTING](https://github.com/DeteCam/.github/blob/main/CONTRIBUTING.md).

## 받기와 올리기

```
git clone https://github.com/DeteCam/receiver.git receiver
cd receiver
git pull                                   # 최신으로
git add -A && git commit -m "무엇을 바꿨나" && git push     # 올리기
```

옵시디언에서 고친 문서도 `docs/` 안의 파일이라 같은 명령으로 올라갑니다. 지난 판의 파일은 따로 두지 않습니다 (깃 기록에서 찾습니다).

## 바탕이 된 연구

- Holl, Reinhard. *Holography of Wi-fi Radiation.* Phys. Rev. Lett. 118, 183901 (2017): 세기만 재서 Wi-Fi 의 위상을 되찾는다
- Xie 외. *SWAN: Stitched Wi-Fi ANtennas.* MobiCom 2018: 스위치로 안테나를 늘린다
- Abedi, Vasisht. *Wi-Peep.* MobiCom 2022: 빈 프레임을 보내 응답(ACK)을 받는다
- DiffLoc (USENIX Security 2025), DiffLoc+ (IEEE JSAC 2026), CamLoPA (IEEE S&P 2025): Wi-Fi 숨은 카메라 위치 추정. 돌판 수신기의 바탕

모두 [설계 검토 15절](docs/설계%20검토.md#15-바탕-자료)에 링크와 함께 있습니다.
