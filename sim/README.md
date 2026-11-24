# DeteCam 5.0 (Palm-12) 시뮬레이션

> 모든 그림과 수치는 아래 2개 스크립트를 다시 돌려 100% 재현할 수 있습니다.

- [`opt_palm_array.py`](opt_palm_array.py) → [`palm_array_results.json`](palm_array_results.json)  
  손바닥 크기(`112×112 mm` 활성 개구) 4종 배열과 기존 `Fold12-Mini(300mm)` / `v4.0 L-8(420mm)`을 2.45/5.2/5.8 GHz 부엽(`PSL`), `CRLB`, 유도-EMF 상호결합 행렬(`C`) 보정 전후 + 15% AX210 위상 반전 + 실내 다중경로 2D Capon 몬테카를로로 정량 비교합니다.
- [`ar_opa_benchmark.py`](ar_opa_benchmark.py) → [`ar_opa_results.json`](ar_opa_results.json)  
  6대 실공간(숙소·화장실·목욕탕·탈의실·40석 교실·150석 강당)에서 대형 스탠드 없이 손 닿는 벽면(`1.1 m`/`2.0 m`) 부착 + 15° 클릭 스위블(`Pan ±30°`, `Tilt 0°/-15°/-25°`) `AR-OPA` 성능을 250회 몬테카를로로 검증합니다.
