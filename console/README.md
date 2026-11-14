# console: 3축 시뮬레이션 화면

브라우저에서 3D 모델을 돌려 보고, 카메라를 아무 데나 숨긴 뒤 재는 과정을 미리 해 보는 화면입니다. 실제 장치와는 연결되어 있지 않습니다 (장치를 조종하는 폰 화면은 `../firmware/pi/web.html`).

- **그냥 보기:** `dist/DeteCam_3축_화면.html` 을 브라우저로 엽니다. 인터넷이 없어도 열립니다 (글꼴만 기본 글꼴로 바뀝니다). 옵시디언에서는 [시뮬레이션 화면](../docs/시뮬레이션%20화면.md) 노트에 들어 있습니다.
- 모터가 움직이는 시간과 방향을 구하는 계산은 수신기에 올리는 코드와 같습니다. 같은 입력에 같은 값이 나오는지 `src/test_core3.js` 가 확인합니다.

## 다시 만들기

```
cd src
node test_core3.js               # 계산이 파이썬 원본과 같은지
python3 build3.py --out ../dist/DeteCam_3축_화면.html
```

3D 모델이나 계산 코드가 바뀌었을 때만 아래를 먼저 돌립니다 (`../../cad`, `../../firmware`, `../../sim` 을 읽습니다).

```
python3 viewer_data3.py          # 3D 모델과 표를 data.json 으로 (manifold3d, trimesh 필요)
python3 make_testvec.py          # 파이썬 원본의 계산 값을 testvec.json 으로
```

| 파일 | 내용 |
|---|---|
| `src/template.html` | 화면의 글과 그림 |
| `src/core3.js` | 계산 (모터 시간표, 방향 추정). 파이썬 코드를 옮긴 것 |
| `src/data.json` | 3D 모델, 회절 표, 미리 돌려 둔 결과 |
| `src/build3.py` | 위 셋을 HTML 한 장으로 묶는다 (다 묶은 뒤 `../docs/화면/` 에도 같은 파일을 둔다) |
