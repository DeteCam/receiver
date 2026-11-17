# 005 혼자 동작과 조종: 방법

- `firmware/pi/install.sh` 가 서비스(`detecam.service`)와 수신기 Wi-Fi(`link_setup.sh`)를 깐다.
- `firmware/pi/node.py` 가 웹 서버다. 모터(`rx3.py`), 측정(`csi.py`), 탐지(`detect.py`)를 부르고, 계산은 `firmware/core` 를 쓴다.
- 오래 걸리는 일(탐지, 측정)은 뒤에서 돌리고, 앱은 상태를 1초마다 물어본다.
- 폰 브라우저용 화면은 `web.html` 한 장이다.

자세한 설명과 화면은 [docs/혼자 동작](../../docs/혼자%20동작.md).
