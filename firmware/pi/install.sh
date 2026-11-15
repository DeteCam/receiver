#!/usr/bin/env bash
# 수신기(파이)에 '혼자 동작' 코드를 넣는다. 노트북에서 실행한다 (개발할 때처럼 랜선으로 SSH 가 되는 상태에서).
#   pi/install.sh [pi@raspberrypi.local]
# 하는 일: ① 파일을 파이의 ~/detecam 에 복사  ② 파이에서 계산 검사(test_core.py)  ③ 켜질 때 자동으로 뜨게 등록  ④ /boot/config.txt 의 버튼·모터 핀 줄 확인
# 하지 않는 일: apt upgrade (nexmon 이 깨진다), 무선 동글 설정 (그것은 파이에서 link_setup.sh)
set -euo pipefail
PI=${1:-${PI:-pi@raspberrypi.local}}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
FILES=("$HERE/rx3.py" "$HERE/node.py" "$HERE/csi.py" "$HERE/detect.py" "$HERE/web.html" "$HERE/icon-192.png" "$HERE/icon-512.png" "$HERE/marker.json" "$HERE/detecam.service" "$HERE/link_setup.sh"
       "$ROOT/core/analyze3.py" "$ROOT/core/estimator.py" "$ROOT/core/model.py" "$ROOT/core/fusion.py" "$ROOT/core/locate.py" "$ROOT/core/identify.py" "$ROOT/core/score.py" "$ROOT/core/world.py"
       "$ROOT/tests/test_core.py" "$ROOT/tests/fixture.json" "$ROOT/tests/fixture.npz" "$ROOT/tests/fixture_mon.json" "$ROOT/tests/fixture_mon.pcap")
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "파일이 없습니다: $f"; exit 1; }; done
echo "== 파이: $PI"
ssh "$PI" "mkdir -p ~/detecam/data"
scp -q "${FILES[@]}" "$PI:~/detecam/"
ssh "$PI" 'bash -s' <<'REMOTE'
set -e
cd ~/detecam && chmod +x link_setup.sh
python3 -c "import numpy" 2>/dev/null || { echo "numpy 가 없어 설치합니다 (apt install python3-numpy)"; sudo apt-get install -y python3-numpy; }
echo "-- 계산 검사 (적어 둔 패킷으로 방향·구간·판별·탐지를 다시 계산해 노트북의 답과 견준다)"
python3 test_core.py
python3 - <<'PY'
import sys; sys.path.insert(0, "/home/pi/detecam")
import csi, detect
d = detect.find_monitor()
print("CSI 를 받는 쪽 = %s, USB 무선 = %s, 탐지 동글 = %s" % (csi.csi_iface(), [x["name"] + "/" + x["driver"] for x in detect.usb_wifi()] or "없음", (d["name"] + " (" + d["driver"] + ")") if d else "없음"))
PY
CFG=/boot/config.txt
add() { grep -qF "$1" $CFG || { echo "$1" | sudo tee -a $CFG >/dev/null; echo "config.txt 에 추가: $1"; NEED_REBOOT=1; }; }
[ -f $CFG.detecam.bak ] || sudo cp $CFG $CFG.detecam.bak
NEED_REBOOT=0
add "dtoverlay=gpio-shutdown,gpio_pin=3,debounce=2000"
add "gpio=6,12,13,16,17,19,20,21,22,23,26,27=op,dl"
grep -q "^dtparam=i2c_arm=on" $CFG && echo "주의: config.txt 에 dtparam=i2c_arm=on 이 있습니다. 전원 버튼이 5번 핀(GPIO3)을 쓰므로 그 줄 앞에 # 을 붙여 꺼 주세요."
sudo cp ~/detecam/detecam.service /etc/systemd/system/detecam.service
sudo systemctl daemon-reload && sudo systemctl enable detecam.service >/dev/null 2>&1 && sudo systemctl restart detecam.service
sleep 3; systemctl is-active detecam.service && echo "detecam 서비스가 켜졌습니다. 로그: journalctl -u detecam -f"
[ "$NEED_REBOOT" = 1 ] && echo "config.txt 를 고쳤습니다. 한 번 다시 켜야 적용됩니다: sudo reboot"
echo "브라우저에서 열 주소 (지금 연결된 것): $(hostname -I | tr ' ' '\n' | grep -v '^$' | sed 's|^|http://|' | tr '\n' ' ')"
REMOTE
echo "== 끝. 무선 동글 설정은 파이에서:  cd ~/detecam && sudo ./link_setup.sh check   (폰 연결: ap 또는 client, 탐지 동글: mon)"
