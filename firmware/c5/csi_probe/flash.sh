#!/bin/bash
# 미리 빌드한 csi_probe 를 ESP32-C5 에 올린다.  ./flash.sh /dev/ttyACM0
# 필요한 것: pip install esptool   (ESP-IDF 는 없어도 된다)
set -e
PORT=${1:?포트를 적는다. 예: ./flash.sh /dev/ttyACM0  (python3 ../../bench/probe.py ports 로 찾는다)}
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -m esptool --chip esp32c5 -p "$PORT" -b 460800 --before default_reset --after hard_reset \
  write_flash 0x0 "$HERE/bin/csi_probe_merged.bin"
echo "올렸다. 다음: cd ../../bench && python3 probe.py setup 보내는포트 받는포트"
