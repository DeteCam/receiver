#!/bin/bash
# 미리 빌드한 csi_probe 를 ESP32-C5 보드에 올린다.  ./flash.sh /dev/ttyACM0
# 필요한 것: pip install esptool pyserial   (ESP-IDF 는 없어도 된다)
# 보드의 USB 가 칩에 바로 붙었는지(USB-JTAG, 예: XIAO, FireBeetle) 변환 칩(CH340 등, 예: DevKitC 호환보드)을 거치는지 보고
# 맞는 파일을 고른다. 억지로 고르려면 두 번째 인자로 usb 또는 uart.
set -e
PORT=${1:?포트를 적는다. 예: ./flash.sh /dev/ttyACM0  (python3 ../../bench/probe.py ports 로 찾는다)}
HERE="$(cd "$(dirname "$0")" && pwd)"
KIND=${2:-$(python3 - "$PORT" <<'PY'
import sys
from serial.tools import list_ports
p = [x for x in list_ports.comports() if x.device == sys.argv[1]]
print("usb" if p and p[0].vid == 0x303a else "uart")
PY
)}
BIN="$HERE/bin/csi_probe_merged.bin"; [ "$KIND" = "uart" ] && BIN="$HERE/bin/csi_probe_uart.bin"
echo "보드 연결: $KIND -> $(basename "$BIN")"
python3 -m esptool --chip esp32c5 -p "$PORT" -b 460800 --before default_reset --after hard_reset write_flash 0x0 "$BIN"
echo "올렸다. 다음: cd ../../bench && python3 probe.py setup 보내는포트 받는포트"
