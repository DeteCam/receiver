#!/usr/bin/env bash
# USB 무선 동글 두 개의 설정. 파이에서 sudo 로 실행한다. 내장 Wi-Fi 는 CSI 를 받는 데만 쓰고 건드리지 않는다.
#   ① 폰과 통신하는 동글 (예: NETGEAR WNA3100M)          -> 이름 wlink0
#   ② 둘레 기기를 엿듣는 탐지 동글 (모니터 모드가 되는 것. 예: RT5370, MT7601U, AR9271 칩) -> 이름 wmon0. 없어도 수신기는 돈다 (탐지가 덜 정확할 뿐)
#
#   sudo ./link_setup.sh check                  동글이 잡혔는지, 어느 것을 어디에 쓸지 본다 (아무것도 바꾸지 않는다)
#   sudo ./link_setup.sh ap [이름] [비밀번호]     수신기가 자기 Wi-Fi 를 만든다 (이름 기본 DeteCam-RX). 폰은 여기에 붙어 http://192.168.4.1
#                                                비밀번호를 안 주면 처음 한 번 무작위로 만들어 /etc/detecam/ap_password 에 두고 화면에 보여 준다 (다시 보기: sudo cat 그 파일)
#   sudo ./link_setup.sh client 이름 비밀번호     폰 핫스팟이나 공유기에 붙는다 (AP 가 안 될 때)
#   sudo ./link_setup.sh mon                    탐지 동글을 준비한다: 이름을 고정하고, 다른 프로그램이 건드리지 않게 하고, 모니터 모드를 시험한다
#   sudo ./link_setup.sh off                    위 설정을 모두 되돌린다
#
# 주의: 이 스크립트는 실물에서 돌려 보지 못한 첫 판이다. 한 줄씩 무엇을 하는지 출력하니 보면서 진행한다.
#       랜선 SSH 는 건드리지 않으므로, 잘못돼도 랜선으로 들어와 'off' 로 되돌릴 수 있다. apt upgrade 는 하지 않는다.
set -euo pipefail
NAME=wlink0                                  # 폰과 통신하는 동글에 붙이는 고정 이름 (꽂는 순서에 따라 wlan0/wlan1 이 바뀌는 것을 피한다)
MON=wmon0                                    # 탐지 동글의 고정 이름
MON_OK="rt2800usb mt7601u ath9k_htc mt76x0u mt76x2u"     # 모니터 모드가 된다고 알려진 드라이버 (detect.py 의 목록과 같다)
MARK_A="# >>> DeteCam link"; MARK_B="# <<< DeteCam link"
MARK_C="# >>> DeteCam mon"; MARK_D="# <<< DeteCam mon"
say() { echo "== $*"; }
need_root() { [ "$(id -u)" = 0 ] || { echo "sudo 로 실행하세요."; exit 1; }; }

usb_wifi() {                                  # USB 로 붙은 무선 장치마다 한 줄: "이름 드라이버"
  local d
  for d in /sys/class/net/*; do
    [ -e "$d/phy80211" ] || [ -d "$d/wireless" ] || continue
    [ "$(basename "$(readlink -f "$d/device/subsystem")")" = usb ] || continue
    echo "$(basename "$d") $(basename "$(readlink -f "$d/device/driver")")"
  done
}
is_mon_drv() { case " $MON_OK " in *" $1 "*) return 0 ;; esac; return 1; }

find_dongle() {                               # 폰과 통신할 동글: 이름이 wlink0 인 것. 없으면 탐지용 드라이버가 아닌 첫째 것. 그것도 없으면 첫째 것
  local n d first=""
  while read -r n d; do [ "$n" = "$NAME" ] && { echo "$n"; return 0; }; done < <(usb_wifi)
  while read -r n d; do
    [ "$n" = "$MON" ] && continue
    is_mon_drv "$d" || { echo "$n"; return 0; }
    [ -z "$first" ] && first=$n
  done < <(usb_wifi)
  [ -n "$first" ] && { echo "$first"; return 0; }
  return 1
}

find_mon() {                                  # 탐지 동글: 이름이 wmon0 인 것. 없으면 폰용으로 고른 것 말고, 탐지용 드라이버를 쓰는 것
  local n d link; link=$(find_dongle || true)
  while read -r n d; do [ "$n" = "$MON" ] && { echo "$n"; return 0; }; done < <(usb_wifi)
  while read -r n d; do
    [ "$n" = "$link" ] && continue
    is_mon_drv "$d" && { echo "$n"; return 0; }
  done < <(usb_wifi)
  return 1
}

undo() {
  say "DeteCam 무선 설정을 지웁니다"
  systemctl disable --now hostapd >/dev/null 2>&1 || true
  rm -f /etc/dnsmasq.d/detecam.conf /etc/wpa_supplicant/wpa_supplicant-$NAME.conf
  [ -f /etc/hostapd/hostapd.conf ] && grep -q "DeteCam" /etc/hostapd/hostapd.conf && rm -f /etc/hostapd/hostapd.conf
  [ -f /etc/dhcpcd.conf ] && sed -i "/^$MARK_A/,/^$MARK_B/d" /etc/dhcpcd.conf
  systemctl restart dnsmasq >/dev/null 2>&1 || true
}

undo_mon() {
  rm -f /etc/systemd/network/10-detecam-mon.link
  [ -f /etc/dhcpcd.conf ] && sed -i "/^$MARK_C/,/^$MARK_D/d" /etc/dhcpcd.conf
  return 0
}

link_file() {                                 # 동글의 MAC 주소로 이름을 고정한다 (다음에 켤 때부터 적용)
  local ifc=$1 mac; mac=$(cat "/sys/class/net/$ifc/address")
  say "동글 $ifc (MAC $mac) 의 이름을 $NAME 으로 고정합니다"
  printf '[Match]\nMACAddress=%s\n\n[Link]\nName=%s\n' "$mac" "$NAME" > /etc/systemd/network/10-detecam-link.link
}

case "${1:-}" in
check)
  say "USB 장치"; lsusb | grep -i -E "0846:9021|realtek|ralink|mediatek|atheros|wireless|802\.11" || echo "   (무선 동글이 안 보입니다. 다른 USB 자리에 꽂아 보세요)"
  say "무선 장치와 드라이버"
  for d in /sys/class/net/*; do
    [ -e "$d/phy80211" ] || [ -d "$d/wireless" ] || continue
    echo "   $(basename "$d")  드라이버 $(basename "$(readlink -f "$d/device/driver")")  ($(basename "$(readlink -f "$d/device/subsystem")"))"
  done
  if IFC=$(find_dongle); then
    PHY=$(basename "$(readlink -f "/sys/class/net/$IFC/phy80211")")
    say "폰과 통신할 동글: $IFC ($PHY). 지원하는 방식"; iw phy "$PHY" info | sed -n '/Supported interface modes/,/Band/p' | grep '\*' | sed 's/^/   /'
    if iw phy "$PHY" info | sed -n '/Supported interface modes/,/Band/p' | grep -q '\* AP$'; then
      echo "   -> AP 가 목록에 있습니다:  sudo ./link_setup.sh ap  를 해 보세요."
    else
      echo "   -> AP 가 목록에 없습니다:  폰 핫스팟에 붙는 방식으로 하세요 (sudo ./link_setup.sh client 이름 비밀번호)."
    fi
  else
    echo "   USB 무선 동글을 찾지 못했습니다."
  fi
  if M=$(find_mon); then
    DRV=$(basename "$(readlink -f "/sys/class/net/$M/device/driver")")
    say "탐지 동글: $M (드라이버 $DRV)"
    echo "   -> 준비:  sudo ./link_setup.sh mon   (이름 고정 + 모니터 모드 시험)"
    dmesg 2>/dev/null | grep -i -E "$DRV|firmware" | grep -i -E "fail|error|not found" | tail -3 | sed 's/^/   주의(dmesg): /' || true
  else
    say "탐지 동글: 없음"
    echo "   모니터 모드가 되는 동글(칩 RT5370, MT7601U, AR9271 등. 드라이버 $MON_OK)을 가운데 USB 자리의 위 칸에 꽂는다."
    echo "   없어도 된다: 그때는 내장 Wi-Fi 의 CSI 수로 '초당 몇 개 보내는지'만 보고 후보를 고른다."
    echo "   (WNA3100M 의 드라이버 rtl8192cu 는 목록에 monitor 가 있어도 다른 기기의 프레임을 엿듣는 데는 못 쓴다.)"
  fi
  say "전원"; vcgencmd get_throttled 2>/dev/null | sed 's/^/   /' || true; echo "   (throttled=0x0 이 아니면 전압이 모자랐던 적이 있는 것: 굵은 USB 선, 2A 넘게 나오는 배터리를 쓴다)"
  ;;
ap)
  need_root; SSID=${2:-DeteCam-RX}
  mkdir -p /etc/detecam; PF=/etc/detecam/ap_password                  # 비밀번호는 장치마다 따로: 저장소에 적힌 기본값을 쓰지 않는다
  if [ -n "${3:-}" ]; then PASS=$3
  elif [ -s "$PF" ]; then PASS=$(cat "$PF")
  else PASS=$(head -c 512 /dev/urandom | tr -dc 'a-z0-9' | cut -c1-10); fi
  [ ${#PASS} -ge 8 ] || { echo "비밀번호는 8자 이상이어야 합니다."; exit 1; }
  printf '%s\n' "$PASS" > "$PF"; chmod 600 "$PF"
  IFC=$(find_dongle) || { echo "USB 무선 동글을 찾지 못했습니다."; exit 1; }
  dpkg -s hostapd dnsmasq >/dev/null 2>&1 || { say "hostapd, dnsmasq 를 설치합니다 (인터넷이 필요합니다. apt upgrade 는 하지 않습니다)"; apt-get install -y hostapd dnsmasq; }
  undo; link_file "$IFC"
  say "hostapd 설정 (/etc/hostapd/hostapd.conf)"
  cat > /etc/hostapd/hostapd.conf <<CONF
# DeteCam 수신기의 Wi-Fi (link_setup.sh 가 만들었다)
interface=$NAME
driver=nl80211
country_code=KR
ssid=$SSID
hw_mode=g
channel=1
auth_algs=1
wpa=2
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
wpa_passphrase=$PASS
CONF
  chmod 600 /etc/hostapd/hostapd.conf
  sed -i 's|^#\?DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
  say "주소 나눠 주기 (/etc/dnsmasq.d/detecam.conf): 폰은 192.168.4.10~50 을 받는다"
  printf 'interface=%s\nbind-dynamic\ndhcp-range=192.168.4.10,192.168.4.50,255.255.255.0,12h\naddress=/detecam.lan/192.168.4.1\n' "$NAME" > /etc/dnsmasq.d/detecam.conf
  say "수신기 주소 192.168.4.1 (/etc/dhcpcd.conf 맨 아래에 덧붙임. 원본은 dhcpcd.conf.detecam.bak)"
  [ -f /etc/dhcpcd.conf.detecam.bak ] || cp /etc/dhcpcd.conf /etc/dhcpcd.conf.detecam.bak
  printf '%s\ninterface %s\n    static ip_address=192.168.4.1/24\n    nohook wpa_supplicant\n%s\n' "$MARK_A" "$NAME" "$MARK_B" >> /etc/dhcpcd.conf
  systemctl unmask hostapd >/dev/null 2>&1 || true; systemctl enable hostapd dnsmasq >/dev/null 2>&1
  say "끝. 다시 켜면 적용됩니다:  sudo reboot"
  echo "   켠 뒤 폰에서 Wi-Fi '$SSID' (비밀번호 $PASS) 에 붙고, 브라우저에서 http://192.168.4.1 을 엽니다."
  echo "   비밀번호는 $PF 에 있다 (다시 보기: sudo cat $PF). 바꾸려면: sudo ./link_setup.sh ap $SSID 새비밀번호"
  echo "   Wi-Fi 가 안 보이면 랜선으로 들어와서:  sudo systemctl status hostapd ;  sudo journalctl -u hostapd -n 30"
  ;;
client)
  need_root; SSID=${2:?이름을 적어 주세요}; PASS=${3:?비밀번호를 적어 주세요}
  IFC=$(find_dongle) || { echo "USB 무선 동글을 찾지 못했습니다."; exit 1; }
  undo; link_file "$IFC"
  say "폰 핫스팟(또는 공유기) '$SSID' 에 붙도록 설정합니다 (/etc/wpa_supplicant/wpa_supplicant-$NAME.conf)"
  { printf 'ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\ncountry=KR\n'; wpa_passphrase "$SSID" "$PASS" | grep -v '#psk'; } > /etc/wpa_supplicant/wpa_supplicant-$NAME.conf
  chmod 600 /etc/wpa_supplicant/wpa_supplicant-$NAME.conf
  say "끝. 다시 켜면 적용됩니다:  sudo reboot"
  echo "   핫스팟은 2.4GHz 로 켜야 합니다 (이 동글은 5GHz 를 못 씁니다)."
  echo "   켠 뒤 같은 핫스팟에 붙은 기기에서 http://$(hostname).local 을 엽니다. 안 열리면 핫스팟의 '연결된 기기' 목록에서 수신기의 주소를 찾아 http://그 주소 로 엽니다."
  ;;
mon)
  need_root
  IFC=$(find_mon) || { echo "탐지 동글을 찾지 못했습니다. 모니터 모드가 되는 동글(드라이버 $MON_OK)을 꽂고 'sudo ./link_setup.sh check' 로 확인하세요."; exit 1; }
  ADDR=$(cat "/sys/class/net/$IFC/address"); undo_mon
  say "탐지 동글 $IFC (MAC $ADDR) 의 이름을 $MON 으로 고정합니다 (다음에 켤 때부터 적용)"
  printf '[Match]\nMACAddress=%s\n\n[Link]\nName=%s\n' "$ADDR" "$MON" > /etc/systemd/network/10-detecam-mon.link
  say "이 동글은 주소를 받거나 Wi-Fi 에 붙지 않게 합니다 (/etc/dhcpcd.conf 맨 위에 denyinterfaces. 원본은 dhcpcd.conf.detecam.bak)"
  [ -f /etc/dhcpcd.conf.detecam.bak ] || cp /etc/dhcpcd.conf /etc/dhcpcd.conf.detecam.bak
  { printf '%s\ndenyinterfaces %s %s\n%s\n' "$MARK_C" "$MON" "$IFC" "$MARK_D"; cat /etc/dhcpcd.conf; } > /etc/dhcpcd.conf.tmp && mv /etc/dhcpcd.conf.tmp /etc/dhcpcd.conf
  say "모니터 모드 시험 (채널 6 에서 3초 듣기)"
  ip link set "$IFC" down
  if iw dev "$IFC" set type monitor && ip link set "$IFC" up && iw dev "$IFC" set channel 6; then
    N=$(timeout 3 tcpdump -i "$IFC" -s 64 -w /tmp/detecam_mon_test.pcap 2>&1 | grep -o '[0-9]* packets captured' | grep -o '^[0-9]*' || true)
    echo "   모니터 모드로 바뀌었습니다. 3초 동안 들은 프레임: ${N:-0}개 (0 이면 둘레에 6번 채널 기기가 없거나 동글이 못 듣는 것: 다른 채널의 공유기 옆에서 다시 해 본다)"
  else
    echo "   모니터 모드로 바꾸지 못했습니다. 이 동글(드라이버)은 탐지에 못 씁니다."
  fi
  say "끝. 다시 켜면 이름이 적용됩니다:  sudo reboot   (수신기 화면의 '장치' 줄에 '탐지 동글: $MON' 이 보이면 된 것)"
  ;;
off)
  need_root; undo; undo_mon; rm -f /etc/systemd/network/10-detecam-link.link
  say "되돌렸습니다. 다시 켜면 적용됩니다:  sudo reboot"
  ;;
*)
  sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 1
  ;;
esac
