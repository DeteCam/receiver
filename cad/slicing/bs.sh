#!/bin/bash
# Bambu Studio 를 화면 없이 돌린다 (화면이 없는 리눅스 컴퓨터용). slice_bambu.py 가 이 파일을 부른다.
#   BAMBU_STUDIO_DIR = Bambu Studio AppImage 를 푼 폴더 (./Bambu_Studio.AppImage --appimage-extract 하면 생기는 squashfs-root)
# 화면이 있는 컴퓨터에서는 이 파일 없이:  BAMBU_STUDIO=/경로/bambu-studio python3 slice_bambu.py ...
# 미리보기 그림(썸네일)을 만들려면 Wayland 화면과 OSMesa 가 있어야 해서, 화면 없는 weston 을 하나 띄워 둔다 (apt: weston libosmesa6).
D=${BAMBU_STUDIO_DIR:-$HOME/dl/bambu_2.5.1/squashfs-root}
HERE="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR=${BS_RUNTIME_DIR:-/tmp/detecam/xdg}
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
if [ ! -S "$XDG_RUNTIME_DIR/wl-bs" ] || ! pgrep -x weston > /dev/null; then
  rm -f "$XDG_RUNTIME_DIR/wl-bs" "$XDG_RUNTIME_DIR/wl-bs.lock"
  (weston --backend=headless --renderer=gl --socket=wl-bs --width=1280 --height=800 --idle-time=0 > "$XDG_RUNTIME_DIR/weston.log" 2>&1 &)
  for i in $(seq 1 40); do [ -S "$XDG_RUNTIME_DIR/wl-bs" ] && break; python3 -c "import time; time.sleep(0.25)"; done
fi
export WAYLAND_DISPLAY=wl-bs LD_LIBRARY_PATH="$D/bin" LC_ALL=C
# OpenGL 호출을 OSMesa 로 보낸다. shim/glshim.so 는  gcc -shared -fPIC -o shim/glshim.so shim/glshim.c  로 만든다
export LD_PRELOAD="${OSMESA_LIB:-/usr/lib/x86_64-linux-gnu/libOSMesa.so.8}:$HERE/shim/glshim.so"
exec "$D/bin/bambu-studio" "$@"
