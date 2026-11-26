#!/usr/bin/env bash
# 스크립트 안의 자리표시 경로(/opt/detecam/sim, /opt/detecam/receiver-legacy)를 이 컴퓨터 경로로 바꾼다.
# 사용: bash setup_paths.sh /path/to/receiver-legacy
# 바꾼 뒤의 파일은 커밋하지 않는다(경로가 컴퓨터마다 다르다).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LEG="${1:?receiver-legacy 저장소 경로를 주세요}"
grep -rl "/opt/detecam/receiver-legacy" "$HERE/mysim" | xargs -r sed -i "s#/opt/detecam/receiver-legacy#$LEG#g"
grep -rl "/opt/detecam/sim" "$HERE/mysim" "$HERE/v1_ring" | xargs -r sed -i "s#/opt/detecam/sim#$HERE#g"
mkdir -p "$HERE/results" "$HERE/data_wall" "$HERE/data_real" "$HERE/logs"
echo "경로를 $HERE, $LEG 로 바꿨습니다."
