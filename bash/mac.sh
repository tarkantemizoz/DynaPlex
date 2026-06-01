#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <executable> [args...]" >&2
    exit 1
fi

EXE="$1"
shift
PRESET="${PRESET:-MacRel}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO/out/$PRESET"
BIN_DIR="$BUILD_DIR/bin"
LOG_DIR="$REPO/bash/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$LOG_DIR/${EXE}_${STAMP}.out"
ERR="$LOG_DIR/${EXE}_${STAMP}.err"

echo "[build] $EXE  (preset=$PRESET)"
cmake --build "$BUILD_DIR" --target "$EXE" -j

echo "[run]   $BIN_DIR/$EXE $*"
echo "[log]   $OUT"
cd "$BIN_DIR"
set +e
./"$EXE" "$@" > "$OUT" 2> "$ERR"
STATUS=$?
set -e

if [ ! -s "$ERR" ]; then
    rm -f "$ERR"
    echo "[done]  exit=$STATUS"
else
    echo "[done]  exit=$STATUS  stderr=$ERR"
fi
exit $STATUS