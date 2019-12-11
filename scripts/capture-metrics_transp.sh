#!/bin/bash
set -e

EXTRA_PARAMS=() # e.g., "-i"

# For simplicity, and better compatibility with binaries, use powers of two
ROWS=(8192)
COLS=(524288)
BLKS=(128 256 512)
THRS=(1 16)

SER=(
    transp-fftwf-naive
    transp-fftwf-lib-lmkl
    transp-fftwf-avx512-intr
    transp-fftwf-avx512-intr-ss
)
SER_BLK=(
    transp-fftwf-blocked
)
THR=(
    transp-fftwf-thrrow
    transp-fftwf-thrcol
    transp-fftwf-thrrow-avx512-intr
    transp-fftwf-thrrow-avx512-intr-ss
    transp-fftwf-thrcol-avx512-intr
    transp-fftwf-thrcol-avx512-intr-ss
)
THR_BLK=(
    transp-fftwf-thrrow-blocked
    transp-fftwf-thrcol-blocked
)

function capture() {
    local log=$1
    local t=$2
    shift 2
    if [ -z "$log" ]; then
        echo "Missing param: log"
        exit 1
    fi
    if [ -z "$t" ]; then
        echo "Missing param: t"
        exit 1
    fi
    if [ -f "$log" ]; then
        echo "File exists, skipping execution: ${log}"
        return 0
    fi
    echo "$@" "${EXTRA_PARAMS[@]}"
    numactl -l -C 0-$((t-1)) "$@" "${EXTRA_PARAMS[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    return "$rc"
}

function capture_ser() {
    local bin=$1
    for r in "${ROWS[@]}"; do
    for c in "${COLS[@]}"; do
        capture "${bin}_r-${r}_c-${c}.log" 1 "$bin" -r "$r" -c "$c"
    done # COLS
    done # ROWS
}

function capture_ser_blk() {
    local bin=$1
    for r in "${ROWS[@]}"; do
    for c in "${COLS[@]}"; do
    for b in "${BLKS[@]}"; do
        capture "${bin}_r-${r}_c-${c}_R-${b}_C-${b}.log" 1 "$bin" \
                -r "$r" -c "$c" -R "$b" -C "$b"
    done # BLKS
    done # COLS
    done # ROWS
}

function capture_thr() {
    local bin=$1
    for r in "${ROWS[@]}"; do
    for c in "${COLS[@]}"; do
    for t in "${THRS[@]}"; do
        capture "${bin}_r-${r}_c-${c}_t-${t}.log" "$t" "$bin" \
                -r "$r" -c "$c" -t "$t"
    done # THRS
    done # COLS
    done # ROWS
}

function capture_thr_blk() {
    local bin=$1
    for r in "${ROWS[@]}"; do
    for c in "${COLS[@]}"; do
    for t in "${THRS[@]}"; do
    for b in "${BLKS[@]}"; do
        capture "${bin}_r-${r}_c-${c}_t-${t}_R-${b}_C-${b}.log" "$t" "$bin" \
                -r "$r" -c "$c" -t "$t" -R "$b" -C "$b"
    done # BLKS
    done # THRS
    done # COLS
    done # ROWS
}


for bin in "${SER[@]}"; do
    capture_ser "$bin"
done
for bin in "${SER_BLK[@]}"; do
    capture_ser_blk "$bin"
done
for bin in "${THR[@]}"; do
    capture_thr "$bin"
done
for bin in "${THR_BLK[@]}"; do
    capture_thr_blk "$bin"
done
