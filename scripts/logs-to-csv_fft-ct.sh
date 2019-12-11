#!/bin/bash
set -e

function parse_time()
{
    grep "$2" "$1" | cut -d: -f2 | tr -d '[:space:]'
}

function log_to_csv() {
    local log=$1
    local fill init fft1 transp fft2
    fill=$(parse_time "$log" "fill")
    init=$(parse_time "$log" "init")
    fft1=$(parse_time "$log" "fft-1d-1")
    transp=$(parse_time "$log" "transpose")
    fft2=$(parse_time "$log" "fft-1d-2")
    echo "${log},${fill},${init},${fft1},${transp},${fft2}"
}

echo "File,Fill,Init,FFT1,Transpose,FFT2"
for f in "$@"; do
    log_to_csv "$f"
done
