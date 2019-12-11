#!/bin/bash
set -e

function parse_time()
{
    grep "$2" "$1" | cut -d: -f2 | tr -d '[:space:]'
}

function log_to_csv() {
    local log=$1
    local fill transp
    fill=$(parse_time "$log" "fill")
    transp=$(parse_time "$log" "transpose")
    echo "${log},${fill},${transp}"
}

echo "File,Fill,Transpose"
for f in "$@"; do
    log_to_csv "$f"
done
