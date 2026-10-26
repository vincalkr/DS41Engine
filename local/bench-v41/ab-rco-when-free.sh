#!/usr/bin/env bash
# ab-rco-when-free.sh - start this in a terminal, then close the chat: it waits until the box is free and runs the A/B.
#
# The engine needs ~110 GiB of the 122 GiB and both cards' VRAM, so the A/B only runs while nothing else is loaded:
# whichever model server is up (the ds41e chat on :8150, or the sibling Strata host) holds 58 GiB and 25 GB of VRAM.
# Close that server yourself - this script never stops or starts one - and the loop below notices it by MemAvailable.
#
#   bash local/bench-v41/ab-rco-when-free.sh            # then close the chat; results in local/sweep-rco2
#   OUT=local/sweep-rco2 MODELS="rco rcogen" REPS=2 bash local/bench-v41/ab-rco-when-free.sh
#
# rcogen is the same pack with STRATA_NO_IQ512=1: ggml-cpu's fallback for every IQ1_M expert row instead of the
# AVX-512 kernel.  The two arms decode the same forced token stream (sweep.sh forces OUT/ref-*.ids), so a
# margin-diff of 0 also says the kernel is bit-exact where it runs.
set -u
cd "$(dirname "$0")/../.." || exit 1

MIN_AVAIL=${MIN_AVAIL:-86}          # GiB free before the A/B starts (sweep.sh's memguard asks for 82)
export OUT=${OUT:-local/sweep-rco2} REPS=${REPS:-2} MODELS=${MODELS:-"rco rcogen"}

echo "ab-rco-when-free: waiting for ${MIN_AVAIL} GiB available (close the chat; ./stop.sh for the ds41e one)"
while :; do
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail" -ge "$MIN_AVAIL" ] && ! ss -ltn 2>/dev/null | grep -q ':8150 '; then break; fi
    sleep 15
done
echo "ab-rco-when-free: ${avail} GiB free, starting the A/B ($(date '+%F %T'))"
exec bash local/bench-v41/ab-rco-iq2xxs.sh
