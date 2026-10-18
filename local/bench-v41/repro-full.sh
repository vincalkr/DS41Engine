#!/usr/bin/env bash
# The request sequence that ended in the 11:25 GPU page fault (2026-10-18), against the server on 8150.
cd "$(dirname "$0")/../.."
B=local/bench-v41/chatbench.py
python3 $B "chi era Beatrice per Dante? rispondi in 3 frasi" --quiet
python3 $B --file local/bench-v41/long22k.txt --max 150 --quiet
python3 $B --file local/bench-v41/long22k-b.txt --max 150 --quiet
python3 $B "elenca i nove cerchi dell'Inferno, solo i nomi" --quiet
local/bench-v41/repro-park.sh
