#!/usr/bin/env bash
# restore-gsq.sh - copy the GSQ model and its Engram back from the USB SSD backup into models/ and check each file
# against the sha256 measured on 2026-10-18 (the GSQ's matches the GSQ repo's own SHA256SUMS).
set -euo pipefail
cd "$(dirname "$0")/.."
U=${1:-"/run/media/vincenzo/SSD NVME SAMSUNG/models"}
while read -r sha f; do
  [ -f "models/$f.sha-ok" ] && { echo "have  $f"; continue; }
  echo "$(date +%T) copy  $f"
  cp "$U/$f" "models/$f.part" && mv "models/$f.part" "models/$f"
  echo "$sha  models/$f" | sha256sum -c --quiet - && touch "models/$f.sha-ok" && echo "$(date +%T) ok    $f"
done <<'SUMS'
3897de644b77410e2d353ce9e1fc3f0bb1894d674f3db97374efdf7c155b33dd V41-GSQ/V41-GSQ.gguf
f4103b8109c562cc676a8c958de68879c9ba214890487a78bd59cf79295d0601 V41-GSQ/V41-MTP-Vision.Q2_K.gguf
4829279efb8c383ea70db2f4111609baad4fd4ad21e314546016da58412b1c05 engram/layers1/engram-L1.safetensors
dd0a2a66fb3a48882784babce5284f359123bec34578c3084877b4da1ab6b5b2 engram/layers14/engram-L14.safetensors
SUMS
echo done
