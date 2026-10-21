#!/usr/bin/env bash
# gpu-perf.sh high|compute|auto - the power management level of the RX 7900 XTX cards (PCI id 0x744c); needs root:
#   sudo bash local/gpu-perf.sh high      the driver's `high` level - on the RX 7900 XTX here it holds the memory clock at
#                                         772 MHz (level 2), not 1249: VRAM ~580 GB/s instead of ~920
#   sudo bash local/gpu-perf.sh compute   manual level + the driver's COMPUTE power profile (min. active clock 1 GHz)
#   sudo bash local/gpu-perf.sh auto      back to the driver default (what the machine boots with)
# It does not persist across a reboot.  Measured 2026-10-20 with local/power-tune.sh on, 5 runs each: `high` is
# SLOWER than `auto` (17K 13.09 vs 13.63 tok/s, Sposi 16.89 vs 17.70; docs/ds4/V41_EXACT_FAST.md), because of that
# memory clock - keep `auto`.
set -e
mode=${1:?usage: gpu-perf.sh high|compute|auto}
for d in /sys/class/drm/card*/device; do
    [ -f "$d/power_dpm_force_performance_level" ] || continue
    [ "$(cat "$d/device")" = 0x744c ] || continue
    case $mode in
        high | auto) echo "$mode" > "$d/power_dpm_force_performance_level" ;;
        compute)
            echo manual > "$d/power_dpm_force_performance_level"
            echo 5 > "$d/pp_power_profile_mode" ;;
        *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    echo "$(grep PCI_SLOT_NAME "$d/uevent" | cut -d= -f2): $(cat "$d/power_dpm_force_performance_level"), profile $(grep '\*' "$d/pp_power_profile_mode" | awk '{print $2}')"
done
