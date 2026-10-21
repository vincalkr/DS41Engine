#!/usr/bin/env bash
# power-tune.sh on|off|show - the idle power states decode trips over (on/off need root; nothing persists across a
# reboot).  Measured 2026-10-20 (docs/ds4/V41_EXACT_FAST.md, "Where a two-card token goes"): a 9.7 MiB expert read
# every 10 ms took 2.77 ms and every 20 ms 5.54 ms, against ~1.2 ms when the drive and a CPU core are kept busy;
# decode's file reads come 10-20 ms apart, and the engine paid 2.5-2.9 ms per read in its slow runs.
#   on    NVMe APST off (the drive stays in its active power state), the CPUs' deepest idle state (C3, 350 us exit
#         latency here) off, the cpufreq governor `performance`
#   off   back to what this machine boots with: APST tolerance 100000 us, C3 on, governor `powersave` with EPP
#         `balance_performance`
#   show  the current values
#   sudo bash local/power-tune.sh on
set -e
NVME=${NVME:-nvme0}
apst=/sys/class/nvme/$NVME/power/pm_qos_latency_tolerance_us
show() {
    echo "NVMe $NVME APST latency tolerance: $(cat $apst) us (0 = APST off)"
    for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
        echo "cpu idle $(cat $s/name): latency $(cat $s/latency) us, disabled on cpu0: $(cat $s/disable)"
    done
    echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), EPP $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
}
deep_states() {   # the idle states with an exit latency over 100 us
    for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
        [ "$(cat $s/latency)" -gt 100 ] && basename $s
    done
}
case ${1:?usage: power-tune.sh on|off|show} in
    on)
        echo 0 > $apst
        for st in $(deep_states); do for c in /sys/devices/system/cpu/cpu*/cpuidle/$st/disable; do echo 1 > $c; done; done
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $g; done
        show ;;
    off)
        echo 100000 > $apst
        for st in $(deep_states); do for c in /sys/devices/system/cpu/cpu*/cpuidle/$st/disable; do echo 0 > $c; done; done
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo powersave > $g; done
        for e in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo balance_performance > $e; done
        show ;;
    show) show ;;
    *) echo "usage: power-tune.sh on|off|show" >&2; exit 2 ;;
esac
