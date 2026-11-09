#!/bin/bash
# Read-only power and clock metadata. Hostnames and PCI ids are not printed.
echo "== rocm-smi"; rocm-smi --showperflevel --showpowerprofile --showmaxpower --showpower --showclocks --showclkvolt --showvoltage --showproductname 2>&1 | grep -vE "^$|Serial|UUID|Unique" | sed -E 's/[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]/PCI/g' | cut -c1-140 | head -150
echo "== amd-smi"; which amd-smi && amd-smi metric --power --clock 2>&1 | head -60
for d in /sys/class/drm/card[0-9]*/device; do c=$(echo $d|cut -d/ -f5); echo "== sysfs $c"; ls $d | grep -E "gpu_metrics|pp_|power|hwmon|mem_info|gpu_busy" | tr '\n' ' '; echo
  for f in power_dpm_force_performance_level power_dpm_state pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_od_clk_voltage pp_power_profile_mode gpu_busy_percent; do [ -r $d/$f ] && { echo "-- $f"; cat $d/$f 2>&1 | head -40; }; done
  for h in $d/hwmon/hwmon*; do for f in name power1_average power1_input power1_cap power1_cap_max power1_cap_min power1_cap_default freq1_input freq1_label freq2_input freq2_label in0_input in0_label temp1_input temp1_crit temp2_input temp2_crit temp3_input temp3_crit; do [ -r $h/$f ] && echo "$f $(cat $h/$f 2>&1)"; done; done
  [ -r $d/gpu_metrics ] && { echo "-- gpu_metrics"; od -A d -t x1 -N 128 $d/gpu_metrics; }
  for f in numa_node thermal_throttling/*; do [ -r $d/$f ] && echo "$f $(cat $d/$f 2>&1)"; done
done
