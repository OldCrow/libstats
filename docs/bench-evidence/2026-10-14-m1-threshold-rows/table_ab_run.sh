#!/bin/bash
# ABAB: current table, proposed table, current, proposed; back to back, same machine state.
set -u
R=/Users/wolfman/Development/libstats-m-20261013/rows
log(){ echo "$(date '+%T') load=$(sysctl -n vm.loadavg | awk '{print $2}') $*"; }
for pass in 1 2; do
  log "old pass $pass start"; $R/table_ab_old > $R/tab_old_$pass.csv 2>/dev/null; log "old pass $pass end rows=$(wc -l < $R/tab_old_$pass.csv)"
  log "new pass $pass start"; $R/table_ab_new > $R/tab_new_$pass.csv 2>/dev/null; log "new pass $pass end rows=$(wc -l < $R/tab_new_$pass.csv)"
done
log DONE
