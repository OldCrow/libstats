#!/bin/bash
# M1 quiet runs of tools/bench/pool_wake_bench.cpp and an idle-burn probe (libstats PLAN.md,
# "Measurements this calls for", items 2, 5, 6, 7, 8). Each run its own process, gated on
# ambient CPU = 100 - top's idle %, two consecutive 10 s windows below 8%. Under caffeinate -is.
# Second version (2026-11-24, user's decision): mediaanalysisd and corespotlightd held one to
# two cores almost without pause (wake_m_20261123.sh got one run in 6.5 h), so they are
# suspended (SIGSTOP) before each gate and resumed (SIGCONT) after the run. A gate that has not
# passed in 5 minutes resumes them, waits 2 minutes and tries again, so they are never stopped
# for more than about 7 minutes at a time.
set -u
W=/Users/wolfman/Development/libstats
O=/Users/wolfman/Development/libstats-m-2026-11-21/wake
B=$O/pool_wake_bench
LOG=$O/run.txt
MAXAMB=8; WIN=10; DAEMONS='mediaanalysisd corespotlightd'
DEADLINE=$(( $(date +%s) + 8*3600 ))
caffeinate -is -w $$ &

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
ambient() { /usr/bin/top -l 2 -s $WIN -n 0 2>/dev/null | grep 'CPU usage' | tail -1 \
              | sed -E 's/.* ([0-9.]+)% idle.*/\1/' | awk '{printf "%.2f", 100-$1}'; }
top3() { /bin/ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{n=$2; sub(/.*\//,"",n); printf "%s%%%s ", $1, n}'; }
log() { echo "$(date '+%F %T') load=$(load1) $*" >> $LOG; }
hold() { for d in $DAEMONS; do pkill "-$1" -u "$(id -u)" -x "$d"; done; }
trap 'hold CONT' EXIT
trap 'exit 1' INT TERM
gate() {  # daemons suspended on return 0, running on return 1
  local q a t0
  while :; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "$1: DEADLINE passed, SKIPPED"; return 1; }
    hold STOP; q=0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt 300 ]; do
      a=$(ambient)
      if awk -v a="$a" -v m="$MAXAMB" 'BEGIN{exit !(a<m)}'; then
        q=$((q+1)); [ $q -ge 2 ] && { log "$1: gate passed (<${MAXAMB}%) ambient=${a}% top: $(top3)"; return 0; }
      else
        q=0; echo "$(date '+%F %T') [$1] ambient=${a}% top: $(top3)" >> $O/gate_samples.txt
      fi
    done
    hold CONT; log "$1: no window in 5 min with the daemons suspended; resumed, retry in 2 min"; sleep 120
  done
}
noise() { while kill -0 "$1" 2>/dev/null; do echo "$(date '+%T') load=$(load1) top: $(top3)" >> "$2"; sleep 15; done; }
item() {  # item <name> <ext> <command...>
  local name=$1 ext=$2; shift 2
  gate "$name" || return
  log "$name: start"
  "$@" > $O/$name.$ext 2> $O/$name.err & local pid=$!
  noise $pid $O/noise_$name.txt & local np=$!
  wait $pid; local rc=$?; wait $np 2>/dev/null
  hold CONT
  log "$name: end exit=$rc rows=$(wc -l < $O/$name.$ext | tr -d ' ')"
}

cd "$W" || exit
log "wake start sha=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ') daemons suspended per run: $DAEMONS"
item wake_plain          csv $B
item wake_init           csv $B --init
item wake_init_ws        csv $B --init --ws
item wake_plain_sleep    csv $B --sleep --gaps 0,2000,20000
item wake_init_ws_sleep  csv $B --init --ws --sleep --gaps 0,2000,20000
item wake_init_ws_fine   csv $B --init --ws --gaps 0,5,10,20,40,100
item wake_local_pool     csv $B --init --ws --local-pool 4,8 --gaps 0,2000
item idle_burn           txt $O/idle_burn
item idle_burn_plain     txt $O/idle_burn --plain
log "DONE"
