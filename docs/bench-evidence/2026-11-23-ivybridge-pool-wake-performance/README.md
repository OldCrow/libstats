# Ivy Bridge — pool wake-up under the `performance` governor, 2026-11-23

The second half of a two-governor comparison. Conditions, file list and
findings for both halves are in
`../2026-11-22-ivybridge-pool-wake/README.md`; the script is
`pool_wake_run.sh.txt` there.

This directory holds the runs made with the CPU frequency governor set to
`performance` (`environment.txt`). The user switched it by hand for these
runs and restored `schedutil` afterwards; it is not the machine's normal
state, so these numbers name a cause and are not what a caller on this
machine gets by default.
