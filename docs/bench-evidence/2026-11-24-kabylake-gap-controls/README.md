# Kaby Lake — what the VECTORIZED step after a gap is not, and how long a 100 µs sleep takes, 2026-11-24

Two controls from `PLAN.md`, "Pool wake-up and batch spacing": measurement 6
(the step of about 10 µs that a VECTORIZED call pays after a busy gap) and
the timer-coalescing candidate under "K, measured" (why the work-stealing
pool starts late after a busy gap and not after a sleeping caller).

## Conditions

- MacBook Pro 14,1, i7-7820HQ, macOS 13.7.8, AppleClang 15,
  `dev/v3.0.0-corvus-adoption` at `164cbce`, Release `libstats.a`;
  `batch_gap_bench` from the working tree with `--wide-gap` (a 256-bit add in
  the gap loop each turn).
- Screen locked, AC power. Seven gated items, each started below a 1-minute
  load of 2.0 (1.88, 1.71, 1.94, 1.99, then 1.68, 1.62, 1.65; `run.txt`).
- `indicative.txt` holds runs made by hand afterwards with no load gate. They
  are marked as such below and are not records.

| File | What |
|---|---|
| `ctl_busy.csv`, `ctl_busy_2.csv` | Gaussian and Exponential PDF, 4096 and 10,000, gaps 0, 250, 500 µs, 1, 2, 20 ms |
| `ctl_busy256.csv`, `ctl_busy256_2.csv` | the same with `--wide-gap`, run alternately with the two above |
| `sleep_probe.txt` | `sleep_probe.cpp.txt`: eight threads loop ten yields and `sleep_for(100 µs)`, as the pool's idle workers do |
| `indicative.txt` | `--fma-gap`, the library on `main`, and `step_probe.cpp.txt` |

## Result

VECTORIZED, µs per call, at gaps 0 / 250 µs / 500 µs / 1 ms / 2 ms / 20 ms.
The two runs of each kind agree within 0.9 µs in every cell:

| Row | Plain gap | 256-bit add in the gap |
|---|---|---|
| Gaussian, 4096 | 9.4 / 9.5–10.0 / 9.5 / 18.4–18.5 / 18.4 / 18.5–18.9 | 9.4 / 9.5–9.7 / 9.5 / 18.3–18.5 / 18.3–18.4 / 18.6–19.0 |
| Gaussian, 10,000 | 23.0 / 23.1–23.2 / 23.1 / 32.3–32.6 / 32.2–32.5 / 32.4–32.6 | 23.0 / 23.1–23.8 / 23.1 / 32.2–32.7 / 32.3–32.8 / 32.6–33.1 |
| Exponential, 4096 | 10.7 / 10.7–11.3 / 10.7 / 19.5–19.6 / 19.5 / 19.6–20.3 | 10.7 / 10.7 / 10.7 / 19.5–19.6 / 19.4–19.5 / 19.9–20.4 |
| Exponential, 10,000 | 26.5–27.3 / 26.3 / 26.3–26.4 / 34.8–35.3 / 34.9 / 35.4–36.2 | 26.3 / 26.3–26.4 / 26.4–27.3 / 34.9 / 35.4–35.9 / 35.5–35.8 |

1. **The step is 9 µs, the same at 4096 and at 10,000 elements and on both
   rows, and it sets in between a 500 µs and a 1 ms gap.** Up to 500 µs there
   is none; from 1 ms it is complete and does not grow to 20 ms.
2. **A 256-bit add in the gap loop does not remove it.** So it is not the
   upper half of the vector registers being powered down. A 256-bit FMA in the
   gap does not remove it either (indicative: 4.5 against 2.3 ns per element
   at 4096, the same as with a plain gap).
3. **`main`'s library has it too** (indicative): Gaussian at 4096 reads 4.5
   against 2.3 ns per element, Exponential 4.8 against 2.7 (the branch, in the
   same ungated run: 4.1 against 2.5 and 4.4 against 2.8). It is not corvus,
   Highway or anything else the branch added.
4. **It is smaller on a shorter batch and on a kernel with no transcendental**
   (indicative): 3.5–4.2 µs at 1024 elements for Gaussian and Exponential;
   0.2–0.4 µs at 1024 and 1.4–1.5 at 4096 for Uniform PDF.
5. **Loops outside the library pay 1.2 µs or less** (indicative,
   `step_probe`): none on register-only integer work, 0.5–0.6 µs on a
   4096-element scalar or 256-bit FMA pass over memory, 1.1–1.2 on a chain of
   8192 register FMAs and on a scalar pass over 1 MiB.
6. **Forced PARALLEL's delay has the same onset.** At 4096 the call costs
   22–25 µs back to back, 28–31 at 250 and 500 µs, and 79–84 from 1 ms (63
   and 76 in two cells); at
   10,000 it is 44–46, then 48–56 at 250 µs, 73–81 at 500 µs and 95–108
   from 1 ms. The 256-bit add changes none of it.
7. **`sleep_for(100 µs)` takes 121–123 µs at the median whatever the main
   thread does** (p10 108, p90 156–158, p99 163–166, over 286,000 sleeps in
   each case): spinning, sleeping 2 ms at a time, or 20 ms at a time. Timer
   coalescing does not stretch the idle workers' sleep in either case, so it
   does not explain why the work-stealing pool starts late after a busy gap
   and not after a sleeping caller.

## Not explained

- What the 9 µs is. It needs a gap of 1 ms, a kernel with some code and table
  footprint, and it saturates by 4096 elements. Untested candidates: a
  periodic interrupt on the calling core whose handler displaces the kernel's
  code, tables and translations; anything a counter would show (`PLAN.md`,
  measurement 3, which needs the user at the keyboard).
- Why the work-stealing pool's start delay follows a busy gap and not a
  sleeping caller, now that the sleep itself is ruled out.
