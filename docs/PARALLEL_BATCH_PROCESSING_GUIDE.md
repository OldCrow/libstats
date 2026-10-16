# libstats Parallel and Batch Processing Guide

This guide describes the v2.x batch API for probability, log-probability, and CDF evaluation.

## Design model

Batch APIs use C++20 spans:

- input: `std::span<const double>`
- output: `std::span<double>`

The caller owns storage. The distribution writes results into the output span. Input and output spans must have the same size.

Manual strategy-specific v1.x methods were removed in v2.0.0. Use the unified span API plus `detail::PerformanceHint`.

## Basic usage

```cpp
#include "libstats/distributions/gaussian.h"
#include <span>
#include <vector>

int main() {
    auto normal = stats::GaussianDistribution::create(0.0, 1.0).unwrap();

    std::vector<double> x = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> pdf(x.size());

    normal.getProbability(std::span<const double>(x),
                          std::span<double>(pdf));
}
```

## Supported batch operations

All standard distributions expose the same three batch operations:

```cpp
dist.getProbability(values, results, hint);
dist.getLogProbability(values, results, hint);
dist.getCumulativeProbability(values, results, hint);
```

`hint` is optional. With the default hint, libstats auto-dispatches based on batch size, operation type, distribution type, and system capabilities.

There is no batch quantile or batch survival. `getQuantile`, `getSurvival`
and `getHazard` are scalar methods.

## One parameter state per batch

A batch call under `VECTORIZED`, `PARALLEL` or `WORK_STEALING` takes one
parameter snapshot and evaluates every element on it, so a concurrent setter
cannot mix two states into one result span. `SCALAR` is the exception: it
calls the locking scalar method per element, so a forced `SCALAR` batch, or
AUTO's choice of `SCALAR` for counts below the SIMD minimum, can mix states
under a concurrent setter. Per-element is the v2.x decision; one state under
every strategy is a v3 design input.

## Aliasing

The input and output spans of every batch overload must not overlap. The
size check (`values.size() == results.size()`, enforced) says nothing about
overlap, and several kernels re-read `values` after `results` has been
written — the Gaussian CDF tail fixups, the von Mises PDF/LogPDF NaN and
±inf fixups, the Gamma PDF pipeline, the LogNormal LogPDF support fixup and
the InverseGamma and Cauchy LogPDF fixups — so an in-place call
(`dist.getCumulativeProbability(buf, buf, hint)`) returns wrong values with
no error (review 2026-08-21, issue #112). Use a separate output buffer. The
lower-level `VectorOps` kernels *are* in-place safe by design; that guarantee
does not extend to the distribution layer.

#112 was resolved by documenting the contract rather than by making the tail
fixups aliasing-safe. Debug builds carry an assert
(`LIBSTATS_ASSERT_NO_OVERLAP`) beside the size check in
`detail::DispatchUtils::autoDispatch`. Every batch overload reaches
`autoDispatch`, directly or through its delegate; Cauchy and InverseGamma
LogPDF repeat the size check and the overlap assert on the caller's spans
before delegating. An aliased call therefore trips immediately in a Debug
build. Under `NDEBUG` the assert compiles to nothing and
nothing checks it: Release builds of an aliased call are silently wrong, as
before.

## Exceptions from batch kernels

`ParallelUtils::parallelFor`, which backs `Strategy::PARALLEL`, propagates an
exception thrown inside a chunk: it waits for every chunk, then harvests the
futures in order and rethrows the first exception (#118).
`ParallelUtils::parallelReduce` and `parallelStatOperation` share that contract
(#127); their combine step does not run when a chunk threw. The work-stealing
path does not — `WorkStealingPool::parallelFor` swallows chunk exceptions
silently (no log), and the rest of the failing chunk is skipped, because its
completion latch must be decremented on every path.
A batch call dispatched to `Strategy::WORK_STEALING` therefore loses a throwing
kernel's exception where `Strategy::PARALLEL` reports it.

## PerformanceHint

`detail::PerformanceHint` lets advanced callers influence dispatch without using removed strategy-specific APIs.

```cpp
stats::detail::PerformanceHint hint;
hint.strategy = stats::detail::PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT;

dist.getProbability(std::span<const double>(values),
                    std::span<double>(results),
                    hint);
```

Available preferences:

| Hint | Meaning |
|---|---|
| `AUTO` | Let libstats choose the strategy (default) |
| `FORCE_SCALAR` | Use scalar loop |
| `FORCE_VECTORIZED` | Use SIMD/vectorized path |
| `FORCE_PARALLEL` | Use standard parallel path |
| `MINIMIZE_LATENCY` | Prefer low overhead for small batches |
| `MAXIMIZE_THROUGHPUT` | The dispatcher's multi-threaded choice: `PARALLEL` on Windows and Linux, `WORK_STEALING` on hyper-threaded macOS |

No hint forces `WORK_STEALING`. When a call does run `WORK_STEALING`, it uses the shared `GlobalWorkStealingPool`, not a per-thread pool. This avoids spawning `N * hardware_concurrency` worker threads when multiple callers run concurrently.

`PARALLEL` (forced or chosen) falls back to the serial batch kernel below
`arch::get_min_elements_for_parallel()` (1,536–8,192 elements, by SIMD
tier), where forking costs more than it saves. Above it, the kernel runs
over slices on the thread pool.

## Strategy selection

The default dispatcher considers:

- batch size
- operation type (`PDF`, `LogPDF`, `CDF`)
- distribution type
- current CPU SIMD capabilities
- calibrated dispatch thresholds

Thresholds live in:

```text
include/libstats/core/dispatch_thresholds.h
```

These thresholds are architecture-specific. Performance results from AVX, AVX2, NEON, and AVX-512 systems should not be compared directly.

## Output storage

The output span must be pre-sized:

```cpp
std::vector<double> results(values.size());
dist.getLogProbability(values, results);
```

The function does not allocate output memory. This keeps batch paths predictable and avoids hidden allocations in hot loops.

## Error handling

Batch operations throw `std::invalid_argument` when:

- input and output span sizes differ
- the distribution-specific implementation rejects an invalid input shape

Individual out-of-support values are generally represented by distribution semantics:

- PDF/PMF: usually `0.0`
- LogPDF/LogPMF: usually `-inf`
- CDF: clamped to `[0, 1]` as appropriate

NaN propagates through every scalar and batch method of every distribution
(`getQuantile(NaN)` included); ±inf inputs follow the distribution's limits.

Discrete distributions take non-integer x by two rules: the PMF and log-PMF
round to the nearest integer (halves away from zero), while the CDF and
survival floor. So pmf(x) ≠ F(x) − F(x − 1) at non-integer x.

## Parallel batch fitting

All 27 distributions expose:

```cpp
Distribution::parallelBatchFit(datasets, results);
```

where:

- `datasets` is `std::vector<std::vector<double>>`
- `results` is `std::vector<Distribution>`

Example:

```cpp
std::vector<std::vector<double>> datasets = {sample1, sample2, sample3};
std::vector<stats::GaussianDistribution> results;

stats::GaussianDistribution::parallelBatchFit(datasets, results);
```

`parallelBatchFit` is still an output-parameter API in v2.x. Changing it to `Result<T>` is tracked separately as an API design decision.

## SIMD validation

Use:

```bash
./build/tools/simd_verification
```

This validates SIMD correctness and reports per-operation geometric mean speedups. It is the right tool to verify that SIMD paths are active and correct on a given machine.

## Timing tests

Timing-sensitive tests are labelled separately and should run alone on a quiet machine:

```bash
ctest --test-dir build --output-on-failure -j1 -L timing
```

Correctness suites should exclude timing tests:

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

## Migration from v1.x

Removed v1.x API:

- strategy-specific suffix methods for PDF, LogPDF, and CDF batch operations

v2.x replacement:

```cpp
stats::detail::PerformanceHint hint;
hint.strategy = stats::detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;

dist.getProbability(std::span<const double>(values),
                    std::span<double>(results),
                    hint);
```

Removed v1.x vector-returning base API:

- vector-returning batch helpers on `DistributionBase`

v2.x replacement:

```cpp
std::vector<double> y(values.size());
dist.getProbability(std::span<const double>(values),
                    std::span<double>(y));
```

## Practical guidance

- Use the default hint unless profiling shows a reason to override it.
- Use `MINIMIZE_LATENCY` for small batches in latency-sensitive code.
- Use `MAXIMIZE_THROUGHPUT` to take the platform's multi-threaded strategy for large batches.
- Pre-size output vectors and reuse them across calls.
- Profile in Release builds only.
- Re-run `system_inspector --quick` and `simd_verification` when moving between machines.
