#!/usr/bin/env python3

"""Summarize a dispatcher profile bundle into derived CSV/JSON artifacts.

Reads strategy_profile_results.csv (canonical raw data from strategy_profile)
and produces crossovers.csv, best_strategies.csv, and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_strategy_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "distribution": row["Distribution"],
                    "operation": row["Operation"],
                    "batch_size": int(row["BatchSize"]),
                    "strategy": row["Strategy"],
                    "median_time_us": float(row["MedianTime_us"]),
                }
            )
    return rows


GroupKey = tuple[str, str]  # (distribution, operation)


def group_rows(
    rows: list[dict[str, Any]],
) -> dict[GroupKey, dict[int, dict[str, float]]]:
    """Group rows into {(dist, op): {batch_size: {strategy: time}}}."""
    grouped: dict[GroupKey, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        key = (row["distribution"], row["operation"])
        grouped[key][row["batch_size"]][row["strategy"]] = row["median_time_us"]
    return grouped


def best_strategy_at_size(timings: dict[str, float]) -> tuple[str, float]:
    best = min(timings.items(), key=lambda item: item[1])
    return best[0], best[1]


def find_first_crossover(
    size_map: dict[int, dict[str, float]],
    slower: str,
    faster: str,
) -> int | None:
    for batch_size in sorted(size_map.keys()):
        timings = size_map[batch_size]
        slower_time = timings.get(slower)
        faster_time = timings.get(faster)
        if slower_time is not None and faster_time is not None:
            if faster_time < slower_time:
                return batch_size
    return None


def build_crossover_rows(
    grouped: dict[GroupKey, dict[int, dict[str, float]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for (dist, op) in sorted(grouped.keys()):
        size_map = grouped[(dist, op)]
        s_to_v = find_first_crossover(size_map, "SCALAR", "VECTORIZED")
        v_to_p = find_first_crossover(size_map, "VECTORIZED", "PARALLEL")
        p_to_ws = find_first_crossover(size_map, "PARALLEL", "WORK_STEALING")

        largest_size = max(size_map.keys())
        best_strat, best_time = best_strategy_at_size(size_map[largest_size])

        results.append(
            {
                "distribution": dist,
                "operation": op,
                "scalar_to_vectorized": s_to_v,
                "vectorized_to_parallel": v_to_p,
                "parallel_to_work_stealing": p_to_ws,
                "best_strategy_at_max_size": best_strat,
                "best_time_us_at_max_size": round(best_time, 3),
                "max_batch_size": largest_size,
            }
        )
    return results


def build_best_strategy_rows(
    grouped: dict[GroupKey, dict[int, dict[str, float]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for (dist, op) in sorted(grouped.keys()):
        for batch_size in sorted(grouped[(dist, op)].keys()):
            timings = grouped[(dist, op)][batch_size]
            best_strat, best_time = best_strategy_at_size(timings)

            scalar_time = timings.get("SCALAR")
            speedup_vs_scalar = (
                round(scalar_time / best_time, 3)
                if scalar_time and best_time and best_time > 0
                else None
            )

            results.append(
                {
                    "distribution": dist,
                    "operation": op,
                    "batch_size": batch_size,
                    "best_strategy": best_strat,
                    "best_time_us": round(best_time, 3),
                    "scalar_time_us": round(scalar_time, 3) if scalar_time else None,
                    "speedup_vs_scalar": speedup_vs_scalar,
                }
            )
    return results


def safe_number(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 6)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_summary(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    crossover_rows: list[dict[str, Any]],
    best_strategy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    distributions = sorted({r["distribution"] for r in rows})
    operations = sorted({r["operation"] for r in rows})
    batch_sizes = sorted({r["batch_size"] for r in rows})

    strategy_wins: dict[str, int] = defaultdict(int)
    for row in best_strategy_rows:
        strategy_wins[row["best_strategy"]] += 1

    vectorized_never_wins = [
        {"distribution": r["distribution"], "operation": r["operation"]}
        for r in crossover_rows
        if r["scalar_to_vectorized"] is None
    ]

    return {
        "run_id": metadata["run_id"],
        "data_source": "strategy_profile_results.csv",
        "metadata": metadata,
        "coverage": {
            "distributions": distributions,
            "operations": operations,
            "batch_sizes": batch_sizes,
            "total_measurements": len(rows),
        },
        "strategy_win_counts": dict(
            sorted(strategy_wins.items(), key=lambda x: -x[1])
        ),
        "crossover_summary": {
            "groups": len(crossover_rows),
            "vectorized_never_wins": vectorized_never_wins,
            "parallel_crossover_sizes": [
                {
                    "distribution": r["distribution"],
                    "operation": r["operation"],
                    "vectorized_to_parallel": r["vectorized_to_parallel"],
                }
                for r in crossover_rows
                if r["vectorized_to_parallel"] is not None
            ],
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: safe_number(row.get(field)) for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate derived dispatcher profiling summary files for a saved run."
    )
    parser.add_argument("run_dir", help="Path to a dispatcher profile bundle directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    strategy_csv_path = run_dir / "strategy_profile_results.csv"

    if not strategy_csv_path.exists():
        print(f"Strategy profile CSV not found: {strategy_csv_path}")
        return 1

    metadata = load_metadata(metadata_path)
    rows = load_strategy_rows(strategy_csv_path)
    grouped = group_rows(rows)

    crossover_rows = build_crossover_rows(grouped)
    best_strategy_rows = build_best_strategy_rows(grouped)

    write_csv(
        run_dir / "crossovers.csv",
        crossover_rows,
        [
            "distribution",
            "operation",
            "scalar_to_vectorized",
            "vectorized_to_parallel",
            "parallel_to_work_stealing",
            "best_strategy_at_max_size",
            "best_time_us_at_max_size",
            "max_batch_size",
        ],
    )

    write_csv(
        run_dir / "best_strategies.csv",
        best_strategy_rows,
        [
            "distribution",
            "operation",
            "batch_size",
            "best_strategy",
            "best_time_us",
            "scalar_time_us",
            "speedup_vs_scalar",
        ],
    )

    summary = build_summary(metadata, rows, crossover_rows, best_strategy_rows)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Derived files written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
