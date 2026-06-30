/**
 * @file threshold_validator.cpp
 * @brief Dispatch threshold validation tool
 *
 * Reads a strategy_profile_results.csv file (produced by strategy_profile --output-csv)
 * and compares the measured VECTORIZED→PARALLEL crossover batch sizes against the
 * V→P thresholds compiled into dispatch_thresholds.h for the active SIMD level.
 *
 * The measured V→P crossover is the smallest batch size where
 * min(PARALLEL, WORK_STEALING) first beats VECTORIZED, matching the definition
 * in scripts/PROFILING_METHOD.md.
 *
 * This closes the loop in the threshold recalibration workflow:
 *   1. Run: ./build/tools/strategy_profile --large --output-csv results.csv
 *   2. Run: ./build/tools/threshold_validator results.csv
 *   3. Update dispatch_thresholds.h for any highlighted entry.
 *
 * NOTE: pass strategy_profile_results.csv (raw per-batch-size data), NOT
 * crossovers.csv (the derived summary produced by summarize_dispatcher_profile.py).
 *
 * Interpretation:
 *   MATCH     — compiled V→P threshold within 2× of measured crossover
 *   UPDATE↑   — compiled threshold below measured; parallel dispatched too eagerly
 *   UPDATE↓   — compiled threshold above measured; parallel win starts earlier
 *   ADD THRESH — crossover detected but compiled threshold is NEVER
 *   SET NEVER? — compiled threshold set but no crossover seen in this run
 *   BOTH NEVER — no crossover and not calibrated (expected for disabled ops)
 *
 * Usage:
 *   ./build/tools/threshold_validator <strategy_profile_results.csv>
 *   ./build/tools/threshold_validator <csv> --simd avx2
 *   ./build/tools/threshold_validator <csv> --simd avx512
 *
 * --simd <level>  Override the SIMD table to compare against. Without this
 *   flag the active machine's level is used. Valid levels (case-insensitive):
 *   neon, sse2, avx, avx2, avx512, none.
 *
 * The CSV must have the format produced by strategy_profile --output-csv:
 *   Distribution,Operation,BatchSize,Strategy,MedianTime_us
 */

#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/distribution_meta.h"
#include "tool_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace stats;
using namespace stats::detail;

// ─────────────────────────────────────────────────────────────────────────────
// CSV parsing
// ─────────────────────────────────────────────────────────────────────────────

struct ProfileRow {
    std::string distribution;
    std::string operation;
    size_t batch_size;
    std::string strategy;
    double median_time_us;
};

std::vector<ProfileRow> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open CSV: " + path);

    std::vector<ProfileRow> rows;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (first) {
            first = false;
            continue;
        }  // skip header
        if (line.empty())
            continue;

        std::istringstream ss(line);
        std::string tok;
        ProfileRow r{};
        int col = 0;
        while (std::getline(ss, tok, ',')) {
            switch (col++) {
                case 0:
                    r.distribution = tok;
                    break;
                case 1:
                    r.operation = tok;
                    break;
                case 2:
                    r.batch_size = static_cast<size_t>(std::stoull(tok));
                    break;
                case 3:
                    r.strategy = tok;
                    break;
                case 4:
                    r.median_time_us = std::stod(tok);
                    break;
            }
        }
        if (col >= 5)
            rows.push_back(r);
    }
    return rows;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find first crossover: the smallest batch_size where strategy B beats A
// ─────────────────────────────────────────────────────────────────────────────
std::optional<size_t> find_crossover(const std::map<size_t, std::map<std::string, double>>& timings,
                                     const std::string& from_strategy,
                                     const std::string& to_strategy) {
    for (const auto& [sz, strat_times] : timings) {
        auto it_from = strat_times.find(from_strategy);
        auto it_to = strat_times.find(to_strategy);
        if (it_from == strat_times.end() || it_to == strat_times.end())
            continue;
        if (it_to->second < it_from->second)
            return sz;
    }
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Map display distribution name to DistributionType
// ─────────────────────────────────────────────────────────────────────────────
std::optional<DistributionType> name_to_type(const std::string& name) {
    for (size_t i = 0; i < kDistributionTypeCount; ++i) {
        auto dt = static_cast<DistributionType>(i);
        if (distributionDisplayName(dt) == name || distributionEnumName(dt) == name)
            return dt;
    }
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Map operation string to OperationType
// ─────────────────────────────────────────────────────────────────────────────
std::optional<OperationType> op_to_type(const std::string& op) {
    if (op == "PDF")
        return OperationType::PDF;
    if (op == "LogPDF")
        return OperationType::LOG_PDF;
    if (op == "CDF")
        return OperationType::CDF;
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Classify the comparison between compiled and measured crossover
// ─────────────────────────────────────────────────────────────────────────────
std::string classify(std::optional<size_t> compiled_opt, std::optional<size_t> measured_opt) {
    constexpr long NEVER_SENTINEL = -1L;

    long compiled = compiled_opt.has_value() ? static_cast<long>(*compiled_opt) : NEVER_SENTINEL;
    long measured = measured_opt.has_value() ? static_cast<long>(*measured_opt) : NEVER_SENTINEL;

    if (compiled == NEVER_SENTINEL && measured == NEVER_SENTINEL)
        return "BOTH NEVER";
    if (compiled == NEVER_SENTINEL && measured != NEVER_SENTINEL)
        return "ADD THRESH";
    if (compiled != NEVER_SENTINEL && measured == NEVER_SENTINEL)
        return "SET NEVER?";

    // Both finite: within 2x is a match
    double ratio = static_cast<double>(compiled) / static_cast<double>(measured);
    if (ratio >= 0.5 && ratio <= 2.0)
        return "MATCH";
    if (compiled < measured)
        return "UPDATE↑";  // should raise threshold
    return "UPDATE↓";      // should lower threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse optional --simd <level> override
// ─────────────────────────────────────────────────────────────────────────────
std::optional<stats::arch::simd::SIMDLevel> parse_simd_level(const std::string& s) {
    using L = stats::arch::simd::SIMDLevel;
    // Case-insensitive compare via simple lowercase fold
    std::string ls;
    ls.reserve(s.size());
    for (char c : s)
        ls += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ls == "neon")
        return L::NEON;
    if (ls == "sse2")
        return L::SSE2;
    if (ls == "avx")
        return L::AVX;
    if (ls == "avx2")
        return L::AVX2;
    if (ls == "avx512")
        return L::AVX512;
    if (ls == "none")
        return L::None;
    return std::nullopt;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: threshold_validator <strategy_profile_results.csv> [--simd <level>]\n"
                  << "  level: neon  sse2  avx  avx2  avx512  none\n"
                  << "  Default: active machine SIMD level\n"
                  << "\nGenerates the CSV with:\n"
                  << "  ./build/tools/strategy_profile --large --output-csv results.csv\n";
        return 1;
    }

    const std::string csv_path = argv[1];

    // Parse optional --simd override
    std::optional<stats::arch::simd::SIMDLevel> simd_override;
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--simd") {
            simd_override = parse_simd_level(argv[i + 1]);
            if (!simd_override) {
                std::cerr << "Unknown --simd level: " << argv[i + 1]
                          << "\nValid levels: neon  sse2  avx  avx2  avx512  none\n";
                return 1;
            }
            break;
        }
    }

    return stats::detail::detail::runTool("Threshold Validator", [&]() {
        // Resolve and display the SIMD level being compared.
        const auto active_level =
            simd_override.value_or(stats::arch::simd::SIMDPolicy::getBestLevel());
        std::string level_label;
        switch (active_level) {
            case stats::arch::simd::SIMDLevel::AVX512:
                level_label = "AVX-512";
                break;
            case stats::arch::simd::SIMDLevel::AVX2:
                level_label = "AVX2";
                break;
            case stats::arch::simd::SIMDLevel::AVX:
                level_label = "AVX";
                break;
            case stats::arch::simd::SIMDLevel::SSE2:
                level_label = "SSE2";
                break;
            case stats::arch::simd::SIMDLevel::NEON:
                level_label = "NEON";
                break;
            default:
                level_label = "None";
                break;
        }
        stats::detail::detail::displayToolHeader(
            "Dispatch Threshold Validator",
            "Comparing kTable: " + level_label +
                (simd_override ? " (--simd override)" : " (active machine)") +
                " vs strategy_profile crossovers");

        // Load CSV
        auto rows = read_csv(csv_path);
        std::cout << "Loaded " << rows.size() << " rows from " << csv_path << "\n\n";

        // Group by (distribution, operation) → batch_size → {strategy → time}
        using GroupKey = std::pair<std::string, std::string>;
        std::map<GroupKey, std::map<size_t, std::map<std::string, double>>> groups;
        for (const auto& r : rows) {
            groups[{r.distribution, r.operation}][r.batch_size][r.strategy] = r.median_time_us;
        }

        stats::detail::detail::ColumnFormatter fmt({16, 8, 14, 14, 14, 12});
        std::cout << fmt.formatRow({"Distribution", "Op", "Compiled V→P", "Measured S→V",
                                    "Measured V→P", "Status"})
                  << "\n";
        std::cout << fmt.getSeparator() << "\n";

        int n_match = 0, n_update = 0, n_other = 0;

        for (const auto& [key, timings] : groups) {
            const auto& [dist_name, op_name] = key;

            auto dt_opt = name_to_type(dist_name);
            auto op_opt = op_to_type(op_name);
            if (!dt_opt || !op_opt)
                continue;

            // Compiled V→P threshold: use --simd override if given, else active level.
            // getParallelThreshold() returns the VECTORIZED→PARALLEL threshold from
            // dispatch_thresholds.h (SIZE_MAX = NEVER = VECTORIZED always preferred).
            const auto simd_level =
                simd_override.value_or(stats::arch::simd::SIMDPolicy::getBestLevel());
            size_t compiled_vp_raw =
                stats::detail::getParallelThreshold(simd_level, *dt_opt, *op_opt);
            std::optional<size_t> compiled_vp =
                (compiled_vp_raw == std::numeric_limits<size_t>::max())
                    ? std::nullopt
                    : std::make_optional(compiled_vp_raw);

            // Measured crossovers from the raw strategy_profile_results.csv.
            // S→V: informational only (not in dispatch_thresholds.h).
            // V→P: min(PARALLEL, WORK_STEALING) crossover per PROFILING_METHOD.md.
            auto measured_sv = find_crossover(timings, "SCALAR", "VECTORIZED");
            auto measured_vp = find_crossover(timings, "VECTORIZED", "PARALLEL");
            auto measured_vp2 = find_crossover(timings, "VECTORIZED", "WORK_STEALING");
            // Take min(PARALLEL, WORK_STEALING) — matches the Step-2 V→P definition.
            if (!measured_vp.has_value())
                measured_vp = measured_vp2;
            else if (measured_vp2.has_value())
                measured_vp = std::min(*measured_vp, *measured_vp2);

            // Primary comparison: compiled V→P vs measured V→P.
            std::string vp_status = classify(compiled_vp, measured_vp);

            auto fmt_thresh = [](std::optional<size_t> v) -> std::string {
                return v.has_value() ? std::to_string(*v) : "NEVER";
            };

            std::string status = vp_status;
            if (status == "MATCH") {
                ++n_match;
            } else if (status.find("UPDATE") != std::string::npos) {
                ++n_update;
            } else {
                ++n_other;
            }

            std::cout << fmt.formatRow({dist_name, op_name, fmt_thresh(compiled_vp),
                                        fmt_thresh(measured_sv), fmt_thresh(measured_vp), status})
                      << "\n";
        }

        std::cout << "\nSummary: " << n_match << " MATCH, " << n_update << " UPDATE, " << n_other
                  << " other\n\n";
        std::cout
            << "UPDATE↑: compiled V→P too low; parallel dispatched more eagerly than needed\n"
            << "UPDATE↓: compiled V→P too high; parallel wins earlier than the table assumes\n"
            << "ADD THRESH: V→P crossover detected but compiled threshold is NEVER\n"
            << "SET NEVER?: compiled threshold set but no V→P crossover seen in this run\n"
            << "BOTH NEVER: no crossover and not calibrated (expected for VECTORIZED-dominant "
               "ops)\n\n"
            << "Matching tolerance: compiled within 2x of measured is MATCH.\n"
            << "Input must be strategy_profile_results.csv (raw data), not crossovers.csv.\n"
            << "Edit include/core/dispatch_thresholds.h to apply changes.\n";
    });
}
