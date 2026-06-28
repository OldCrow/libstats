/**
 * @file threshold_validator.cpp
 * @brief Dispatch threshold validation tool
 *
 * Reads a strategy_profile CSV file (produced by `./strategy_profile --export`)
 * and compares the measured SCALAR→VECTORIZED and VECTORIZED→PARALLEL crossover
 * batch sizes against the thresholds compiled into dispatch_thresholds.h.
 *
 * This closes the loop in the threshold recalibration workflow:
 *   1. Run `./strategy_profile --large --export` to generate the CSV.
 *   2. Run `./threshold_validator <csv_path>` to see which entries differ.
 *   3. Update dispatch_thresholds.h for any highlighted entry.
 *
 * Interpretation:
 *   MATCH     — compiled threshold equals measured crossover (±factor 2)
 *   UPDATE↑   — compiled threshold is larger than measured; consider lowering
 *   UPDATE↓   — compiled threshold is smaller than measured; consider raising
 *   NEVER/NEW — one side has NEVER (no crossover observed or not calibrated)
 *
 * Usage:
 *   ./build/tools/threshold_validator <strategy_profile_results.csv>
 *
 * The CSV must have the format produced by strategy_profile:
 *   Distribution,Operation,BatchSize,Strategy,MedianTime_us
 */

#include "tool_utils.h"

#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/distribution_meta.h"

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
    if (!f) throw std::runtime_error("Cannot open CSV: " + path);

    std::vector<ProfileRow> rows;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (first) { first = false; continue; }  // skip header
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string tok;
        ProfileRow r{};
        int col = 0;
        while (std::getline(ss, tok, ',')) {
            switch (col++) {
                case 0: r.distribution = tok;  break;
                case 1: r.operation    = tok;  break;
                case 2: r.batch_size   = static_cast<size_t>(std::stoull(tok)); break;
                case 3: r.strategy     = tok;  break;
                case 4: r.median_time_us = std::stod(tok); break;
            }
        }
        if (col >= 5) rows.push_back(r);
    }
    return rows;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find first crossover: the smallest batch_size where strategy B beats A
// ─────────────────────────────────────────────────────────────────────────────
std::optional<size_t> find_crossover(
    const std::map<size_t, std::map<std::string, double>>& timings,
    const std::string& from_strategy, const std::string& to_strategy)
{
    for (const auto& [sz, strat_times] : timings) {
        auto it_from = strat_times.find(from_strategy);
        auto it_to   = strat_times.find(to_strategy);
        if (it_from == strat_times.end() || it_to == strat_times.end()) continue;
        if (it_to->second < it_from->second) return sz;
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
    if (op == "PDF")    return OperationType::PDF;
    if (op == "LogPDF") return OperationType::LOG_PDF;
    if (op == "CDF")    return OperationType::CDF;
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Classify the comparison between compiled and measured crossover
// ─────────────────────────────────────────────────────────────────────────────
std::string classify(std::optional<size_t> compiled_opt, std::optional<size_t> measured_opt) {
    constexpr long NEVER_SENTINEL = -1L;

    long compiled  = compiled_opt.has_value()
        ? static_cast<long>(*compiled_opt) : NEVER_SENTINEL;
    long measured  = measured_opt.has_value()
        ? static_cast<long>(*measured_opt) : NEVER_SENTINEL;

    if (compiled == NEVER_SENTINEL && measured == NEVER_SENTINEL) return "BOTH NEVER";
    if (compiled == NEVER_SENTINEL && measured != NEVER_SENTINEL) return "ADD THRESH";
    if (compiled != NEVER_SENTINEL && measured == NEVER_SENTINEL) return "SET NEVER?";

    // Both finite: within 2x is a match
    double ratio = static_cast<double>(compiled) / static_cast<double>(measured);
    if (ratio >= 0.5 && ratio <= 2.0) return "MATCH";
    if (compiled < measured) return "UPDATE↑";  // should raise threshold
    return "UPDATE↓";                            // should lower threshold
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: threshold_validator <strategy_profile_results.csv>\n"
                  << "\nGenerates the CSV with:\n"
                  << "  ./build/tools/strategy_profile --export\n";
        return 1;
    }

    const std::string csv_path = argv[1];

    return stats::detail::detail::runTool("Threshold Validator", [&]() {
        stats::detail::detail::displayToolHeader(
            "Dispatch Threshold Validator",
            "Compares compiled thresholds in dispatch_thresholds.h against "
            "strategy_profile crossovers");

        // Load CSV
        auto rows = read_csv(csv_path);
        std::cout << "Loaded " << rows.size() << " rows from " << csv_path << "\n\n";

        // Group by (distribution, operation) → batch_size → {strategy → time}
        using GroupKey = std::pair<std::string, std::string>;
        std::map<GroupKey, std::map<size_t, std::map<std::string, double>>> groups;
        for (const auto& r : rows) {
            groups[{r.distribution, r.operation}][r.batch_size][r.strategy] = r.median_time_us;
        }

        stats::detail::detail::ColumnFormatter fmt({16, 8, 14, 14, 14, 14, 12});
        std::cout << fmt.formatRow({"Distribution", "Op",
            "Compiled S→V", "Measured S→V", "Compiled V→P", "Measured V→P", "Status"})
                  << "\n";
        std::cout << fmt.getSeparator() << "\n";

        int n_match = 0, n_update = 0, n_other = 0;

        for (const auto& [key, timings] : groups) {
            const auto& [dist_name, op_name] = key;

            auto dt_opt = name_to_type(dist_name);
            auto op_opt = op_to_type(op_name);
            if (!dt_opt || !op_opt) continue;

            // Compiled threshold — returns std::numeric_limits<size_t>::max() for NEVER
            size_t compiled_sv_raw = parallelThresholdFromTable(*dt_opt, *op_opt);
            std::optional<size_t> compiled_sv = (compiled_sv_raw == std::numeric_limits<size_t>::max())
                ? std::nullopt : std::make_optional(compiled_sv_raw);

            // Measured crossovers from CSV
            auto measured_sv = find_crossover(timings, "SCALAR", "VECTORIZED");
            auto measured_vp = find_crossover(timings, "VECTORIZED", "PARALLEL");

            // Note: compiled threshold is the S→V threshold; V→P is derived from parallel
            // infrastructure. We report both for completeness.
            auto measured_vp2 = find_crossover(timings, "VECTORIZED", "WORK_STEALING");
            // Use whichever parallel crossover appears first
            if (!measured_vp.has_value() && measured_vp2.has_value())
                measured_vp = measured_vp2;

            std::string sv_status = classify(compiled_sv, measured_sv);

            auto fmt_thresh = [](std::optional<size_t> v) -> std::string {
                return v.has_value() ? std::to_string(*v) : "NEVER";
            };

            std::string status = sv_status;
            if      (status == "MATCH")       { ++n_match; }
            else if (status.find("UPDATE") != std::string::npos) { ++n_update; }
            else    { ++n_other; }

            std::cout << fmt.formatRow({dist_name, op_name,
                fmt_thresh(compiled_sv), fmt_thresh(measured_sv),
                "N/A", fmt_thresh(measured_vp),
                status}) << "\n";
        }

        std::cout << "\nSummary: " << n_match << " MATCH, " << n_update
                  << " UPDATE, " << n_other << " other\n\n";
        std::cout << "UPDATE↑: raise the compiled threshold (SCALAR is being preferred too eagerly)\n"
                  << "UPDATE↓: lower the compiled threshold (VECTORIZED should fire sooner)\n"
                  << "SET NEVER?: compiled threshold set but no crossover seen in profiling data\n"
                  << "ADD THRESH: crossover detected but compiled threshold is NEVER\n\n"
                  << "Matching tolerance: compiled within 2x of measured is MATCH.\n"
                  << "Edit include/core/dispatch_thresholds.h to apply changes.\n";
    });
}
