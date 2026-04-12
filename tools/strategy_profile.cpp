/**
 * @file strategy_profile.cpp
 * @brief Canonical forced-strategy profiler for dispatcher threshold tuning
 *
 * Profiles forced SCALAR, VECTORIZED, PARALLEL, and WORK_STEALING execution
 * across all dispatcher-supported distributions, core batch operations, and a
 * representative batch-size sweep. The output is intended to be the canonical
 * raw dataset for tuning dispatcher thresholds.
 */

#include "libstats/distributions/beta.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "tool_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

using namespace stats;
using namespace stats::detail;
using namespace std::chrono;

namespace {

constexpr int DEFAULT_RNG_SEED = 42;
constexpr int WARMUP_ITERATIONS = 3;
constexpr int TIMING_REPEATS = 7;
constexpr const char* RESULTS_CSV_FILENAME = "strategy_profile_results.csv";

enum class ProfileOperation { PDF, LOG_PDF, CDF };

struct StrategyProfileResult {
    std::string distribution;
    std::string operation;
    std::size_t batch_size;
    Strategy strategy;
    double median_time_us;
};

double median_us(std::vector<double>& timings) {
    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

std::string operation_to_string(ProfileOperation operation) {
    switch (operation) {
        case ProfileOperation::PDF:
            return "PDF";
        case ProfileOperation::LOG_PDF:
            return "LogPDF";
        case ProfileOperation::CDF:
            return "CDF";
        default:
            return "Unknown";
    }
}

constexpr std::array<ProfileOperation, 3> OPERATIONS = {
    ProfileOperation::PDF, ProfileOperation::LOG_PDF, ProfileOperation::CDF};

constexpr std::array<Strategy, 4> STRATEGIES = {Strategy::SCALAR, Strategy::VECTORIZED,
                                                Strategy::PARALLEL, Strategy::WORK_STEALING};

}  // namespace

class StrategyProfiler {
   public:
    explicit StrategyProfiler(bool include_large) : gen_(DEFAULT_RNG_SEED) {
        initialize_batch_sizes(include_large);
    }

    void run(const std::string& output_csv_path) {
        stats::detail::detail::displayToolHeader(
            "Strategy Profile", "Forced-strategy timing profiler for dispatcher threshold tuning");

        std::cout << "Batch sizes:";
        for (auto size : batch_sizes_) {
            std::cout << " " << size;
        }
        std::cout << "\n\n";

        profile_all_distributions();
        print_summary();
        save_results(output_csv_path);
    }

   private:
    std::mt19937 gen_;
    std::vector<StrategyProfileResult> results_;
    std::vector<std::size_t> batch_sizes_;

    void initialize_batch_sizes(bool include_large) {
        batch_sizes_ = {8,    16,   32,    64,    128,   256,    512,    1000,
                        2000, 5000, 10000, 20000, 50000, 100000, 250000, 500000};

        if (include_large) {
            batch_sizes_.push_back(1000000);
            batch_sizes_.push_back(2000000);
        }
    }

    void profile_all_distributions() {
        profile_uniform_distribution();
        profile_gaussian_distribution();
        profile_exponential_distribution();
        profile_discrete_distribution();
        profile_poisson_distribution();
        profile_gamma_distribution();
        profile_student_t_distribution();
        profile_beta_distribution();
        profile_chi_squared_distribution();
    }

    template <typename Distribution, typename Generator>
    void profile_distribution(const std::string& distribution_name,
                              const Distribution& distribution, Generator&& generator) {
        stats::detail::detail::subsectionHeader(distribution_name + " Strategy Profile");

        for (auto batch_size : batch_sizes_) {
            std::cout << "  Profiling batch size " << batch_size << "..." << std::flush;

            const auto input_values = generator(batch_size);

            for (auto operation : OPERATIONS) {
                for (auto strategy : STRATEGIES) {
                    const double median_time_us =
                        benchmark_strategy(distribution, input_values, operation, strategy);

                    results_.push_back({distribution_name, operation_to_string(operation),
                                        batch_size, strategy, median_time_us});
                }
            }

            std::cout << " ✓\n";
        }
        std::cout << "\n";
    }

    template <typename Distribution>
    double benchmark_strategy(const Distribution& distribution,
                              const std::vector<double>& input_values, ProfileOperation operation,
                              Strategy strategy) const {
        std::vector<double> output_values(input_values.size());
        std::span<const double> input_span(input_values);
        std::span<double> output_span(output_values);

        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            perform_operation(distribution, input_span, output_span, operation, strategy);
        }

        std::vector<double> timings_us;
        timings_us.reserve(TIMING_REPEATS);

        for (int i = 0; i < TIMING_REPEATS; ++i) {
            const auto start = high_resolution_clock::now();
            perform_operation(distribution, input_span, output_span, operation, strategy);
            const auto end = high_resolution_clock::now();
            timings_us.push_back(duration<double, std::micro>(end - start).count());
        }

        return median_us(timings_us);
    }

    template <typename Distribution>
    void perform_operation(const Distribution& distribution, std::span<const double> input_values,
                           std::span<double> output_values, ProfileOperation operation,
                           Strategy strategy) const {
        switch (operation) {
            case ProfileOperation::PDF:
                distribution.getProbabilityWithStrategy(input_values, output_values, strategy);
                break;
            case ProfileOperation::LOG_PDF:
                distribution.getLogProbabilityWithStrategy(input_values, output_values, strategy);
                break;
            case ProfileOperation::CDF:
                distribution.getCumulativeProbabilityWithStrategy(input_values, output_values,
                                                                  strategy);
                break;
        }
    }

    void profile_uniform_distribution() {
        const auto uniform = stats::UniformDistribution::create(0.0, 1.0).value;
        profile_distribution("Uniform", uniform, [this](std::size_t count) {
            std::vector<double> values(count);
            std::uniform_real_distribution<double> dist(-0.5, 1.5);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void profile_gaussian_distribution() {
        const auto gaussian = stats::GaussianDistribution::create(0.0, 1.0).value;
        profile_distribution("Gaussian", gaussian, [](std::size_t count) {
            std::vector<double> values(count);
            const double denominator =
                static_cast<double>(std::max<std::size_t>(1, count > 0 ? count - 1 : 0));
            for (std::size_t i = 0; i < count; ++i) {
                values[i] = -4.0 + 8.0 * static_cast<double>(i) / denominator;
            }
            return values;
        });
    }

    void profile_exponential_distribution() {
        const auto exponential = stats::ExponentialDistribution::create(1.0).value;
        profile_distribution("Exponential", exponential, [this](std::size_t count) {
            std::vector<double> values(count);
            std::exponential_distribution<double> dist(1.0);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void profile_discrete_distribution() {
        const auto discrete = stats::DiscreteDistribution::create(0, 10).value;
        profile_distribution("Discrete", discrete, [this](std::size_t count) {
            std::vector<double> values(count);
            std::uniform_int_distribution<int> dist(0, 10);
            for (auto& value : values) {
                value = static_cast<double>(dist(gen_));
            }
            return values;
        });
    }

    void profile_poisson_distribution() {
        const auto poisson = stats::PoissonDistribution::create(3.5).value;
        profile_distribution("Poisson", poisson, [this](std::size_t count) {
            std::vector<double> values(count);
            std::poisson_distribution<int> dist(3);
            for (auto& value : values) {
                value = static_cast<double>(dist(gen_));
            }
            return values;
        });
    }

    void profile_gamma_distribution() {
        const auto gamma = stats::GammaDistribution::create(2.0, 1.0).value;
        profile_distribution("Gamma", gamma, [this](std::size_t count) {
            std::vector<double> values(count);
            std::gamma_distribution<double> dist(1.5, 2.0);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void profile_student_t_distribution() {
        const auto student_t = stats::StudentTDistribution::create(5.0).value;
        profile_distribution("StudentT", student_t, [this](std::size_t count) {
            std::vector<double> values(count);
            std::student_t_distribution<double> dist(5.0);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void profile_beta_distribution() {
        const auto beta = stats::BetaDistribution::create(2.0, 5.0).value;
        profile_distribution("Beta", beta, [this](std::size_t count) {
            std::vector<double> values(count);
            std::uniform_real_distribution<double> dist(-0.1, 1.1);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void profile_chi_squared_distribution() {
        const auto chi_squared = stats::ChiSquaredDistribution::create(4.0).value;
        profile_distribution("ChiSquared", chi_squared, [this](std::size_t count) {
            std::vector<double> values(count);
            std::chi_squared_distribution<double> dist(4.0);
            for (auto& value : values) {
                value = dist(gen_);
            }
            return values;
        });
    }

    void print_summary() const {
        stats::detail::detail::sectionHeader("Best Strategy Summary");

        using SummaryKey = std::tuple<std::string, std::string, std::size_t>;
        std::map<SummaryKey, std::vector<const StrategyProfileResult*>> grouped_results;
        for (const auto& result : results_) {
            grouped_results[{result.distribution, result.operation, result.batch_size}].push_back(
                &result);
        }

        stats::detail::detail::ColumnFormatter formatter({14, 10, 10, 16, 14});
        std::cout << formatter.formatRow(
                         {"Distribution", "Operation", "Size", "Best Strategy", "Time (μs)"})
                  << "\n";
        std::cout << formatter.getSeparator() << "\n";

        for (const auto& [key, result_group] : grouped_results) {
            const auto* best_result = *std::min_element(
                result_group.begin(), result_group.end(),
                [](const StrategyProfileResult* left, const StrategyProfileResult* right) {
                    return left->median_time_us < right->median_time_us;
                });

            std::cout << formatter.formatRow(
                             {std::get<0>(key), std::get<1>(key), std::to_string(std::get<2>(key)),
                              stats::detail::detail::strategyToDisplayString(best_result->strategy),
                              stats::detail::detail::formatDouble(best_result->median_time_us, 2)})
                      << "\n";
        }

        std::cout << "\n";
        print_crossover_summary(grouped_results);
    }

    void print_crossover_summary(
        const std::map<std::tuple<std::string, std::string, std::size_t>,
                       std::vector<const StrategyProfileResult*>>& grouped_results) const {
        stats::detail::detail::sectionHeader("Crossover Summary");

        using GroupKey = std::pair<std::string, std::string>;
        std::map<GroupKey, std::map<std::size_t, std::map<Strategy, double>>> timings_by_group;

        for (const auto& [key, result_group] : grouped_results) {
            const GroupKey group_key{std::get<0>(key), std::get<1>(key)};
            auto& size_timings = timings_by_group[group_key][std::get<2>(key)];
            for (const auto* result : result_group) {
                size_timings[result->strategy] = result->median_time_us;
            }
        }

        stats::detail::detail::ColumnFormatter formatter({14, 10, 16, 16, 18});
        std::cout << formatter.formatRow(
                         {"Distribution", "Operation", "S→V", "V→P", "P→Work-Steal"})
                  << "\n";
        std::cout << formatter.getSeparator() << "\n";

        for (const auto& [group_key, size_map] : timings_by_group) {
            const auto scalar_to_vectorized =
                find_first_crossover(size_map, Strategy::SCALAR, Strategy::VECTORIZED);
            const auto vectorized_to_parallel =
                find_first_crossover(size_map, Strategy::VECTORIZED, Strategy::PARALLEL);
            const auto parallel_to_work_stealing =
                find_first_crossover(size_map, Strategy::PARALLEL, Strategy::WORK_STEALING);

            std::cout << formatter.formatRow({group_key.first, group_key.second,
                                              crossover_to_string(scalar_to_vectorized),
                                              crossover_to_string(vectorized_to_parallel),
                                              crossover_to_string(parallel_to_work_stealing)})
                      << "\n";
        }

        std::cout << "\n";
    }

    static std::optional<std::size_t> find_first_crossover(
        const std::map<std::size_t, std::map<Strategy, double>>& size_map, Strategy slower_strategy,
        Strategy faster_strategy) {
        for (const auto& [batch_size, timings] : size_map) {
            const auto slower_it = timings.find(slower_strategy);
            const auto faster_it = timings.find(faster_strategy);
            if (slower_it == timings.end() || faster_it == timings.end()) {
                continue;
            }
            if (faster_it->second < slower_it->second) {
                return batch_size;
            }
        }
        return std::nullopt;
    }

    static std::string crossover_to_string(const std::optional<std::size_t>& crossover) {
        return crossover.has_value() ? std::to_string(*crossover) : "never";
    }

    void save_results(const std::string& output_csv_path) const {
        std::ofstream csv_file(output_csv_path);
        csv_file << "Distribution,Operation,BatchSize,Strategy,MedianTime_us\n";
        csv_file << std::fixed << std::setprecision(6);

        for (const auto& result : results_) {
            csv_file << result.distribution << "," << result.operation << "," << result.batch_size
                     << "," << stats::detail::detail::strategyToString(result.strategy) << ","
                     << result.median_time_us << "\n";
        }

        std::cout << "Results saved to " << output_csv_path << "\n";
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -l, --large              Include 1M and 2M batch sizes\n";
    std::cout << "  -o, --output-csv PATH    Write CSV results to PATH\n";
    std::cout << "  -h, --help               Show this help message\n";
    std::cout << "\nDefault output file: " << RESULTS_CSV_FILENAME << "\n";
}

int main(int argc, char* argv[]) {
    bool include_large = false;
    std::string output_csv_path = RESULTS_CSV_FILENAME;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-l" || arg == "--large") {
            include_large = true;
        } else if (arg == "-o" || arg == "--output-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                return 1;
            }
            output_csv_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    return stats::detail::detail::runTool("Strategy Profile", [include_large, &output_csv_path]() {
        StrategyProfiler profiler(include_large);
        profiler.run(output_csv_path);
    });
}
