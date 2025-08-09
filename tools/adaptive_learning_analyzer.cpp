#include "../include/libstats.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/core/performance_history.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <map>

using namespace libstats;
using namespace libstats::performance;

// Helper function to format duration
std::string formatDuration(std::chrono::nanoseconds ns) {
    if (ns.count() < 1000) return std::to_string(ns.count()) + "ns";
    if (ns.count() < 1000000) return std::to_string(ns.count() / 1000) + "μs";
    if (ns.count() < 1000000000) return std::to_string(ns.count() / 1000000) + "ms";
    return std::to_string(ns.count() / 1000000000) + "s";
}

// Test function that exercises different strategies across various batch sizes
template<typename Distribution>
void exerciseDistribution(const std::string& dist_name, DistributionType dist_type, 
                         Distribution& dist, const std::vector<size_t>& batch_sizes) {
    std::cout << "\n=== Testing " << dist_name << " Distribution ===\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\nBatch size: " << batch_size << std::endl;
        
        // Create test data
        std::vector<double> values(batch_size);
        std::uniform_real_distribution<double> value_gen(0.1, 10.0);
        for (auto& v : values) {
            v = value_gen(gen);
        }
        
        // Test PDF operations (medium complexity)
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> results(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                results[i] = dist.getProbability(values[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            // Record performance for SCALAR strategy (simulated)
            PerformanceDispatcher::recordPerformance(
                Strategy::SCALAR, dist_type, batch_size, duration.count()
            );
            
            std::cout << "  PDF (scalar): " << formatDuration(duration) 
                     << " (" << (duration.count() / batch_size) << "ns/op)" << std::endl;
        }
        
        // Test CDF operations (higher complexity)
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> results(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                results[i] = dist.getCumulativeProbability(values[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            // For larger batches, simulate SIMD performance (typically 2-4x faster)
            if (batch_size >= 32) {
                auto simd_duration = duration / 3;  // Simulate 3x SIMD speedup
                PerformanceDispatcher::recordPerformance(
                    Strategy::SIMD_BATCH, dist_type, batch_size, simd_duration.count()
                );
                std::cout << "  CDF (simd):   " << formatDuration(simd_duration) 
                         << " (" << (simd_duration.count() / batch_size) << "ns/op)" << std::endl;
            }
            
            // For very large batches, simulate parallel performance
            if (batch_size >= 1000) {
                auto parallel_duration = duration / 6;  // Simulate 6x parallel speedup
                PerformanceDispatcher::recordPerformance(
                    Strategy::PARALLEL_SIMD, dist_type, batch_size, parallel_duration.count()
                );
                std::cout << "  CDF (parallel): " << formatDuration(parallel_duration) 
                         << " (" << (parallel_duration.count() / batch_size) << "ns/op)" << std::endl;
            }
            
            PerformanceDispatcher::recordPerformance(
                Strategy::SCALAR, dist_type, batch_size, duration.count()
            );
            std::cout << "  CDF (scalar): " << formatDuration(duration) 
                     << " (" << (duration.count() / batch_size) << "ns/op)" << std::endl;
        }
        
        // For very large batches, test work stealing and cache-aware strategies
        if (batch_size >= 10000) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> results(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                results[i] = dist.getProbability(values[i]) + dist.getCumulativeProbability(values[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto base_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            // Simulate work-stealing (good for irregular workloads)
            auto work_stealing_duration = base_duration / 8;  // 8x speedup for large irregular work
            PerformanceDispatcher::recordPerformance(
                Strategy::WORK_STEALING, dist_type, batch_size, work_stealing_duration.count()
            );
            std::cout << "  Mixed (work-stealing): " << formatDuration(work_stealing_duration) 
                     << " (" << (work_stealing_duration.count() / batch_size) << "ns/op)" << std::endl;
            
            if (batch_size >= 50000) {
                // Simulate cache-aware (excellent for very large datasets)
                auto cache_aware_duration = base_duration / 12;  // 12x speedup for cache optimization
                PerformanceDispatcher::recordPerformance(
                    Strategy::CACHE_AWARE, dist_type, batch_size, cache_aware_duration.count()
                );
                std::cout << "  Mixed (cache-aware): " << formatDuration(cache_aware_duration) 
                         << " (" << (cache_aware_duration.count() / batch_size) << "ns/op)" << std::endl;
            }
        }
    }
}

void analyzePerformanceHistory() {
    auto& history = PerformanceDispatcher::getPerformanceHistory();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ADAPTIVE LEARNING ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nTotal executions recorded: " << history.getTotalExecutions() << std::endl;
    
    // Test strategy recommendations for different scenarios
    std::vector<DistributionType> distributions = {
        DistributionType::UNIFORM, DistributionType::GAUSSIAN, 
        DistributionType::EXPONENTIAL, DistributionType::DISCRETE,
        DistributionType::POISSON, DistributionType::GAMMA
    };
    
    std::vector<size_t> test_sizes = {10, 100, 1000, 5000, 25000, 100000};
    
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "STRATEGY RECOMMENDATIONS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (auto dist_type : distributions) {
        std::string dist_name;
        switch(dist_type) {
            case DistributionType::UNIFORM: dist_name = "Uniform"; break;
            case DistributionType::GAUSSIAN: dist_name = "Gaussian"; break;
            case DistributionType::EXPONENTIAL: dist_name = "Exponential"; break;
            case DistributionType::DISCRETE: dist_name = "Discrete"; break;
            case DistributionType::POISSON: dist_name = "Poisson"; break;
            case DistributionType::GAMMA: dist_name = "Gamma"; break;
        }
        
        std::cout << "\n" << dist_name << " Distribution:" << std::endl;
        std::cout << "  Size      Strategy        Confidence  Expected Time" << std::endl;
        std::cout << "  --------  --------------  ----------  -------------" << std::endl;
        
        for (size_t size : test_sizes) {
            auto recommendation = history.getBestStrategy(dist_type, size);
            
            std::string strategy_name;
            switch(recommendation.recommended_strategy) {
                case Strategy::SCALAR: strategy_name = "SCALAR"; break;
                case Strategy::SIMD_BATCH: strategy_name = "SIMD_BATCH"; break;
                case Strategy::PARALLEL_SIMD: strategy_name = "PARALLEL_SIMD"; break;
                case Strategy::WORK_STEALING: strategy_name = "WORK_STEALING"; break;
                case Strategy::CACHE_AWARE: strategy_name = "CACHE_AWARE"; break;
            }
            
            std::cout << "  " << std::setw(8) << size
                     << "  " << std::setw(14) << strategy_name
                     << "  " << std::setw(10) << std::fixed << std::setprecision(3) 
                     << recommendation.confidence_score
                     << "  " << std::setw(8) << (recommendation.expected_time_ns / 1000) << " μs"
                     << (recommendation.has_sufficient_data ? "" : " (insufficient data)")
                     << std::endl;
        }
    }
    
    // Test threshold learning
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "LEARNED OPTIMAL THRESHOLDS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (auto dist_type : distributions) {
        std::string dist_name;
        switch(dist_type) {
            case DistributionType::UNIFORM: dist_name = "Uniform"; break;
            case DistributionType::GAUSSIAN: dist_name = "Gaussian"; break;
            case DistributionType::EXPONENTIAL: dist_name = "Exponential"; break;
            case DistributionType::DISCRETE: dist_name = "Discrete"; break;
            case DistributionType::POISSON: dist_name = "Poisson"; break;
            case DistributionType::GAMMA: dist_name = "Gamma"; break;
        }
        
        auto thresholds = history.learnOptimalThresholds(dist_type);
        std::cout << dist_name << ": ";
        if (thresholds) {
            std::cout << "SIMD >= " << thresholds->first 
                     << ", Parallel >= " << thresholds->second << std::endl;
        } else {
            std::cout << "Insufficient data for learning" << std::endl;
        }
    }
    
    // Show performance statistics for each strategy
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "STRATEGY PERFORMANCE STATISTICS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (auto dist_type : distributions) {
        std::string dist_name;
        switch(dist_type) {
            case DistributionType::UNIFORM: dist_name = "Uniform"; break;
            case DistributionType::GAUSSIAN: dist_name = "Gaussian"; break;
            case DistributionType::EXPONENTIAL: dist_name = "Exponential"; break;
            case DistributionType::DISCRETE: dist_name = "Discrete"; break;
            case DistributionType::POISSON: dist_name = "Poisson"; break;
            case DistributionType::GAMMA: dist_name = "Gamma"; break;
        }
        
        std::cout << "\n" << dist_name << " Performance:" << std::endl;
        
        std::vector<Strategy> strategies = {
            Strategy::SCALAR, Strategy::SIMD_BATCH, Strategy::PARALLEL_SIMD,
            Strategy::WORK_STEALING, Strategy::CACHE_AWARE
        };
        
        for (auto strategy : strategies) {
            auto stats = history.getPerformanceStats(strategy, dist_type);
            if (stats) {
                std::string strategy_name;
                switch(strategy) {
                    case Strategy::SCALAR: strategy_name = "SCALAR"; break;
                    case Strategy::SIMD_BATCH: strategy_name = "SIMD_BATCH"; break;
                    case Strategy::PARALLEL_SIMD: strategy_name = "PARALLEL_SIMD"; break;
                    case Strategy::WORK_STEALING: strategy_name = "WORK_STEALING"; break;
                    case Strategy::CACHE_AWARE: strategy_name = "CACHE_AWARE"; break;
                }
                
                std::cout << "  " << std::setw(14) << strategy_name << ": "
                         << std::setw(6) << stats->execution_count << " runs, "
                         << "avg: " << std::setw(8) << (stats->getAverageTimeNs() / 1000) << " μs, "
                         << "min: " << std::setw(6) << (stats->min_time_ns / 1000) << " μs, "
                         << "max: " << std::setw(6) << (stats->max_time_ns / 1000) << " μs"
                         << std::endl;
            }
        }
    }
}

int main() {
    std::cout << "LIBSTATS ADAPTIVE LEARNING ANALYZER" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "\nThis tool exercises the adaptive learning system by running" << std::endl;
    std::cout << "various distribution operations across different batch sizes" << std::endl;
    std::cout << "and strategies, then analyzes the collected performance data.\n" << std::endl;
    
    // Test batch sizes from very small to very large
    std::vector<size_t> batch_sizes = {
        5, 16, 64, 128, 512, 1024, 2048, 8192, 16384, 32768, 65536, 131072
    };
    
    try {
        // Exercise different distributions
        {
            UniformDistribution uniform_dist(0.0, 10.0);
            exerciseDistribution("Uniform", DistributionType::UNIFORM, uniform_dist, batch_sizes);
        }
        
        {
            GaussianDistribution gaussian_dist(0.0, 1.0);
            exerciseDistribution("Gaussian", DistributionType::GAUSSIAN, gaussian_dist, batch_sizes);
        }
        
        {
            ExponentialDistribution exp_dist(1.0);
            exerciseDistribution("Exponential", DistributionType::EXPONENTIAL, exp_dist, batch_sizes);
        }
        
        {
            DiscreteDistribution disc_dist(1, 100);
            exerciseDistribution("Discrete", DistributionType::DISCRETE, disc_dist, batch_sizes);
        }
        
        {
            PoissonDistribution poisson_dist(5.0);
            exerciseDistribution("Poisson", DistributionType::POISSON, poisson_dist, batch_sizes);
        }
        
        {
            GammaDistribution gamma_dist(2.0, 1.0);
            exerciseDistribution("Gamma", DistributionType::GAMMA, gamma_dist, batch_sizes);
        }
        
        // Analyze the collected performance data
        analyzePerformanceHistory();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nAdaptive learning analysis completed successfully!" << std::endl;
    return 0;
}
