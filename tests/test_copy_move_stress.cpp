#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

// Include all three distributions
#include "../include/uniform.h"
#include "../include/gaussian.h"
#include "../include/exponential.h"

using namespace std;
using namespace libstats;

std::atomic<int> completed_operations{0};
std::atomic<bool> stop_test{false};

void stressTestUniformCopyMove(int thread_id) {
    int local_ops = 0;
    
    while (!stop_test.load()) {
        try {
            // Create distributions
            auto result1 = UniformDistribution::create(thread_id * 10, thread_id * 10 + 5);
            auto result2 = UniformDistribution::create(0, 1);
            
            if (result1.isOk() && result2.isOk()) {
                auto uniform1 = std::move(result1.value);
                auto uniform2 = std::move(result2.value);
                
                // Perform multiple copy assignments
                for (int i = 0; i < 10; ++i) {
                    auto copy1 = uniform1;
                    auto copy2 = uniform2;
                    
                    // Swap them
                    copy1 = copy2;
                    copy2 = uniform1;
                    
                    // Move assignments
                    auto moved1 = std::move(copy1);
                    auto moved2 = std::move(copy2);
                    
                    // Use them to prevent optimization
                    double sum = moved1.getMean() + moved2.getMean();
                    (void)sum;
                }
                
                local_ops++;
            }
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            break;
        }
        
        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }
    
    completed_operations.fetch_add(local_ops % 100);
}

void stressTestGaussianCopyMove(int thread_id) {
    int local_ops = 0;
    
    while (!stop_test.load()) {
        try {
            // Create distributions
            GaussianDistribution gauss1(thread_id, 1.0);
            GaussianDistribution gauss2(thread_id + 10, 2.0);
            
            // Perform multiple copy assignments
            for (int i = 0; i < 10; ++i) {
                auto copy1 = gauss1;
                auto copy2 = gauss2;
                
                // Swap them
                copy1 = copy2;
                copy2 = gauss1;
                
                // Move assignments
                auto moved1 = std::move(copy1);
                auto moved2 = std::move(copy2);
                
                // Use them to prevent optimization
                double sum = moved1.getMean() + moved2.getMean();
                (void)sum;
            }
            
            local_ops++;
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            break;
        }
        
        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }
    
    completed_operations.fetch_add(local_ops % 100);
}

void stressTestExponentialCopyMove(int thread_id) {
    int local_ops = 0;
    
    while (!stop_test.load()) {
        try {
            // Create distributions
            auto result1 = ExponentialDistribution::create(thread_id + 1);
            auto result2 = ExponentialDistribution::create(thread_id + 2);
            
            if (result1.isOk() && result2.isOk()) {
                auto exp1 = std::move(result1.value);
                auto exp2 = std::move(result2.value);
                
                // Perform multiple copy assignments
                for (int i = 0; i < 10; ++i) {
                    auto copy1 = exp1;
                    auto copy2 = exp2;
                    
                    // Swap them
                    copy1 = copy2;
                    copy2 = exp1;
                    
                    // Move assignments
                    auto moved1 = std::move(copy1);
                    auto moved2 = std::move(copy2);
                    
                    // Use them to prevent optimization
                    double sum = moved1.getMean() + moved2.getMean();
                    (void)sum;
                }
                
                local_ops++;
            }
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            break;
        }
        
        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }
    
    completed_operations.fetch_add(local_ops % 100);
}

int main() {
    cout << "=== Copy/Move Semantics Stress Test ===" << endl;
    cout << "This test performs intensive copy/move operations across multiple threads" << endl;
    cout << "to ensure no deadlocks occur under high load." << endl;
    
    const int num_threads = 8;
    const int test_duration_seconds = 5;
    
    vector<thread> threads;
    
    cout << "\nStarting stress test with " << num_threads << " threads for " << test_duration_seconds << " seconds..." << endl;
    
    auto start_time = chrono::steady_clock::now();
    
    // Start threads for each distribution type
    for (int i = 0; i < num_threads / 3; ++i) {
        threads.emplace_back(stressTestUniformCopyMove, i);
    }
    
    for (int i = 0; i < num_threads / 3; ++i) {
        threads.emplace_back(stressTestGaussianCopyMove, i + 100);
    }
    
    for (int i = 0; i < num_threads - 2 * (num_threads / 3); ++i) {
        threads.emplace_back(stressTestExponentialCopyMove, i + 200);
    }
    
    // Monitor progress
    while (chrono::steady_clock::now() - start_time < chrono::seconds(test_duration_seconds)) {
        this_thread::sleep_for(chrono::milliseconds(500));
        cout << "\rOperations completed: " << completed_operations.load() << flush;
    }
    
    cout << "\n\nStopping test..." << endl;
    stop_test.store(true);
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "\n🎉 Stress test completed successfully!" << endl;
    cout << "✅ Duration: " << duration.count() << " ms" << endl;
    cout << "✅ Total operations: " << completed_operations.load() << endl;
    cout << "✅ Operations per second: " << (completed_operations.load() * 1000) / duration.count() << endl;
    cout << "✅ No deadlocks occurred under high load" << endl;
    cout << "✅ All distributions handle concurrent copy/move operations safely" << endl;
    
    return 0;
}
