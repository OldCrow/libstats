#include <iostream>
#include "../include/distributions/gaussian.h"

using namespace std;
using namespace libstats;

int main() {
    cout << "=== Dynamic Library Linking Test ===" << endl;
    
    // Test basic functionality to ensure dynamic linking works
    auto normal = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    cout << "Mean: " << normal.getMean() << endl;
    cout << "Variance: " << normal.getVariance() << endl;
    cout << "PDF at 0: " << normal.getProbability(0.0) << endl;
    cout << "CDF at 0: " << normal.getCumulativeProbability(0.0) << endl;
    
    cout << "✓ Dynamic library linking works!" << endl;
    
    return 0;
}
