#pragma once

/**
 * @file common/libstats_string_common.h
 * @brief Consolidated string header - Phase 2 STL optimization
 *
 * This header consolidates string usage across the library, reducing redundant
 * includes of <string> which is used in 20% of headers (10 headers).
 *
 * Benefits:
 *   - Reduces string template instantiation overhead
 *   - Provides common string utilities for libstats
 *   - Centralized string formatting and parsing
 *   - Optimized string operations for statistical contexts
 */

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace stats {
namespace common {

/// Common string type aliases
using String = std::string;
using StringView = std::string_view;
using StringStream = std::ostringstream;

/// String utilities for statistical library
namespace string_utils {

/// Format floating-point numbers with statistical precision
inline std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

/// Format scientific notation for very large/small numbers
inline std::string format_scientific(double value, int precision = 3) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

/// Format statistical results with appropriate precision
inline std::string format_stat_result(double value) {
    if (std::abs(value) < 1e-6 || std::abs(value) > 1e6) {
        return format_scientific(value, 4);
    } else {
        return format_double(value, 6);
    }
}

/// Format confidence intervals
inline std::string format_confidence_interval(double lower, double upper,
                                              double confidence_level = 0.95) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << (confidence_level * 100) << "% CI: [" << lower << ", " << upper << "]";
    return oss.str();
}

/// Format distribution parameters
inline std::string format_distribution_params(
    const std::string& dist_name, const std::vector<std::pair<std::string, double>>& params) {
    std::ostringstream oss;
    oss << dist_name << "(";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << params[i].first << "=" << format_double(params[i].second, 4);
    }
    oss << ")";
    return oss.str();
}

/// Create error message with context
inline std::string create_error_message(const std::string& function_name,
                                        const std::string& error_description,
                                        const std::string& context = "") {
    std::ostringstream oss;
    oss << "stats::" << function_name << ": " << error_description;
    if (!context.empty()) {
        oss << " (context: " << context << ")";
    }
    return oss.str();
}

/// Format vector of doubles for debugging/output
template <typename Container>
inline std::string format_numeric_container(const Container& container,
                                            const std::string& name = "",
                                            std::size_t max_elements = 10) {
    std::ostringstream oss;
    if (!name.empty()) {
        oss << name << ": ";
    }
    oss << "[";

    std::size_t count = 0;
    for (const auto& value : container) {
        if (count > 0)
            oss << ", ";
        if (count >= max_elements) {
            oss << "... (" << (container.size() - max_elements) << " more)";
            break;
        }
        oss << format_double(static_cast<double>(value), 3);
        ++count;
    }
    oss << "]";
    return oss.str();
}

/// Parse string to double with error handling
inline bool parse_double(const std::string& str, double& result) {
    try {
        std::size_t pos = 0;
        result = std::stod(str, &pos);
        return pos == str.length();  // Ensure entire string was consumed
    } catch (...) {
        return false;
    }
}

/// Parse comma-separated values into vector
inline std::vector<double> parse_csv_doubles(const std::string& csv_str) {
    std::vector<double> result;
    std::istringstream iss(csv_str);
    std::string token;

    while (std::getline(iss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        double value;
        if (parse_double(token, value)) {
            result.push_back(value);
        }
    }
    return result;
}

/// Create human-readable duration string
inline std::string format_duration_ms(double milliseconds) {
    if (milliseconds < 1.0) {
        return format_double(milliseconds * 1000.0, 2) + " μs";
    } else if (milliseconds < 1000.0) {
        return format_double(milliseconds, 2) + " ms";
    } else if (milliseconds < 60000.0) {
        return format_double(milliseconds / 1000.0, 2) + " s";
    } else {
        double minutes = milliseconds / 60000.0;
        return format_double(minutes, 2) + " min";
    }
}

/// Create human-readable memory size string
inline std::string format_memory_size(std::size_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        return format_double(bytes / 1024.0, 1) + " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        return format_double(bytes / (1024.0 * 1024.0), 1) + " MB";
    } else {
        return format_double(bytes / (1024.0 * 1024.0 * 1024.0), 2) + " GB";
    }
}

/// Simple string hash for caching/indexing (not cryptographic)
inline std::size_t simple_hash(const std::string& str) {
    std::size_t hash = 0;
    for (char c : str) {
        hash = hash * 31 + static_cast<std::size_t>(c);
    }
    return hash;
}

/// Check if string contains only numeric characters (including decimal point)
inline bool is_numeric_string(const std::string& str) {
    if (str.empty())
        return false;

    bool has_decimal = false;
    std::size_t start = 0;

    // Handle optional leading sign
    if (str[0] == '+' || str[0] == '-') {
        start = 1;
        if (str.length() == 1)
            return false;  // Just a sign
    }

    for (std::size_t i = start; i < str.length(); ++i) {
        char c = str[i];
        if (c == '.') {
            if (has_decimal)
                return false;  // Multiple decimal points
            has_decimal = true;
        } else if (c < '0' || c > '9') {
            return false;  // Non-numeric character
        }
    }
    return true;
}
}  // namespace string_utils

}  // namespace common
}  // namespace stats
