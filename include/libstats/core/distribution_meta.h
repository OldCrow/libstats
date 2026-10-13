#pragma once

/**
 * @file core/distribution_meta.h
 * @brief Canonical per-distribution metadata table.
 *
 * This is the single source of truth for distribution registration.  Every
 * place that previously maintained its own list — ArchTable struct fields,
 * distributionTypeToString() switches, tool distribution lists — now derives
 * from kDistributionMeta.
 *
 * Adding a new distribution requires:
 *   1. Append the new DistributionType value to the enum in distribution_type.h
 *      (append-only; never reorder existing values).
 *   2. Append a corresponding DistributionMeta row to kDistributionMeta below.
 *   3. Append a ThresholdRow to each kXxx table in dispatch_thresholds.h.
 *   4. Implement the distribution header, source, and tests.
 *   5. Register in CMakeLists.txt and include/libstats.h.
 *
 * Ordering is enforced at compile time by a consteval loop that verifies every
 * row's type field equals its array index.  This catches swaps, insertions, and
 * deletions across the full table without per-row individual asserts.
 */

#include "distribution_type.h"

#include <array>
#include <cstddef>
#include <string_view>

namespace stats {
namespace detail {

/**
 * @brief Compile-time metadata for one distribution.
 *
 * Fields:
 *   type                — The DistributionType enum value.  Must equal the
 *                         array index (enforced by static_assert below).
 *   enum_name           — ALL_CAPS name used in performance-history keys and
 *                         profiling CSV output (e.g. "NEGATIVE_BINOMIAL").
 *   display_name        — PascalCase name for human-readable output and tool
 *                         tables (e.g. "NegativeBinomial").
 *   is_discrete         — true for PMF-based distributions (Poisson, Binomial,
 *                         NegBinomial, Discrete, Geometric, Bernoulli).
 */
struct DistributionMeta {
    DistributionType type;
    std::string_view enum_name;
    std::string_view display_name;
    bool is_discrete;
};

// ============================================================================
// Canonical metadata array — indexed by static_cast<std::size_t>(DistributionType)
//
// APPEND-ONLY.  Never reorder existing rows; the index must equal the enum
// value.  The static_assert below verifies this at compile time.
// ============================================================================
inline constexpr DistributionMeta kDistributionMeta[] = {
    // type                       enum_name              display_name        discrete
    {DistributionType::UNIFORM, "UNIFORM", "Uniform", false},
    {DistributionType::GAUSSIAN, "GAUSSIAN", "Gaussian", false},
    {DistributionType::EXPONENTIAL, "EXPONENTIAL", "Exponential", false},
    {DistributionType::DISCRETE, "DISCRETE", "Discrete", true},
    {DistributionType::POISSON, "POISSON", "Poisson", true},
    {DistributionType::GAMMA, "GAMMA", "Gamma", false},
    {DistributionType::STUDENT_T, "STUDENT_T", "StudentT", false},
    {DistributionType::BETA, "BETA", "Beta", false},
    {DistributionType::CHI_SQUARED, "CHI_SQUARED", "ChiSquared",
     false},  // delegates to Gamma(α=ν/2, β=1/2)
    {DistributionType::LOG_NORMAL, "LOG_NORMAL", "LogNormal", false},
    {DistributionType::PARETO, "PARETO", "Pareto", false},
    {DistributionType::WEIBULL, "WEIBULL", "Weibull", false},
    {DistributionType::RAYLEIGH, "RAYLEIGH", "Rayleigh", false},
    {DistributionType::VON_MISES, "VON_MISES", "VonMises", false},
    {DistributionType::BINOMIAL, "BINOMIAL", "Binomial", true},
    {DistributionType::NEGATIVE_BINOMIAL, "NEGATIVE_BINOMIAL", "NegativeBinomial", true},
    // v2.0.0 additions — implemented 2026-06-28 --------------------------------
    {DistributionType::GEOMETRIC, "GEOMETRIC", "Geometric",
     true},  // delegates to NegativeBinomial(r=1)
    {DistributionType::LAPLACE, "LAPLACE", "Laplace", false},
    {DistributionType::CAUCHY, "CAUCHY", "Cauchy",
     false},  // PDF/LogPDF only; CDF/Quantile closed-form (#48)
    // v2.4.0 additions — scaffolded 2026-09-03 (#54–#57) -----------------------
    {DistributionType::LOGISTIC, "LOGISTIC", "Logistic", false},
    {DistributionType::GUMBEL, "GUMBEL", "Gumbel", false},
    {DistributionType::BERNOULLI, "BERNOULLI", "Bernoulli", true},  // delegates to Binomial(n=1)
    {DistributionType::ERLANG, "ERLANG", "Erlang",
     false},  // delegates to Gamma(k, λ) — libstats Gamma β is a RATE
    {DistributionType::FISHER_F, "FISHER_F", "FisherF",
     false},  // delegates to Beta(d₁/2, d₂/2) via x → d₁x/(d₁x+d₂)
    {DistributionType::INVERSE_GAMMA, "INVERSE_GAMMA", "InverseGamma",
     false},  // delegates to Gamma via x → 1/x (upper-tail complement)
    {DistributionType::HALF_NORMAL, "HALF_NORMAL", "HalfNormal", false},
    {DistributionType::TRUNCATED_NORMAL, "TRUNCATED_NORMAL", "TruncatedNormal",
     false},  // renormalized internal Gaussian
};

/// Number of defined distribution types (= std::size(kDistributionMeta)).
inline constexpr std::size_t kDistributionTypeCount = std::size(kDistributionMeta);

// ============================================================================
// Compile-time ordering verification
//
// validateMetaOrdering() checks EVERY row: the type field must equal its
// array index.  A single loop catches any swap, insertion, or deletion without
// requiring per-row individual asserts.  Adding a new row is automatically
// checked without touching this section.
// ============================================================================
consteval bool validateMetaOrdering() noexcept {
    for (std::size_t i = 0; i < std::size(kDistributionMeta); ++i) {
        if (static_cast<std::size_t>(kDistributionMeta[i].type) != i)
            return false;
    }
    return true;
}
static_assert(validateMetaOrdering(),
              "kDistributionMeta row index does not match its DistributionType enum value. "
              "Rows must be in enum order (append-only; never reorder) because values are "
              "used as array indices and any reordering silently corrupts dispatch.");

// Belt-and-suspenders: catches accidental deletions or enum reorderings.
// All 27 distributions are implemented as of v2.4.0 (#54–#57). Bump to 28
// only when a genuinely new DistributionType enum value is appended.
static_assert(kDistributionTypeCount >= 27,
              "Distribution count regressed below 27 — a DistributionType entry was removed "
              "or the enum was reordered. Values are used as array indices; never remove or "
              "reorder entries.");

// ============================================================================
// Accessor functions
// ============================================================================

/**
 * @brief Look up metadata for a distribution type.
 * @note The index must be a valid DistributionType; behaviour is undefined for
 *       out-of-range values.  Use distributionMetaSafe() if the value is
 *       untrusted.
 */
[[nodiscard]] constexpr const DistributionMeta& distributionMeta(DistributionType dt) noexcept {
    return kDistributionMeta[static_cast<std::size_t>(dt)];
}

/**
 * @brief Look up metadata with bounds-checked fallback.
 * @return Pointer to metadata, or nullptr if dt is out of range.
 */
[[nodiscard]] constexpr const DistributionMeta* distributionMetaSafe(DistributionType dt) noexcept {
    auto idx = static_cast<std::size_t>(dt);
    return (idx < kDistributionTypeCount) ? &kDistributionMeta[idx] : nullptr;
}

/**
 * @brief Return the ALL_CAPS enum name for a distribution (used in keys/CSV).
 * @return e.g. "NEGATIVE_BINOMIAL", or "UNKNOWN" for out-of-range values.
 */
[[nodiscard]] constexpr std::string_view distributionEnumName(DistributionType dt) noexcept {
    const auto* m = distributionMetaSafe(dt);
    return m ? m->enum_name : "UNKNOWN";
}

/**
 * @brief Return the PascalCase display name for a distribution (used in tools/output).
 * @return e.g. "NegativeBinomial", or "Unknown" for out-of-range values.
 */
[[nodiscard]] constexpr std::string_view distributionDisplayName(DistributionType dt) noexcept {
    const auto* m = distributionMetaSafe(dt);
    return m ? m->display_name : "Unknown";
}

}  // namespace detail
}  // namespace stats
