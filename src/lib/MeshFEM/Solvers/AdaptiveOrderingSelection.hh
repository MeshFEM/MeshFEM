////////////////////////////////////////////////////////////////////////////////
// AdaptiveOrderingSelection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Heuristics for adaptively selecting between two sparse matrix orderings
//  based on the factorization performance history.
//
//  We start with a high-quality "primary" ordering method (e.g., nested
//  dissection) that is costly but generally yields significantly faster
//  numeric factorizations.
//  Then, if the problem requires recomputing the symbolic factorization
//  too frequently, we switch to the "alternate" ordering method (e.g., AMD)
//  that is fast to compute but lower quality.
//  We can subsequently switch back if the ratio of symbolic to numeric
//  factorizations decreases again.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/20/2025 10:03:35
*///////////////////////////////////////////////////////////////////////////////
#ifndef ADAPTIVEORDERINGSELECTION_HH
#define ADAPTIVEORDERINGSELECTION_HH

#include <deque>

template<class OrderingChoices>
struct AdaptiveOrderingSelection {
    using OrderingMethod = decltype(OrderingChoices::primary_method);

    bool shouldTriggerSymbolicFactorizationRecompute() const {
        const auto &nt_curr  = factorization_times_for_method[    current_method].numeric;
        const auto &nt_other = factorization_times_for_method[1 - current_method].numeric;
        if (!nt_other.known() || (nt_other.average() > nt_curr.average())) {
            // It is only beneficial to force a switch to a method known to
            // have a faster numeric factorization. Otherwise, we should reap
            // the fast numeric factorization benefits until the next symbolic
            // factorization is needed.
            return false;
        }

        bool trigger = m_shouldSwitchMethod(method_switch_threshold_hard) &&
                       (numericFactorizationsInWindow() > num_factorizations_before_permitting_hard_switch);
        return trigger;
    }

    OrderingMethod updateSelection() {
        if (m_shouldSwitchMethod(method_switch_threshold_soft)) {
            std::cout << "Switching from ordering " << current_method << " to " << 1 - current_method
                      << "  " << factorizationTimingDescription() << std::endl;
            current_method = 1 - current_method;
        }
        return currentOrderingMethod();
    }

    OrderingMethod currentOrderingMethod() const { return ordering_methods[current_method]; }

    size_t currentMethodSymbolicCounts() const { return factorization_times_for_method[current_method].symbolic.count; }
    size_t  currentMethodNumericCounts() const { return factorization_times_for_method[current_method].numeric .count; }

    void recordNumeric(double time) {
        if (currentMethodSymbolicCounts() == 0)                throw std::logic_error("No symbolic factorization for this method yet!");
        if (numeric_facts_per_symbolic_fact_in_window.empty()) throw std::logic_error("Symbolic factorization info record missing.");

        factorization_times_for_method[current_method].numeric.add(time);
        numeric_facts_per_symbolic_fact_in_window.back()++;
    }

    void recordSymbolic(double time) {
        factorization_times_for_method[current_method].symbolic.add(time);

        numeric_facts_per_symbolic_fact_in_window.push_back(0);
        if (numeric_facts_per_symbolic_fact_in_window.size() > window_size)
            numeric_facts_per_symbolic_fact_in_window.pop_front();
    }

    size_t symbolicFactorizationsInWindow() const {
        return numeric_facts_per_symbolic_fact_in_window.size();
    }

    size_t numericFactorizationsInWindow() const {
        return std::accumulate(numeric_facts_per_symbolic_fact_in_window.begin(),
                               numeric_facts_per_symbolic_fact_in_window.end(), 0);
    }

    // How long would it take for each method to do the factorizations recorded
    // over the history window?
    double timeEstimateForMethod(size_t method) const {
        if (numericFactorizationsInWindow() == 0) throw std::runtime_error("No numeric factorizations yet!");
        if (method > 1) throw std::runtime_error("Invalid method index");
        double avg_sym = factorization_times_for_method[0].symbolic.average(); // primary
        double avg_num = factorization_times_for_method[0]. numeric.average(); // primary

        if (method == 1) {
            // The alternate method may not have timings recorded yet.
            // In this case, we apply the speedup/slowdown multiplier estimates
            // to the primary method timings.
            avg_sym = factorization_times_for_method[1].symbolic.average(/* default = */ OrderingChoices::alternate_method_sym_time_multiplier_estimate * avg_sym); // alternate
            avg_num = factorization_times_for_method[1]. numeric.average(/* default = */ OrderingChoices::alternate_method_num_time_multiplier_estimate * avg_num); // alternate
        }

        return avg_sym * symbolicFactorizationsInWindow()
             + avg_num *  numericFactorizationsInWindow();
    }

    std::string factorizationTimingDescription() const {
        std::ostringstream oss;
        if (numericFactorizationsInWindow() > 0) {
            std::cout << "time estimate method Nesdis: " << timeEstimateForMethod(0)
                      << "  time estimate method AMD: " << timeEstimateForMethod(1)
                      << "  "
                      ;
        }
        oss <<    "numericFactorizationsInWindow: " << numericFactorizationsInWindow()
            << "  symbolicFactorizationsInWindow: " << symbolicFactorizationsInWindow()
            <<             "  Symbolic times (0): " << factorization_times_for_method[0].symbolic.average(-1) << "s"
            <<              "  Numeric times (0): " << factorization_times_for_method[0]. numeric.average(-1) << "s"
            <<             "  Symbolic times (1): " << factorization_times_for_method[1].symbolic.average(-1) << "s"
            <<              "  Numeric times (1): " << factorization_times_for_method[1]. numeric.average(-1) << "s";
        return oss.str();
    }

    size_t totalSymbolicFactorizations() const { return factorization_times_for_method[0].symbolic.count + factorization_times_for_method[1].symbolic.count; }
    size_t totalNumericFactorizations()  const { return factorization_times_for_method[0]. numeric.count + factorization_times_for_method[1]. numeric.count; }

    // Heuristic parameters
    size_t window_size = 5;      // how many of the most recent symbolic factorizations (and their associated numeric factorizations) to consider
    size_t warmup_sym_count = 2; // number of symbolic factorizations to do with the primary method before activating heuristics.

    double method_switch_threshold_soft = 1.05; // Speedup factor above which to switch to the other strategy for the next symbolic refactorization
    double method_switch_threshold_hard = 1.10; // Speedup factor above which to switch to the other strategy for the next symbolic refactorization
    // Since a symbolic factorization should be avoided when possible,
    // we force ourselves to live with a suboptimal ordering for a while
    // before triggering a switch back to the primary method.
    size_t num_factorizations_before_permitting_hard_switch = 25;

    // State and historical data
    size_t current_method = 0; // 0: primary, 1: alternate
    std::array<OrderingMethod, 2> ordering_methods = { OrderingChoices::primary_method, OrderingChoices::alternate_method };

    // Over the last `window_size` symbolic factorizations,
    // how many numeric factorizations were done using them?
    std::deque<size_t> numeric_facts_per_symbolic_fact_in_window;

    struct FactorizationTime {
        double total = 0;
        size_t count = 0;

        void add(double time) {
            total += time;
            ++count;
        }
        bool known() const { return count > 0; }

        double average(double default_val = 0) const { return count == 0 ? default_val : total / count; }
    };

    struct FactorizationTimes {
        FactorizationTime symbolic;
        FactorizationTime numeric;
    };

    // 0: primary method, 1: alternate method
    std::array<FactorizationTimes, 2> factorization_times_for_method;

private:
    bool m_shouldSwitchMethod(double method_switch_threshold) const {
        if (totalSymbolicFactorizations() < warmup_sym_count) {
            if (current_method != 0)
                std::cout << "WARNING: alternate method used during warmup period--this shouldn't happen!\n";
            return false; // Stay on primary method
        }

        double time_for_current_method  = timeEstimateForMethod(current_method);
        double time_for_inactive_method = timeEstimateForMethod(1 - current_method);
        double ratio = time_for_current_method / time_for_inactive_method;

        return ratio > method_switch_threshold;
    }
};

#endif /* end of include guard: ADAPTIVEORDERINGSELECTION_HH */

