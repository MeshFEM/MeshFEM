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

#include <iostream>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <array>
#include <sstream>

namespace MeshFEM {

template<class OrderingChoices>
struct AdaptiveOrderingSelection {
    using OrderingMethod = decltype(OrderingChoices::primary_method);
    bool verbose = false;

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
                       (numeric_facts_per_symbolic_fact_in_window.back() > num_factorizations_before_permitting_hard_switch);
        return trigger;
    }

    OrderingMethod updateSelection() {
        if (m_shouldSwitchMethod(method_switch_threshold_soft)) {
            std::cout << "Switching from ordering " << current_method << " to " << 1 - current_method
                      << "  " << factorizationTimingDescription() << std::endl;
            current_method = 1 - current_method;
            factor_nnz_at_last_switch = 0;
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
        if (verbose) std::cout << "Recorded method " << current_method << " numeric time: " << time << ";    " << factorizationTimingDescription() << std::endl;
    }

    void recordSolve(double time) {
        if (currentMethodSymbolicCounts() == 0)         throw std::logic_error("No symbolic factorization for this method yet!");
        if (solves_per_symbolic_fact_in_window.empty()) throw std::logic_error("Symbolic factorization info record missing.");

        factorization_times_for_method[current_method].solve.add(time);
        solves_per_symbolic_fact_in_window.back()++;
    }

    void recordSymbolic(double time, size_t factor_nnz = 0) {
        factorization_times_for_method[current_method].symbolic.add(time);

        numeric_facts_per_symbolic_fact_in_window.push_back(0);
        if (numeric_facts_per_symbolic_fact_in_window.size() > window_size)
            numeric_facts_per_symbolic_fact_in_window.pop_front();

        solves_per_symbolic_fact_in_window.push_back(0);
        if (solves_per_symbolic_fact_in_window.size() > window_size)
            solves_per_symbolic_fact_in_window.pop_front();

        if (verbose) std::cout << "Recorded method " << current_method << " symbolic time: " << time << "\t factor nnz: " << factor_nnz << ";    " << factorizationTimingDescription() << std::endl;

        if (factor_nnz_at_last_switch == 0) {
            if (verbose) std::cout << "First symbolic factorization since last switch. Setting factor_nnz_at_last_switch to " << factor_nnz << std::endl;
            factor_nnz_at_last_switch = factor_nnz;
        }
        if (factor_nnz < factor_nnz_at_last_switch * (1.0 - factor_nnz_reduction_threshold_for_reset)) {
            if (verbose) std::cout << "Significant factor nnz reduction since last switch (" << factor_nnz_at_last_switch << " -> " << factor_nnz << "). Resetting timing history for inactive method." << std::endl;
            factorization_times_for_method[1 - current_method].reset();
        }
    }

    size_t symbolicFactorizationsInWindow() const {
        assert(numeric_facts_per_symbolic_fact_in_window.size() == solves_per_symbolic_fact_in_window.size());
        return numeric_facts_per_symbolic_fact_in_window.size();
    }

    size_t numericFactorizationsInWindow() const {
        return std::accumulate(numeric_facts_per_symbolic_fact_in_window.begin(),
                               numeric_facts_per_symbolic_fact_in_window.end(), 0);
    }

    size_t solvesInWindow() const {
        return std::accumulate(solves_per_symbolic_fact_in_window.begin(),
                               solves_per_symbolic_fact_in_window.end(), 0);
    }

    // How long would it take for each method to do the factorizations recorded
    // over the history window?
    double timeEstimateForMethod(size_t method) const {
        if (numericFactorizationsInWindow() == 0) throw std::runtime_error("No numeric factorizations yet!");
        if (method > 1) throw std::runtime_error("Invalid method index");

        if ((factorization_times_for_method[current_method].symbolic.count == 0) ||
            (factorization_times_for_method[current_method]. numeric.count == 0) ||
            (factorization_times_for_method[current_method].   solve.count == 0)) {
            throw std::runtime_error("No times recorded for current method!");
        }

        // Get average timing statistics for `method`, potentially using an
        // estimate based on the currently active method's timings if
        // representative data is not yet available for it.
        // TODO: update to support more than two methods; requires change to multiplier representation in `OrderingChoices`.
        double avg_sym = factorization_times_for_method[method].symbolic.average(/* default = */ OrderingChoices::optimistic_sym_time_multiplier_estimates[method] * factorization_times_for_method[current_method].symbolic.average());
        double avg_num = factorization_times_for_method[method]. numeric.average(/* default = */ OrderingChoices::optimistic_num_time_multiplier_estimates[method] * factorization_times_for_method[current_method]. numeric.average());
        double avg_sol = factorization_times_for_method[method].   solve.average(/* default = */ OrderingChoices::optimistic_num_time_multiplier_estimates[method] * factorization_times_for_method[current_method].   solve.average());

        return avg_sym * symbolicFactorizationsInWindow()
             + avg_num *  numericFactorizationsInWindow()
             + avg_sol *                 solvesInWindow();
    }

    std::string factorizationTimingDescription() const {
        std::ostringstream oss;
        if (numericFactorizationsInWindow() > 0) {
            try {
                double est_0 = timeEstimateForMethod(0), est_1 = timeEstimateForMethod(1);
                oss << "time estimate method Nesdis: " << est_0
                          << "  time estimate method AMD: " << est_1
                          << "  "
                          ;
            }
            catch (const std::runtime_error &e) {
                oss << "time estimates unavailable: " << e.what() << "  ";
            }
        }
        oss <<    "numericFactorizationsInWindow: " << numericFactorizationsInWindow()
            << "  symbolicFactorizationsInWindow: " << symbolicFactorizationsInWindow()
            <<             "  Symbolic times (0): " << factorization_times_for_method[0].symbolic.average(-1) << "s"
            <<              "  Numeric times (0): " << factorization_times_for_method[0]. numeric.average(-1) << "s"
            <<              "    Solve times (0): " << factorization_times_for_method[0].   solve.average(-1) << "s"
            <<             "  Symbolic times (1): " << factorization_times_for_method[1].symbolic.average(-1) << "s"
            <<              "  Numeric times (1): " << factorization_times_for_method[1]. numeric.average(-1) << "s"
            <<              "    Solve times (1): " << factorization_times_for_method[1].   solve.average(-1) << "s";
        return oss.str();
    }

    size_t totalSymbolicFactorizations() const { return factorization_times_for_method[0].symbolic.count + factorization_times_for_method[1].symbolic.count; }
    size_t  totalNumericFactorizations() const { return factorization_times_for_method[0]. numeric.count + factorization_times_for_method[1]. numeric.count; }
    size_t                 totalSolves() const { return factorization_times_for_method[0].   solve.count + factorization_times_for_method[1].   solve.count; }

    // Heuristic parameters
    size_t window_size = 5;      // how many of the most recent symbolic factorizations (and their associated numeric factorizations) to consider
    size_t warmup_sym_count = 2; // number of symbolic factorizations to do with the primary method before activating heuristics.

    double method_switch_threshold_soft = 1.05; // Speedup factor above which to switch to the other strategy for the next symbolic refactorization
    double method_switch_threshold_hard = 1.10; // Speedup factor above which to force a switch to the other strategy.
    // Since a symbolic factorization should be avoided when possible,
    // we force ourselves to live with a suboptimal ordering for a while
    // before triggering a switch back to the primary method.
    size_t num_factorizations_before_permitting_hard_switch = 25;

    // State and historical data
    size_t current_method = 0; // 0: primary, 1: alternate
    std::array<OrderingMethod, 2> ordering_methods = {{ OrderingChoices::primary_method, OrderingChoices::alternate_method }};

    size_t factor_nnz_at_last_switch = 0; // This is used to detect when the symbolic factorization has changed significantly since the last switch, indicating that the inactive method's performance estimates are no longer reliable.
    double factor_nnz_reduction_threshold_for_reset = 0.15; // If the symbolic factorization's nnz has decreased by this fraction since the last switch, we reset the performance history for the inactive method.

    // Over the last `window_size` symbolic factorizations,
    // how many numeric factorizations were done using them?
    std::deque<size_t> numeric_facts_per_symbolic_fact_in_window, solves_per_symbolic_fact_in_window;

    struct FactorizationTime {
        double total = 0;
        size_t count = 0;
        double average_ema = 0;
        static constexpr double ema_alpha = 0.2; // Exponential moving average interpolation factor

        void add(double time) {
            if (count == 0)
                average_ema = time; // Initialize the EMA with the first time
            else
                average_ema = (1 - ema_alpha) * average_ema + ema_alpha * time;
            total += time;
            ++count;
        }
        bool known() const { return count > 0; }

        // This is now an exponential moving average rather than a simple unweighted average!
        double average(double default_val = 0) const { return count == 0 ? default_val : average_ema; }
    };

    struct FactorizationTimes {
        FactorizationTime symbolic;
        FactorizationTime numeric;
        FactorizationTime solve;

        void reset() {
            symbolic = FactorizationTime();
            numeric  = FactorizationTime();
            solve    = FactorizationTime();
        }
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

} // namespace MeshFEM

#endif /* end of include guard: ADAPTIVEORDERINGSELECTION_HH */
