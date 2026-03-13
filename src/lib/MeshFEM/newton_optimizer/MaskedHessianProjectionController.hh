////////////////////////////////////////////////////////////////////////////////
// MaskedHessianProjectionController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements Hessian projection controllers that selectively enable projection
//  of element Hessians within a `MeshEnergy`,
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  02/20/2026 13:53:26
*///////////////////////////////////////////////////////////////////////////////
#ifndef MASKEDHESSIANPROJECTIONCONTROLLER_HH
#define MASKEDHESSIANPROJECTIONCONTROLLER_HH

#include "HessianProjectionController.hh"
#include <MeshFEM/MeshEnergyBase.hh>
#include <algorithm>

// Configures the per-element Hessian projection flags based on the norm of
// the fully assembled gradient restricted to each element.
// We project only if the norm exceeds `threshold` (eliminating damping in regions that
// have essentially converged to equilibrium).
struct MaskedHessianProjectionControllerGradNorm : public HessianProjectionController {
    MaskedHessianProjectionControllerGradNorm(MeshEnergyBase &me) : m_me(me) { }

    virtual void prepareForInitialFactorizationAttempt() override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MaskedHessianProjectionControllerGradNorm.prepareForInitialFactorizationAttempt");
        m_g = m_me.gradient();
        m_elementGradientNorms = m_me.elementGradientNorms(m_g);

        if (percentileControl)  {
            m_sortOrder.resize(m_me.numElements());
            std::iota(m_sortOrder.begin(), m_sortOrder.end(), 0);
            // TODO: this is a bottleneck; switch to fast selection-based algorithm.
            std::sort(m_sortOrder.begin(), m_sortOrder.end(), [&](size_t i, size_t j) { return m_elementGradientNorms[i] < m_elementGradientNorms[j]; });
        }
        else { m_sortOrder.resize(0); }

        m_referenceNorm = std::sqrt(m_g.squaredNorm() / m_me.numElements());

        m_applyThreshold();
        m_failures = 0;
    }

    virtual bool shouldUseProjection() const override { return true; }

    virtual bool notifyDefiniteness(bool isIndefinite) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MaskedHessianProjectionControllerGradNorm.notifyDefiniteness");
        size_t npe = m_me.numProjectedElements();

        if (verbose)
            std::cout << "notifyDefiniteness(" << isIndefinite << "): " << (100. * npe) / m_me.numElements() << "% projected, relativeThreshold = " << relativeThreshold << ", currentThreshold = " << currentThreshold();

        if (isIndefinite) {
            // Safeguard against numerical positive semidefiniteness triggering infinite failures.
            bool giveUpAndShift = (npe == m_me.numElements())
                                || (++m_failures == maxFailuresBeforeShifting);
            if (giveUpAndShift) return false; // no Hessian reevaluation needed when switching to shifts.

            if (percentileControl) relativeThreshold *= relativeThreshold;
            else relativeThreshold *= 0.5;

            m_applyThreshold();
        }
        else {
            // Positive definite case: try to reduce projection.
            m_lastPDRelThreshold = relativeThreshold;
            m_lastPDThreshold = currentThreshold();
            if (npe > 0) {
                if (percentileControl) relativeThreshold = sqrt(relativeThreshold); // geometric mean between current threshold and 1.0
                else relativeThreshold *= 2.0;
            }
        }

        if (verbose)
            std::cout << " -> new relativeThreshold = " << relativeThreshold << ", currentThreshold = " << currentThreshold() << std::endl;

        return isIndefinite;
    }

    virtual void reset() override { relativeThreshold = 0.5; }

    virtual std::unique_ptr<HessianProjectionController> clone() const override { return std::make_unique<MaskedHessianProjectionControllerGradNorm>(*this); }

    double relativeThreshold = 0.5;
    int maxFailuresBeforeShifting = 10;
    bool percentileControl = true; // If true, adjust the threshold to target a certain percentile of projected elements instead of halving/doubling.

    bool verbose = false;

    double currentThreshold() const {
        double threshold;
        if (percentileControl) {
            if (!m_sortOrder.size()) {
                std::cerr << "Warning: MaskedHessianProjectionControllerGradNorm: percentileControl enabled but sortOrder is empty.\n";
                const_cast<MaskedHessianProjectionControllerGradNorm *>(this)->prepareForInitialFactorizationAttempt();
            }

            size_t targetNPE = size_t(relativeThreshold * m_me.numElements());
            double snap_tol = 1e-5;
            if (relativeThreshold < snap_tol)          threshold = m_elementGradientNorms[m_sortOrder[0]] / 2;
            else if (relativeThreshold > 1 - snap_tol) threshold = m_elementGradientNorms[m_sortOrder[m_me.numElements() - 1]] * 2;
            else                                       threshold = m_elementGradientNorms[m_sortOrder[std::min(targetNPE, m_me.numElements() - 1)]];
        }
        else {
            threshold = relativeThreshold * m_referenceNorm;
        }
        return threshold;
    }

    double getLastPDThreshold() const { return m_lastPDThreshold; }
    double getLastPDRelThreshold() const { return m_lastPDRelThreshold; }

private:
    void m_applyThreshold() {
        m_me.elementHessianProjectionMasks = m_elementGradientNorms.array() >= currentThreshold();
    }

    double m_lastPDThreshold = std::numeric_limits<double>::infinity();
    double m_lastPDRelThreshold = std::numeric_limits<double>::infinity();

    double m_referenceNorm;
    Eigen::VectorXd m_g, m_elementGradientNorms;
    std::vector<int> m_sortOrder;
    MeshEnergyBase &m_me;
    int m_failures = 0;
};

// Configure the per-element Hessian projection flags based on the smallest
// eigenvalue: we project only elements with seriously negative eigenvalues.
// The clamp target can also optionally be configured to coincide with the
// projection threshold (in which case setting the projection flags is simply
// a performance optmization).
struct MaskedHessianProjectionControllerMinEigenvalue : public HessianProjectionController {
    MaskedHessianProjectionControllerMinEigenvalue(MeshEnergyBase &me) : m_me(me) { }

    virtual void prepareForInitialFactorizationAttempt() override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MaskedHessianProjectionControllerGradNorm.prepareForInitialFactorizationAttempt");
        m_minimumEigenvalues = m_me.elementHessianMinimumEigenvalues();
        // m_referenceEigenvalue = m_minimumEigenvalues.minCoeff();
        m_referenceEigenvalue = -1;

        if (percentileControl)  {
            m_sortOrder.resize(m_me.numElements());
            std::iota(m_sortOrder.begin(), m_sortOrder.end(), 0);
            // TODO: this is a bottleneck; switch to fast selection-based algorithm.
            std::sort(m_sortOrder.begin(), m_sortOrder.end(), [&](size_t i, size_t j) { return m_minimumEigenvalues[i] < m_minimumEigenvalues[j]; });
        }
        else { m_sortOrder.resize(0); }

        m_applyThreshold();
        m_failures = 0;
    }

    virtual bool shouldUseProjection() const override { return true; }

    virtual bool notifyDefiniteness(bool isIndefinite) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MaskedHessianProjectionControllerGradNorm.notifyDefiniteness");
        size_t npe = m_me.numProjectedElements();

        if (verbose)
            std::cout << "notifyDefiniteness(" << isIndefinite << "): " << (100. * npe) / m_me.numElements() << "% projected, relativeThreshold = " << relativeThreshold << std::endl;

        if (isIndefinite) {
            // Safeguard against numerical positive semidefiniteness triggering infinite failures.
            bool giveUpAndShift = (npe == m_me.numElements())
                                || (++m_failures == maxFailuresBeforeShifting);
            if (giveUpAndShift) return false; // no Hessian reevaluation needed when switching to shifts.

            if (percentileControl) relativeThreshold = sqrt(relativeThreshold); // geometric mean between current threshold and 1.0
            else relativeThreshold *= 0.5;

            m_applyThreshold();

            return true;
        }

        // Positive definite case: try to reduce projection.
        if (npe > 0) {
            if (percentileControl) relativeThreshold *= relativeThreshold;
            else relativeThreshold *= 2.0;
        }
        return false;
    }

    virtual void reset() override { relativeThreshold = defaultRelativeThreshold; }

    virtual std::unique_ptr<HessianProjectionController> clone() const override { return std::make_unique<MaskedHessianProjectionControllerMinEigenvalue>(*this); }

    double currentThreshold() const {
        double threshold;
        if (percentileControl) {
            if (!m_sortOrder.size()) {
                std::cerr << "Warning: MaskedHessianProjectionControllerMinEigenvalue: percentileControl enabled but sortOrder is empty.\n";
                const_cast<MaskedHessianProjectionControllerMinEigenvalue *>(this)->prepareForInitialFactorizationAttempt();
            }

            size_t targetNPE = size_t(relativeThreshold * m_me.numElements());
            double snap_tol = 1e-5;
            if (relativeThreshold < snap_tol) targetNPE = 0;
            else if (relativeThreshold > 1 - snap_tol) targetNPE = m_me.numElements();
            threshold = m_minimumEigenvalues[m_sortOrder[std::min(targetNPE, m_me.numElements()) - 1]];
            if (targetNPE == 0) threshold = -std::numeric_limits<double>::infinity();
            if (verbose) std::cout << "targetNPE: " << targetNPE << std::endl;
        }
        else {
            threshold = relativeThreshold * m_referenceEigenvalue;
            // if (verbose) std::cout << "referenceEigenvalue: " << m_referenceEigenvalue << ", target threshold: " << threshold << std::endl;
        }
        threshold = std::min(threshold, 0.0); // We only want to project elements with negative eigenvalues...

        return threshold;
    }

    double defaultRelativeThreshold = 1e-3;
    double relativeThreshold = defaultRelativeThreshold;
    int maxFailuresBeforeShifting = 10;
    bool percentileControl = true; // If true, adjust the threshold to target a certain percentile of projected elements instead of halving/doubling.
    bool verbose = false;
    bool clampEigenvaluesToThreshold = false; // If true, also set the `eigenvalueClampTarget` to coincide with the projection threshold

private:
    void m_applyThreshold() {
        double threshold = currentThreshold();
        if (verbose) std::cout << "threshold: " << threshold << std::endl;

        m_me.elementHessianProjectionMasks = m_minimumEigenvalues.array() <= threshold;
        // if (verbose) std::cout << "number of elements actually projected: " << m_me.numProjectedElements() << std::endl;

        if (clampEigenvaluesToThreshold) m_me.setEigenvalueClampTarget(threshold);
        else m_me.setEigenvalueClampTarget(0);
    }

    double m_referenceEigenvalue;
    Eigen::VectorXd m_g, m_minimumEigenvalues;
    std::vector<int> m_sortOrder;
    MeshEnergyBase &m_me;
    int m_failures = 0;
};

#endif /* end of include guard: MASKEDHESSIANPROJECTIONCONTROLLER_HH */
