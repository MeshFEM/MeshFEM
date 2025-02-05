////////////////////////////////////////////////////////////////////////////////
// HessianProjectionController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements various strategies for enabling/disabling an object's Hessian
//  projection within our Newton solver.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/20/2020 18:11:49
////////////////////////////////////////////////////////////////////////////////
#ifndef HESSIANPROJECTIONCONTROLLER_HH
#define HESSIANPROJECTIONCONTROLLER_HH

#include <memory>
#include <Eigen/Dense>

struct HessianProjectionController {
    // Whether projection is currently enabled
    virtual bool shouldUseProjection() const = 0;

    // Inform the controller that the current Hessian is indefinite.
    // Returns `recompute`: whether the controller wants the Hessian to be
    //                      reevaluated with projection for the current iteration
    virtual bool notifyDefiniteness(bool /* isIndefinite */) { return false; }
    virtual void notifyStep(const Eigen::VectorXd & /* step */) { }  // For heuristics that can depend on step length.
    virtual void notifyGradient(const Eigen::VectorXd & /* g */) { } // For heuristics that can depend on gradient
    virtual void notifyDirectionalDerivative(double /* directionalDerivative */) { } // For heuristics that can depend on directional derivative

    virtual void reset() { }

    virtual ~HessianProjectionController() { }

    virtual std::unique_ptr<HessianProjectionController> clone() const = 0;
};

// Always use Hessian projection when available (default)
struct HessianProjectionAlways : public HessianProjectionController {
    virtual bool shouldUseProjection() const override { return true; }
    virtual std::unique_ptr<HessianProjectionController> clone() const override {
        return std::make_unique<HessianProjectionAlways>();
    }

    using State = std::tuple<>;
    static State serialize(const HessianProjectionAlways &) { return std::make_tuple(); }
    static std::unique_ptr<HessianProjectionAlways> deserialize(const State &) { return std::make_unique<HessianProjectionAlways>(); }
};

// Never use Hessian projection
struct HessianProjectionNever : public HessianProjectionController {
    virtual bool shouldUseProjection() const override { return false; }
    virtual std::unique_ptr<HessianProjectionController> clone() const override {
        return std::make_unique<HessianProjectionNever>();
    }

    using State = std::tuple<>;
    static State serialize(const HessianProjectionNever &) { return std::make_tuple(); }
    static std::unique_ptr<HessianProjectionNever> deserialize(const State &) { return std::make_unique<HessianProjectionNever>(); }
};

// Use a simple hysteresis strategy to select between using a
// projection or the full, unprojected Hessian.
// If indefiniteness is repeatedly encountered
// (more than `numConsecutiveIndefiniteStepsBeforeEnable` times in a row),
// we switch to using the Hessian projection for `numProjectionStepsBeforeDisable`
// iterations before switching back to the full Hessian.
// By default we start with the projection active
// (since Hessians are generally indefinite at the start).
//
// If `numConsecutiveIndefiniteStepsBeforeEnable` is set to 0, we will
// enable projection immediately upon detecting indefiniteness, and
// recompute the Hessian for the current Newton step (rather than
// using Hessian shifts for the current iteration).
// This should only be done if the problem actually implements a
// Hessian projection because otherwise it introduces an unnecessary extra
// evaluation and factorization of the same unprojected Hessian
// already known to be indefinite.
struct HessianProjectionAdaptive : public HessianProjectionController {
    size_t numProjectionStepsBeforeDisable = 10;
    size_t numConsecutiveIndefiniteStepsBeforeEnable = 5;

    // When steps stagnate/fail to make progress, disable projection
    double stepLengthThresholdForDisable = 0;
    double directionalDerivativeThresholdForDisable = 0; // if directional derivative exceeds (becomes less negative than) this value, disable projection
    bool startWithProjectionActive = true;

    HessianProjectionAdaptive() { reset(); }
    HessianProjectionAdaptive(const HessianProjectionAdaptive &b) = default;

    virtual void reset() override {
        if (startWithProjectionActive) {
            projectionActive = true;
            switchCounter = numProjectionStepsBeforeDisable;
        }
        else {
            projectionActive = false;
            switchCounter = numConsecutiveIndefiniteStepsBeforeEnable;
        }
    }

    virtual bool shouldUseProjection() const override { return projectionActive; }

    virtual bool notifyDefiniteness(bool isIndefinite) override {
        if (projectionActive) {
            if (!isIndefinite) {
                if (--switchCounter == 0) {
                    projectionActive = false;
                    switchCounter = numConsecutiveIndefiniteStepsBeforeEnable;
                }
            }
            else { switchCounter = numProjectionStepsBeforeDisable; } // Full Hessian must be crazy indefinite if projection didn't even help!
        }
        else {
            if (isIndefinite) {
                if (numConsecutiveIndefiniteStepsBeforeEnable == 0) {
                    projectionActive = true;
                    switchCounter = numProjectionStepsBeforeDisable;
                    return true;
                }
                if (--switchCounter == 0) {
                    projectionActive = true;
                    switchCounter = numProjectionStepsBeforeDisable;
                }
            }
            else {
                switchCounter = numConsecutiveIndefiniteStepsBeforeEnable;
            }
        }
        return false; // Only re-evaluate Hessian in the special `numConsecutiveIndefiniteStepsBeforeEnable == 0` case!
    }

    virtual void notifyStep(const Eigen::VectorXd &step) override {
        if (stepLengthThresholdForDisable <= 0) return; // shortcut unnecessary norm computation
        if (step.norm() < stepLengthThresholdForDisable) {
            projectionActive = false;
            switchCounter = numConsecutiveIndefiniteStepsBeforeEnable;
        }
    }

    virtual void notifyDirectionalDerivative(double directionalDerivative) override {
        if (directionalDerivative > directionalDerivativeThresholdForDisable) {
            projectionActive = false;
            switchCounter = numConsecutiveIndefiniteStepsBeforeEnable;
        }
    }

    virtual std::unique_ptr<HessianProjectionController> clone() const override {
        return std::make_unique<HessianProjectionAdaptive>(*this);
    }

    // Internal state (not intended to be modified directly, but still exposed to Python for experimentation)
    bool projectionActive;
    size_t switchCounter;

    // TODO: update serialization to include all members
    using State = std::tuple<size_t, size_t>; // Only store external state (internal state will be reset before next Newton solve anyway...)
    static State serialize(const HessianProjectionAdaptive &hpa) { return std::make_tuple(hpa.numProjectionStepsBeforeDisable, hpa.numConsecutiveIndefiniteStepsBeforeEnable); }
    static std::unique_ptr<HessianProjectionAdaptive> deserialize(const State &s) {
        auto hpa = std::make_unique<HessianProjectionAdaptive>();
        hpa->numProjectionStepsBeforeDisable           = std::get<0>(s);
        hpa->numConsecutiveIndefiniteStepsBeforeEnable = std::get<1>(s);
        return hpa;
    }
};

#endif /* end of include guard: HESSIANPROJECTIONCONTROLLER_HH */
