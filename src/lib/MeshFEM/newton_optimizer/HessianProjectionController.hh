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

struct HessianProjectionController {
    virtual bool shouldUseProjection() const = 0;
    virtual void notifyDefiniteness(bool /* isIndefinite */) { }
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
// (more than `numConsecutiveIndefiniteStepsBeforeSwitch` times in a row),
// we switch to using the Hessian projection for `numProjectionStepsBeforeSwitch`
// iterations before switching back to the full Hessian.
// By default we start with the projection active
// (since the problem is generally indefinite at the start).
struct HessianProjectionAdaptive : public HessianProjectionController {
    size_t numProjectionStepsBeforeSwitch = 15;
    size_t numConsecutiveIndefiniteStepsBeforeSwitch = 5;

    HessianProjectionAdaptive() { reset(); }
    HessianProjectionAdaptive(const HessianProjectionAdaptive &b) = default;

    virtual void reset() override {
        projectionActive = true;
        switchCounter = numProjectionStepsBeforeSwitch;
    }

    virtual bool shouldUseProjection() const override { return projectionActive; }

    virtual void notifyDefiniteness(bool isIndefinite) override {
        if (projectionActive) {
            if (!isIndefinite) {
                if (--switchCounter == 0) {
                    projectionActive = false;
                    switchCounter = numConsecutiveIndefiniteStepsBeforeSwitch;
                }
            }
            else { switchCounter = numProjectionStepsBeforeSwitch; } // Full Hessian must be crazy indefinite if projection didn't even help!
        }
        else {
            if (isIndefinite) {
                if (--switchCounter == 0) {
                    projectionActive = true;
                    switchCounter = numProjectionStepsBeforeSwitch;
                }
            }
            else {
                switchCounter = numConsecutiveIndefiniteStepsBeforeSwitch;
            }
        }
    }

    virtual std::unique_ptr<HessianProjectionController> clone() const override {
        return std::make_unique<HessianProjectionAdaptive>(*this);
    }

    // Internal state (not intended to be modified directly, but still exposed to Python for experimentation)
    bool projectionActive;
    size_t switchCounter;

    using State = std::tuple<size_t, size_t>; // Only store external state (internal state will be reset before next Newton solve anyway...)
    static State serialize(const HessianProjectionAdaptive &hpa) { return std::make_tuple(hpa.numProjectionStepsBeforeSwitch, hpa.numConsecutiveIndefiniteStepsBeforeSwitch); }
    static std::unique_ptr<HessianProjectionAdaptive> deserialize(const State &s) {
        auto hpa = std::make_unique<HessianProjectionAdaptive>();
        hpa->numProjectionStepsBeforeSwitch            = std::get<0>(s);
        hpa->numConsecutiveIndefiniteStepsBeforeSwitch = std::get<1>(s);
        return hpa;
    }
};

#endif /* end of include guard: HESSIANPROJECTIONCONTROLLER_HH */
