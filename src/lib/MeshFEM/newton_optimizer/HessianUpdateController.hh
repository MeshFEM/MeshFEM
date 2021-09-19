////////////////////////////////////////////////////////////////////////////////
// HessianUpdateController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements various strategies for deciding when to update the Hessian
//  factorization used by our Newton solver and when to reuse the existing one.
//  TODO: smarter strategies based, e.g., on the Affine Covariant Newton algorithm
//  method (like in Mathematica).
//  We could also try to incorporate BFGS updates to an existing factorization
//  here.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/20/2020 22:05:12
////////////////////////////////////////////////////////////////////////////////
#ifndef HESSIANUPDATECONTROLLER_HH
#define HESSIANUPDATECONTROLLER_HH

#include <memory>

struct HessianUpdateController {
    virtual bool needsUpdate() const = 0;
    virtual void newHessian(bool /* isIndefinite */) { }
    virtual void reusedHessian() { }
    virtual void reset() { }

    virtual ~HessianUpdateController() { }
    virtual std::unique_ptr<HessianUpdateController> clone() const = 0;
};

// Always update the Hessian factorization (default)
struct HessianUpdateAlways: public HessianUpdateController {
    virtual bool needsUpdate() const override { return true; }
    virtual std::unique_ptr<HessianUpdateController> clone() const override {
        return std::make_unique<HessianUpdateAlways>(*this);
    }

    using State = std::tuple<>;
    static State serialize(const HessianUpdateAlways &) { return std::make_tuple(); }
    static std::unique_ptr<HessianUpdateAlways> deserialize(const State &) { return std::make_unique<HessianUpdateAlways>(); }
};

// Never update the Hessian factorization
struct HessianUpdateNever: public HessianUpdateController {
    virtual bool needsUpdate() const override { return false; }
    virtual std::unique_ptr<HessianUpdateController> clone() const override {
        return std::make_unique<HessianUpdateNever>(*this);
    }

    using State = std::tuple<>;
    static State serialize(const HessianUpdateNever &) { return std::make_tuple(); }
    static std::unique_ptr<HessianUpdateNever> deserialize(const State &) { return std::make_unique<HessianUpdateNever>(); }
};

// Update the Hessian factorization every `period` iterations
// (Setting period to 0 means the Hessian is always updated.)
struct HessianUpdatePeriodic: public HessianUpdateController {
    size_t period = 2;

    virtual void reset() override { m_counter = 0; }

    virtual bool needsUpdate() const override { return m_counter == 0; }

    virtual void newHessian(bool /* isIndefinite */) override { m_counter = period; }
    virtual void reusedHessian() override {
        if (m_counter > 0) --m_counter;
    }

    virtual std::unique_ptr<HessianUpdateController> clone() const override {
        return std::make_unique<HessianUpdatePeriodic>(*this);
    }

    using State = std::tuple<size_t>; // Only store external state (internal state will be reset before next Newton solve anyway...)
    static State serialize(const HessianUpdatePeriodic &hup) { return std::make_tuple(hup.period); }
    static std::unique_ptr<HessianUpdatePeriodic> deserialize(const State &s) {
        auto hup = std::make_unique<HessianUpdatePeriodic>();
        hup->period = std::get<0>(s);
        return hup;
    }

protected:
    size_t m_counter = 0; // countdown until a new Hessian is needed.
};

#endif /* end of include guard: HESSIANUPDATECONTROLLER_HH */
