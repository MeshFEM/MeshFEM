#ifndef MESHENERGYBASE_HH
#define MESHENERGYBASE_HH

#include "Elements/MaterialAssignment.hh"
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>

struct MeshEnergyBase : public NewtonObjectiveTerm {
    MeshEnergyBase(std::shared_ptr<NewtonVarsBase> vars)
        : NewtonObjectiveTerm(vars) { }

    MaterialBase &materialForElement(size_t ei) {
        if (ei >= numElements()) throw std::runtime_error("Element index out of bounds");
        auto &mat = m_getMaterial(ei);
        return mat;
    }

    virtual size_t numElements() const = 0;

    virtual VXd elementHessianMinimumEigenvalues() const { throw std::runtime_error("elementHessianMinimumEigenvalues not implemented for this energy"); }

    // Norm of the fully assembled gradient restricted to the nodes of an element.
    // Useful to determine if the element's neighborhood is close to equilibrium.
    VXd elementGradientNorms() const { return elementGradientNorms(gradient()); }
    virtual VXd elementGradientNorms(const VXd &g) const = 0;

    bool hasPerElementHessianProjectionMasks() const {
        if (elementHessianProjectionMasks.size() == 0) return false;
        if (elementHessianProjectionMasks.size() != int(numElements())) throw std::runtime_error("elementHessianProjectionMasks size does not match number of elements");
        return true;
    }

    virtual void   setEigenvalueClampTarget(double) { throw std::runtime_error("setEigenvalueClampTarget not implemented for this energy"); }
    virtual double getEigenvalueClampTarget() const { throw std::runtime_error("getEigenvalueClampTarget not implemented for this energy"); }

    size_t numProjectedElements() const {
        if (!hasPerElementHessianProjectionMasks()) return numElements();
        return elementHessianProjectionMasks.array().count();
    }

    virtual ~MeshEnergyBase() { }

    bool useXBasedProjection = false;
    double xBasedProjectionClampEps = 0.0;
    double elementHessianShift = 0.0;

    // Optional flags to disable/enable Hessian projection on a per-element
    // basis (when the global `projectionMask` argument is `true`).
    Eigen::Array<bool, Eigen::Dynamic, 1> elementHessianProjectionMasks;
private:
    virtual MaterialBase &m_getMaterial(size_t ei) = 0;
};

#endif /* end of include guard: MESHENERGYBASE_HH */
