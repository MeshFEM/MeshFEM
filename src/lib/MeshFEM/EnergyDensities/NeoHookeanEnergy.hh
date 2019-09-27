#ifndef NEOHOOKEANENERGY_HH
#define NEOHOOKEANENERGY_HH

#include <Eigen/Dense>
#include <cstdlib>

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/EnergyDensities/DifferentialOperations.hh>

template<typename _Derived>
struct NeoHookeanEnergyTraits;

/**
 *  Implements the Neo-Hookean Energy described in the Neo-Hookean Energy section of doc/doc.pdf
 */
template<typename _Derived>
class NeoHookeanEnergyBase : public NeoHookeanEnergyConcept
{
public:
    using Derived = _Derived;
    static constexpr size_t Dimension = NeoHookeanEnergyTraits<_Derived>::Dimension;
    /**
     *  Used to avoid access to incomplete type member.
     */
    using Real = typename NeoHookeanEnergyTraits<Derived>::Real;
    using Matrix = Eigen::Matrix<Real, Dimension, Dimension>;

    /**
     *  Copy constructor. Does not copy the deformation gradient, only parameters.
     */
    NeoHookeanEnergyBase(const NeoHookeanEnergyBase& other) = default;

    NeoHookeanEnergyBase(Real lame_first_parameter, Real shear_modulus,
        Real finite_continuation_start = -1)
        : m_lame_first_parameter(lame_first_parameter), m_shear_modulus(shear_modulus),
        m_finite_continuation_start(finite_continuation_start) 
    {
    };

    void setDeformationGradient(const Matrix& deformation_gradient)
    {
        m_deformation_gradient = deformation_gradient;
        m_deformation_gradient_determinant = deformation_gradient.determinant();
    }

    Real energy() const
    {
        Real detF = getDeformationGradientDeterminant();

        // Standard behavior: return inf for inverted elements
        if (m_finite_continuation_start < 0 && detF < 0) {
            return std::numeric_limits<Real>::max();
        }

        // Modified behavior to support inverted elements:
        // if det F < eps, we replace the log(I3) term by a constant + exp(- (det (F) - eps) )
        // where the constant is chosen such that the energy remains continuous
        if (m_finite_continuation_start > 0 && detF < m_finite_continuation_start) {
            Real I3 = getI3();
            Real I1 = getI1();

            Matrix tmp_F;
            tmp_F.setIdentity();
            tmp_F(0, 0) = m_finite_continuation_start;
            Real continuation_constant = - std::log(getI3(tmp_F)) * (m_lame_first_parameter / 2 + m_shear_modulus) / 2;

            return m_lame_first_parameter * (I3 - 1) / 4 + m_shear_modulus * (I1 - 3) / 2 
                + continuation_constant + std::exp(-(detF - m_finite_continuation_start)) - 1;
        }

        Real I3 = getI3();
        Real I1 = getI1();
        return m_lame_first_parameter * (I3 - 1) / 4 + m_shear_modulus * (I1 - 3) / 2 -
            std::log(I3) * (m_lame_first_parameter / 2 + m_shear_modulus) / 2;
    }

    Matrix denergy() const {

        Real detF = m_deformation_gradient.determinant();
        if (m_finite_continuation_start > 0 && detF < m_finite_continuation_start) {
            Real dPsi3 = m_lame_first_parameter / 4;
            return (-std::exp(-(detF - m_finite_continuation_start))) * getDifferentiatedDeterminant(m_deformation_gradient)
                + dPsi3 * getDI3()
                + getDPsi1() * getDI1();
        }

        return getDPsi3() * getDI3() + getDPsi1() * getDI1();
    }

    Real denergy(const Matrix& dF) const { return (dF * denergy().transpose()).trace(); }

    Real d2energy(const Matrix& dF_lhs, const Matrix& dF_rhs) const
    {
        return doubleContract(dF_lhs, delta_denergy(dF_rhs));
    }

    /**
     *  Return (H : dF) where H is the hessian of the energy density.
     */
    Matrix delta_denergy(const Matrix& dF) const
    {

        Real detF = m_deformation_gradient.determinant();
        if (m_finite_continuation_start > 0 && detF < m_finite_continuation_start) {
            // ln I3 term is constant, but exp(-(detF)) got added
            Real dPsi3 = m_lame_first_parameter / 4;
            Matrix ddetF = getDifferentiatedDeterminant(m_deformation_gradient);
            Real exp_term = -std::exp(-(detF - m_finite_continuation_start));
            return exp_term * ddetF * ddetF
                + exp_term * getDifferentiatedTwiceDeterminant(m_deformation_gradient, dF)
                + dPsi3 * getDDI3(dF)
                + getDPsi1() * getDDI1(dF);
        }

        return getDPsi1() * getDDI1(dF) + getDDPsi3(dF) * getDI3() + getDPsi3() * getDDI3(dF);
    }

    /**
     *  Return the right cauchy-green deformation tensor's third invariant.
     */
    Real getI3() const { return static_cast<const Derived*>(this)->getI3(m_deformation_gradient); }
    Real getI3(const Matrix&) const { return static_cast<const Derived*>(this)->getI3(m_deformation_gradient); };

    /**
     *  Return the right cauchy-green deformation tensor's first invariant.
     */
    Real getI1() const { return static_cast<const Derived*>(this)->getI1(); }

    /**
     *  Return the right cauchy-green deformation tensor's first invariant differentiated with
     *  respect to the deformation gradient. (dI1/dF)
     */
    Matrix getDI1() const { return static_cast<const Derived*>(this)->getDI1(); }

    /**
     *  Return the right cauchy-green deformation tensor's third invariant differentiated with
     *  respect to the deformation gradient. (dI3/dF)
     */
    Matrix getDI3() const { return static_cast<const Derived*>(this)->getDI3(); }

    /**
     *  Return the right cauchy-green deformation tenso's first invariant differentiated with
     *  respect to the deformation gradient differentiated again in the given deformation gradient
     *  direction. (d^2 I1/dF^2 : dF)
     */
    Matrix getDDI1(const Matrix& dF) const
    {
        return static_cast<const Derived*>(this)->getDDI1(dF);
    }

    /**
     *  Return the right cauchy-green deformation tenso's first invariant differentiated with
     *  respect to the deformation gradient derived again in the given deformation gradient
     *  direction. (d^2 I3/dF^2 : dF)
     */
    Matrix getDDI3(const Matrix& dF) const
    {
        return static_cast<const Derived*>(this)->getDDI3(dF);
    }

    /**
     *  Returns the derivative of the energy density with respect to the deformation gradient
     *  tensor's first invariant. (dPsi/dI1)
     */
    Real getDPsi1() const { return m_shear_modulus / 2; }

    /**
     *  Returns the derivative of the energy density with respect to the deformation gradient
     *  tensor's first invariant. (dPsi/dI3)
     */
    Real getDPsi3() const
    {
        return (m_lame_first_parameter - (2 * m_shear_modulus + m_lame_first_parameter) / getI3()) /
            4;
    }

    /**
     *  Return the derivative of the energy density with respect to the deformation gradient
     *  tensor's first invariant derived again in the given deformation gradient direction.
     *  ((d/dF * dPsi/dI3) : dF)
     */
    Real getDDPsi3(const Matrix& dF) const
    {
        Real I3_squared = getI3();
        I3_squared *= I3_squared;

        return (2 * m_shear_modulus + m_lame_first_parameter) * doubleContract(getDI3(), dF) /
            (4 * I3_squared);
    }

protected:
    Matrix getDifferentiatedTwiceDeformationGradientDeterminantSquared(const Matrix& dF) const
    {
        Matrix deformation_gradient_inverse_transpose = m_deformation_gradient.inverse().transpose();
        return 4 * getDeformationGradientDeterminantSquared() *
            doubleContract(deformation_gradient_inverse_transpose, dF) *
            deformation_gradient_inverse_transpose -
            2 * getDeformationGradientDeterminantSquared() *
            deformation_gradient_inverse_transpose * dF.transpose() *
            deformation_gradient_inverse_transpose;

    }

    Matrix getDifferentiatedDeformationGradientDeterminantSquared() const
    {
        return 2 * getDeformationGradientDeterminantSquared() * m_deformation_gradient.inverse().transpose();
    }

    Real getDeformationGradientDeterminantSquared() const
    {
        return getDeformationGradientDeterminant() * getDeformationGradientDeterminant();
    }

    Real getDeformationGradientDeterminant() const
    {
        return m_deformation_gradient_determinant;
    }

    Matrix m_deformation_gradient;
    Real m_lame_first_parameter, m_shear_modulus;
    Real m_finite_continuation_start;
    Real m_deformation_gradient_determinant;
};

template<typename _Real, size_t _Dimension>
class NeoHookeanEnergy;

template<typename _Real, size_t _Dimension>
struct NeoHookeanEnergyTraits<NeoHookeanEnergy<_Real, _Dimension>>
{
    using Real = _Real;
    static constexpr size_t Dimension = _Dimension;
};

template<typename _Real>
class NeoHookeanEnergy<_Real, 2> : public NeoHookeanEnergyBase<NeoHookeanEnergy<_Real, 2>>
{
    using Base = NeoHookeanEnergyBase<NeoHookeanEnergy<_Real, 2>>;

public:
    static constexpr size_t Dimension = 2;
    using Real = _Real;
    using Matrix = typename Base::Matrix;

    using Base::Base;

    /**
     *  Return the right cauchy-green deformation tensor's third invariant.
     */
    Real getI3(const Matrix& F) const { return getF2DeterminantSquared(F) * getC33(F); }

    /**
     *  Return the right cauchy-green deformation tensor's third invariant differantiated with
     *  respect to the 2D deformation gradient.
     */
    Matrix getDI3() const
    {
        return getC33() * getDF2DeterminantSquared() + getF2DeterminantSquared() * getDC33();
    }

    Matrix getDDI3(const Matrix& dF) const
    {
        Matrix dC33 = getDC33();
        Matrix dF2DeterminantSquared = getDF2DeterminantSquared();

        return getC33() * getDDF2DeterminantSquared(dF) +
            doubleContract(dC33, dF) * dF2DeterminantSquared +
            getF2DeterminantSquared() * getDDC33(dF) +
            doubleContract(dF2DeterminantSquared, dF) * dC33;
    }

    /**
     *  Return the right cauchy-green deformation tensor's first invariant.
     */
    Real getI1() const { return m_deformation_gradient.squaredNorm() + getC33(); }

    /**
     *  Return the right cauchy-green deformation tensor's first invariant differentiated with
     *  respect to the 2D deformation gradient.
     */
    Matrix getDI1() const { return 2 * m_deformation_gradient + getDC33(); }

    Matrix getDDI1(const Matrix& dF2) const { return 2 * dF2 + getDDC33(dF2); }

    /**
     *  Return the deformation normal to the surface squared
     */
    Real getC33(const Matrix& F) const
    {
        return (m_lame_first_parameter + 2 * m_shear_modulus) /
            (m_lame_first_parameter * getF2DeterminantSquared(F) + 2 * m_shear_modulus);
    }
    Real getC33() const { return getC33(m_deformation_gradient); }

    /**
     *  Return the deformation normal to the surface squared differentiated with respect to the
     *  2D deformation gradient.
     */
    Matrix getDC33() const { return getDC33Determinant() * getDF2DeterminantSquared(); }

    Matrix getDDC33(const Matrix& dF2) const
    {
        return getDDC33Determinant(dF2) * getDF2DeterminantSquared() +
            getDC33Determinant() * getDDF2DeterminantSquared(dF2);
    }

    /**
     *  Return the deformation normal to the surface squared differentiated with respect to the
     *  determinant of the deformation gradient squared. (dC33/d(det F2^2))
     */
    Real getDC33Determinant() const
    {
        Real denominator = m_lame_first_parameter * getF2DeterminantSquared() + 2 * m_shear_modulus;
        denominator *= denominator;
        return (-m_lame_first_parameter * (m_lame_first_parameter + 2 * m_shear_modulus) /
            denominator);
    }

    /**
     *  Return the deformation normal to the surface squared differentiated with respect to
     *  the deformation gradient determinant squared then differentiated again in the given
     *  deformation gradient direction. (d/dF2 (d C33/d(det F2)) : dF2)
     */
    Real getDDC33Determinant(const Matrix& dF2) const
    {
        Real denominator = m_lame_first_parameter * getF2DeterminantSquared() + 2 * m_shear_modulus;
        denominator *= denominator * denominator;
        return (2 * m_lame_first_parameter * m_lame_first_parameter *
            (m_lame_first_parameter + 2 * m_shear_modulus) *
            doubleContract(getDF2DeterminantSquared(), dF2)) /
            denominator;
    }

    /**
     *  Return the determinant of the 2D deformation gradient squared differantiated with
     *  respect to the 2D deformation gradient then differentiated again in the given 2D deformation
     *  gradient direction. (d/dF2 (d (det F2^2) /dF2) : dF2)
     */
    Matrix getDDF2DeterminantSquared(const Matrix& dF2) const
    {
        return this->getDifferentiatedTwiceDeformationGradientDeterminantSquared(dF2);
    }

    /**
     *  Return the 2D deformation gradient determinant squared differentiated with respect to the
     *  2D deformation gradient;
     */
    Matrix getDF2DeterminantSquared() const
    {
        return this->getDifferentiatedDeformationGradientDeterminantSquared();
    }

    /**
     *  Return the 2D deformation gradient determinant squared.
     */
    Real getF2DeterminantSquared(const Matrix& F) const { return getDeterminantSquared(F); }
    Real getF2DeterminantSquared() const { return this->getDeformationGradientDeterminantSquared(); }

private:
    using Base::m_deformation_gradient;
    using Base::m_lame_first_parameter;
    using Base::m_shear_modulus;
};

template<typename _Real>
class NeoHookeanEnergy<_Real, 3> : public NeoHookeanEnergyBase<NeoHookeanEnergy<_Real, 3>>
{
    using Base = NeoHookeanEnergyBase<NeoHookeanEnergy>;

public:
    static constexpr size_t Dimension = 2;
    using Real = _Real;
    using Matrix = typename Base::Matrix;

    using Base::Base;

    Real getI3(const Matrix& F) const { return getDeterminantSquared(F); }

    Matrix getDI3() const { return this->getDifferentiatedDeformationGradientDeterminantSquared(); }

    Matrix getDDI3(const Matrix& dF) const
    {
        return this->getDifferentiatedTwiceDeformationGradientDeterminantSquared(dF);
    }

    Real getI1() const
    {
        return (m_deformation_gradient * m_deformation_gradient.transpose()).trace();
    }

    Matrix getDI1() const { return 2 * m_deformation_gradient; }

    Matrix getDDI1(const Matrix& dF) const { return 2 * dF; }

private:
    using Base::m_deformation_gradient;
    using Base::m_lame_first_parameter;
    using Base::m_shear_modulus;
};

#endif
