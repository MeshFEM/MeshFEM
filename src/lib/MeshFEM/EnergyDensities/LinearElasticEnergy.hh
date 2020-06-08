#ifndef LINEARELASTICENERGY_HH
#define LINEARELASTICENERGY_HH

#include <Eigen/Dense>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

template <typename _Real, size_t _Dimension>
struct LinearElasticEnergy : public LinearElaticEnergyConcept {
    using SMatrix = SymmetricMatrixValue<_Real, _Dimension>;

    static constexpr size_t Dimension = _Dimension;
    using Real = _Real;
    using Matrix = Eigen::Matrix<_Real, _Dimension, _Dimension>;
    using ETensor = ElasticityTensor<_Real, _Dimension>;

    /**
     *  Construct a linear elastic energy density with a default initialized
     *  deformation gradient.
     *
     *  It is undefined behavior to call any methods other than
     *  setDeformationGradient before initializing the deformation gradient
     *  with setDeformationGradient.
     */
    LinearElasticEnergy(const ETensor& elasticity_tensor)
        : m_elasticity_tensor(elasticity_tensor) {}

    LinearElasticEnergy(const ETensor& elasticity_tensor,
                        const Matrix& deformation_gradient)
        : m_elasticity_tensor(elasticity_tensor) {
        setDeformationGradient(deformation_gradient);
    }

    LinearElasticEnergy(const LinearElasticEnergy&) = default;

    // Constructor copying material properties only, not the current deformation
    LinearElasticEnergy(const LinearElasticEnergy &other, const UninitializedDeformationTag &)
        : m_elasticity_tensor(other.m_elasticity_tensor) { }

    void setDeformationGradient(const Matrix& deformation_gradient) {
        m_small_strain_tensor = symmetrized(deformation_gradient - Matrix::Identity());
    }

    _Real energy() const {
        return m_small_strain_tensor.doubleContract(
                   m_elasticity_tensor.doubleContract(m_small_strain_tensor)) /
               2;
    }

    /**
     *  Return the gradient of the energy density in respect of the deformation
     *  matrix in the direction of \a dF.
     *
     *  @param dF the direction
     */
    _Real denergy(const Matrix &dF) const {
        return doubleContract(
            dF, m_elasticity_tensor.doubleContract(m_small_strain_tensor));
    }

    Matrix denergy() const { return m_elasticity_tensor.doubleContract(m_small_strain_tensor).toMatrix(); }

#if 1
    /**
     *  Returns dF_lhs : H : dF_rhs, where H is the hessian of the energy
     *  density in respect to the deformation gradient.
     */
    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return symmetrized(dF_rhs).doubleContract(
                    m_elasticity_tensor.doubleContract(symmetrized(dF_rhs)));
    }
#else
    _Real d2energy(const Matrix& dF_lhs, const Matrix& dF_rhs) const {
        _Real result =
            doubleContract(dF_lhs, doubleContract(m_elasticity_tensor, dF_rhs));
        return result;
    }
#endif

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        return m_elasticity_tensor.doubleContract(symmetrized(dF)).toMatrix();
    }

    // Hessian is constant, third derivatives are zero.
    Matrix delta2_denergy(const Matrix &/* dF_a */, const Matrix &/* dF_b */) const { return Matrix::Zero(); }

    Matrix PK2Stress() const { throw std::runtime_error("Unimplemented"); }
private:
    ETensor m_elasticity_tensor;
    SMatrix m_small_strain_tensor;
};

#endif
