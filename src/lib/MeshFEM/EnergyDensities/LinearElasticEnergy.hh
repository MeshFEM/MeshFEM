#ifndef LINEARELASTICENERGY_HH
#define LINEARELASTICENERGY_HH

#include <Eigen/Dense>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

template <typename _Real, size_t _Dimension>
struct LinearElasticEnergy : public Concepts::LinearElaticEnergy {
    static constexpr EDensityType EDType = EDensityType::FBased;

    static constexpr size_t N = _Dimension;

    using Real = _Real;
    using Matrix = Eigen::Matrix<_Real, N, N>;
    using ETensor = ElasticityTensor<_Real, N>;
    using SMatrix = SymmetricMatrixValue<_Real, N>;

    /**
     *  Construct a linear elastic energy density with a default initialized
     *  deformation gradient.
     *
     *  It is undefined behavior to call any methods other than
     *  setDeformationGradient before initializing the deformation gradient
     *  with setDeformationGradient.
     */
    LinearElasticEnergy(const ETensor& elasticity_tensor)
        : m_elasticity_tensor(elasticity_tensor) {
        setDeformationGradient(Matrix::Identity());
    }

    LinearElasticEnergy(const LinearElasticEnergy&) = default;

    // Constructor copying material properties only, not the current deformation
    LinearElasticEnergy(const LinearElasticEnergy &other, const UninitializedDeformationTag &)
        : m_elasticity_tensor(other.m_elasticity_tensor) { }

    void setDeformationGradient(const Matrix &F, const EvalLevel /* elevel */ = EvalLevel::Full) {
        m_F = F;
        m_small_strain_tensor = symmetrized(F - Matrix::Identity());
    }
    const Matrix &getDeformationGradient() const { return m_F; }

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

    /**
     *  Returns dF_lhs : H : dF_rhs, where H is the hessian of the energy
     *  density in respect to the deformation gradient.
     */
    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return symmetrized(dF_rhs).doubleContract(
                    m_elasticity_tensor.doubleContract(symmetrized(dF_lhs)));
    }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        return m_elasticity_tensor.doubleContract(symmetrized(dF)).toMatrix();
    }

    // Hessian is constant, third derivatives are zero.
    Matrix delta2_denergy(const Matrix &/* dF_a */, const Matrix &/* dF_b */) const { return Matrix::Zero(); }

    Matrix PK2Stress() const { throw std::runtime_error("Unimplemented"); }
protected:
    Matrix m_F = Matrix::Identity();
    ETensor m_elasticity_tensor;
    SMatrix m_small_strain_tensor;
};

#endif
