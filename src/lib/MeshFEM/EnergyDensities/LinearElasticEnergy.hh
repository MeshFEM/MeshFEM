#ifndef LINEARELASTICENERGY_HH
#define LINEARELASTICENERGY_HH

#include <Eigen/Dense>
#include "../ElasticityTensor.hh"
#include "../GlobalBenchmark.hh"
#include "../SymmetricMatrix.hh"
#include "../Utilities/Tensor.hh"

// template<typename Energy>
// concept bool EnergyType = 
//     require(Energy e) {
//         {energy()} -> Energy::Real;
//         {denergy()} -> Energy::Matrix;
//         {denergy(Energy::Matrix)} -> Energy::Real;
//         {d2energy(Energy::Matrix, Energy::Matrix)} -> Energy::Real;
//         {delta_denergy(Energy::Matrix)} -> Energy::Matrix;
//     };

template <typename _Real, size_t _Dimension>
struct LinearElasticEnergy {
    using SMatrix = SymmetricMatrixValue<_Real, _Dimension>;
    using FlatSymmetricMatrix = Eigen::Matrix<_Real, SMatrix::flatSize(), 1>;

   public:
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
        : m_elastic_tensor(elasticity_tensor) {}

    LinearElasticEnergy(const ETensor& elasticity_tensor,
                        const Matrix& deformation_gradient)
        : m_elastic_tensor(elasticity_tensor) {
        setDeformationGradient(deformation_gradient);
    }

    LinearElasticEnergy(const LinearElasticEnergy&) = default;

    void setDeformationGradient(const Matrix& deformation_gradient) {
        m_small_strain_tensor = SMatrix(
                0.5 * (deformation_gradient + deformation_gradient.transpose()) - Matrix::Identity(),
            typename SMatrix::skip_validation());
    }

    _Real energy() const {
        return m_small_strain_tensor.doubleContract(
                   m_elastic_tensor.doubleContract(m_small_strain_tensor)) /
               2;
    }

    /**
     *  Return the gradient of the energy density in respect of the deformation
     *  matrix in the direction of \a dF.
     *
     *  @param dF the direction
     */
    _Real denergy(const Matrix& dF) const {
        return doubleContract(
            dF, m_elastic_tensor.doubleContract(m_small_strain_tensor));
    }

    Matrix denergy() const {
        auto stress = m_elastic_tensor.doubleContract(m_small_strain_tensor);
        Matrix result;
        for (size_t i = 0; i < Dimension; ++i)
            for (size_t j = 0; j < Dimension; ++j)
                result(i, j) = stress(i, j);
        return result;
    }

#if 1
    /**
     *  Returns dF_lhs : H : dF_rhs, where H is the hessian of the energy
     *  density in respect to the deformation gradient.
     */
    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        SMatrix a(dF_lhs + dF_lhs.transpose(), typename SMatrix::skip_validation());
        SMatrix b(dF_rhs + dF_rhs.transpose(), typename SMatrix::skip_validation());
        return 0.25 * a.doubleContract(m_elastic_tensor.doubleContract(b));
    }
#else
    _Real d2energy(const Matrix& dF_lhs, const Matrix& dF_rhs) const {
        _Real result =
            doubleContract(dF_lhs, doubleContract(m_elastic_tensor, dF_rhs));
        return result;
    }
#endif

    Matrix delta_denergy(const Matrix& dF) const {
        SMatrix sym = m_elastic_tensor.doubleContract(
                    SMatrix(dF + dF.transpose(), typename SMatrix::skip_validation()));
        sym *= 0.5;
        Matrix result;
        for (size_t i = 0; i < Dimension; ++i)
            for (size_t j = 0; j < Dimension; ++j)
                result(i, j) = sym(i, j);

        return result;
    }

   private:
    ETensor m_elastic_tensor;
    SMatrix m_small_strain_tensor;
};

#endif
