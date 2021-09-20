////////////////////////////////////////////////////////////////////////////////
// StVenantKirchhoff.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Volumetric and plane stress Saint Venant-Kirchhoff model; this is just like
//  a linearly elastic material except it uses the finite Green-Lagrangian strain
//  for proper geometric nonlinearities.
//  While many references define an isotropic version of Saint Venant-Kirchhoff
//  in terms of Lame constants, our implementation supports anisotropic
//  properties.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/23/2020 14:21:03
////////////////////////////////////////////////////////////////////////////////
#ifndef STVENANTKIRCHHOFF_HH
#define STVENANTKIRCHHOFF_HH

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/SymmetricMatrix.hh>

#include "EDensityAdaptors.hh"

// Implementation of psi(C), where C = F^T F is the Cauchy-Green deformation
// tensor Note: the elastic object classes need to use a F-based interface, but
// some applications (e.g., tension field theory relaxation) prefers to work
// with C and PK2 stress.
template <typename _Real, size_t _Dimension>
struct StVenantKirchhoffEnergyCBased : public Concepts::StVKEnergy {
    static constexpr size_t Dimension = _Dimension;
    static constexpr size_t N         = _Dimension;
    static constexpr EDensityType EDType = EDensityType::CBased;
    using Real    = _Real;
    using Matrix  = Eigen::Matrix<_Real, N, N>;
    using ETensor = ElasticityTensor<_Real, N>;
    using SMatrix = SymmetricMatrixValue<_Real, N>;

    // Default to an isotropic tensor that should be roughly equivalent to the old
    // incompressible neo-Hookean membrane energy with "stiffness = 1"  (E = 1 / 6)
    StVenantKirchhoffEnergyCBased(const ETensor &E = ETensor(6.0, 0.5)) : m_elasticity_tensor(E) {
        setC(Matrix::Identity());
    }

    StVenantKirchhoffEnergyCBased(const StVenantKirchhoffEnergyCBased&) = default;

    // Constructor copying material properties only, not the current deformation
    StVenantKirchhoffEnergyCBased(const StVenantKirchhoffEnergyCBased &other,
                                  UninitializedDeformationTag &&)
        : m_elasticity_tensor(other.m_elasticity_tensor) { }

    void setC(const Matrix &C) {
        m_strain = SMatrix((C - Matrix::Identity()).eval(), typename SMatrix::skip_validation());
        m_strain *= 0.5;
        m_stress = m_elasticity_tensor.doubleContract(m_strain);
    }

    Real energy() const {
        return 0.5 * m_strain.doubleContract(m_stress);
    }

    // d psi / d E,     E := 0.5 (C - I)
    Matrix PK2Stress() const { return m_stress.toMatrix(); }

    template<class Mat_>
    Matrix delta_PK2Stress(const Mat_ &dC) const {
        return m_elasticity_tensor.doubleContract(SMatrix(0.5 * dC)).toMatrix();
    }

    // Hessian is constant, third derivatives are zero.
    template<class Mat_, class Mat2_>
    Matrix delta2_PK2Stress(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const { return Matrix::Zero(); }

    void setElasticityTensor(const ETensor &et) {
        m_elasticity_tensor = et;
        m_stress = m_elasticity_tensor.doubleContract(m_strain);
    }

    void copyMaterialProperties(const StVenantKirchhoffEnergyCBased &other) { setElasticityTensor(other.elasticityTensor()); }

    const ETensor &elasticityTensor() const { return m_elasticity_tensor; }
protected:
    ETensor m_elasticity_tensor;
    SMatrix m_strain, m_stress; // Work-conjugate Green-Lagrange strain and PK2 stress.
};

template <typename _Real, size_t _Dimension>
using StVenantKirchhoffEnergy = EnergyDensityFBasedFromCBased<StVenantKirchhoffEnergyCBased<_Real, _Dimension>>;

template <typename _Real>
struct StVenantKirchhoffMembraneEnergy : public EnergyDensityFBasedFromCBased<StVenantKirchhoffEnergyCBased<_Real, 2>, 3> {
    using Base = EnergyDensityFBasedFromCBased<StVenantKirchhoffEnergyCBased<_Real, 2>, 3>;
    using Base::Base;
    static constexpr EDensityType EDType = EDensityType::Membrane;
    static constexpr const char *name() { return "StVenantKirchhoffMembrane"; }
};

#endif /* end of include guard: STVENANTKIRCHHOFF_HH */
