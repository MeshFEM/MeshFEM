#pragma once

#include <MeshFEM/Elements/ParametrizationElement.hh>
#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>

namespace MeshFEM {

template<size_t Deg, class Psi_2x2, class CustomMat_ = ParametrizationMaterial<Psi_2x2>>
struct ParametrizationElementProjectToRestHessian : public ParametrizationElement<Deg, Psi_2x2, CustomMat_> {
    using Base      = ParametrizationElement<Deg, Psi_2x2, CustomMat_>;
    using Real      = typename Base::Real;
    using HLE       = typename Base::HLE;
    using LocalVars = typename Base::LocalVars;
    using Hessian   = typename Base::Hessian;

    using Base::Base;

    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const {
        const auto &m = Base::material();
        if (projectionMask)
            return HLE::template hessian<SetLowerTri>(m.psi, RotationGetter(*this, x), this->elementData, /* projectionDisabled  = */ true, weight);
        return HLE::template hessian<SetLowerTri>(m.psi, this->FBGetter(x), this->elementData, /* projectionDisabled  = */ true, weight);
    }

private:
    struct RotationGetter {
        using M2d = Eigen::Matrix<Real, 2, 2>;

        RotationGetter(const Base &base, const LocalVars &x) : m_base(base), m_x(x) { }

        template<class GradPhis>
        M2d operator()(const GradPhis &gphis) const {
            return fast_decompositions::closest_rotation(m_base.FBGetter(m_x)(gphis));
        }

        const Base &m_base;
        const LocalVars &m_x;
    };
};

template<size_t Deg, class Psi_2x2, class CustomMat_ = ParametrizationMaterial<Psi_2x2>>
struct ParametrizationElementAKVF : public ParametrizationElement<Deg, Psi_2x2, CustomMat_> {
    using Base      = ParametrizationElement<Deg, Psi_2x2, CustomMat_>;
    using Real      = typename Base::Real;
    using HLE       = typename Base::HLE;
    using LocalVars = typename Base::LocalVars;
    using Hessian   = typename Base::Hessian;

    using Base::Base;

    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const {
        const auto &m = Base::material();
        if (projectionMask) {
            LinearlyEmbeddedElement<2, Deg, Vec2_T<Real>> deformed_edata;
            deformed_edata.embed(x);
            return HLE::template hessian<SetLowerTri>(m.psi, IdentityFGetter(), deformed_edata, /* projectionDisabled  = */ true, weight);
        }
        return HLE::template hessian<SetLowerTri>(m.psi, this->FBGetter(x), this->elementData, /* projectionDisabled  = */ true, weight);
    }

private:
    struct IdentityFGetter {
        using M2d = Eigen::Matrix<Real, 2, 2>;

        template<class GradPhis>
        const M2d &operator()(const GradPhis &/* gphis */) const {
            static const M2d I = M2d::Identity();
            return I;
        }
    };
};

template<class Psi_2x2, size_t Deg = 1>
using ParametrizationMeshEnergyProjectToRestHessian = MeshEmbeddingEnergy<FEMMesh<2, Deg, Vector3D>, 2, ParametrizationElementProjectToRestHessian<Deg, Psi_2x2>>;

template<class Psi_2x2, size_t Deg = 1>
using ParametrizationMeshEnergyAKVF = MeshEmbeddingEnergy<FEMMesh<2, Deg, Vector3D>, 2, ParametrizationElementAKVF<Deg, Psi_2x2>>;

} // namespace MeshFEM
