#ifndef SHELLELEMENT_HH
#define SHELLELEMENT_HH

#include "HyperelasticLagrange.hh"
#include "PlateBending.hh"

// Implements both the `MembraneMaterial` and `PlateBendingMaterial` interfaces.
template<class Psi_2x2>
struct ShellMaterial : public PlateBendingMaterialProperties<typename Psi_2x2::Real> {
    using Base = PlateBendingMaterialProperties<typename Psi_2x2::Real>;
    using Real = typename Psi_2x2::Real;
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;
    using ETensor = ElasticityTensor<Real, 2>;
    using SM2d = SymmetricMatrixValue<Real, 2>;

    template<class Psi_>
    void setPsi(const Psi_ &psi_) {
        psi = Psi(psi_);
        Base::setPsi(psi_);
    }

    void setThickness(Real h) {
        thickness = h;
        Base::setThickness(h);
    }

    Real getThickness() const { return thickness; }

    const Psi &getPsi() const { return psi; }
          Psi &getPsi()       { return psi; }

private:
    // MembraneMaterial interface
    Psi psi;
    Real thickness = 1;
};

template<class Psi_2x2, class AngleFunction>
struct ShellElement;

template<class Psi_2x2, class AngleFunction>
struct ElementTraits<ShellElement<Psi_2x2, AngleFunction>> {
    using Material = ShellMaterial<Psi_2x2>;
};

template<class Psi_2x2, class AngleFunction = AngleFunctionIdentity>
struct ShellElement {
    using     Real = typename Psi_2x2::Real;
    using Material = ShellMaterial<Psi_2x2>; // Note: material assignments are tracked by `plate`
    using PBE = PlateBending<Real, AngleFunction, const elements::EmbeddedMembraneEData<2, 1, Vec3_T<Real>> &, Material>;
    using DTG = typename PBE::DTG;

    using Gradient = typename PBE::Gradient;
    using Hessian  = typename PBE::Hessian;

    enum class EnergyType { Full, Membrane, Bending };

    static constexpr size_t K = 2;
    static constexpr size_t N = 3;
    static constexpr size_t Deg = 1;
    using V3d = Eigen::Matrix<Real, 3, 1>;

    using HLE = elements::HyperelasticLagrange<typename Material::Psi, K, N, Deg>;
    using EData = elements::EmbeddedMembraneEData<K, Deg, V3d>;

    static constexpr bool CachesDeformedQuantities = true;

    template<class Mesh>
    ShellElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : elementData(*(m.element(ei))),
          plate(ei, elementData, materials) { }

    // Special implementation of FBGetter that evaluates the constant 3x2
    // deformation gradient for this element using the edge vectors already
    // precomputed in the plate bending element.
    struct FBGetter {
        FBGetter(const DTG &dtg) : dtg(dtg) { }
        auto operator()(const typename EData::GradPhis &gradPhis) const {
            return (dtg.edgeVecs.col(1) * gradPhis.col(0).transpose()
                  - dtg.edgeVecs.col(0) * gradPhis.col(1).transpose()).eval();
        }
        const DTG &dtg;
    };

    auto getFB() const { return FBGetter(plate.de)(elementData.gradPhis()); }

    // Update the triangle's deformed embedding.
    template<class CPosDerived>
    void embed(const Eigen::MatrixBase<CPosDerived> &cornerPositions) { plate.embed(cornerPositions); }

    // Update the gamma variables at the mid-edges
    void setGammas(const Eigen::Ref<const V3d> &g, EvalLevel /* elevel */ = EvalLevel::Full) { plate.setGammas(g); }

    // void setRestConfiguration(const LocalVars &X) {
    // }

    Real energy(EnergyType etype = EnergyType::Full) const {
        const auto &mat = plate.material();
        Real result = 0;

        // Membrane energy contribution
        if (etype == EnergyType::Full || etype == EnergyType::Membrane)
            result += HLE::energy(mat.getPsi(), FBGetter(plate.de), elementData) * mat.getThickness();

        // Bending energy contribution
        // (Only an approximation unless Psi is actually St Venant Kirchhoff...)
        if (etype == EnergyType::Full || etype == EnergyType::Bending)
            result += plate.energy();

        return result;
    }

    Gradient gradient(Real weight, EnergyType etype = EnergyType::Full) const {
        const auto &mat = plate.material();
        Gradient result = Gradient::Zero();

        // Membrane energy contribution
        if (etype == EnergyType::Full || etype == EnergyType::Membrane)
            result.template head<9>() = HLE::gradient(mat.getPsi(), FBGetter(plate.de), elementData, (weight * mat.getThickness()));

        // Bending energy contribution
        if (etype == EnergyType::Full || etype == EnergyType::Bending)
            plate.accumulateGradient(result, weight);

        return result;
    }

    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool membraneProjection, EnergyType etype = EnergyType::Full) const {
        const auto &mat = plate.material();
        Hessian result = Hessian::Zero();

        // Membrane energy contribution
        if (etype == EnergyType::Full || etype == EnergyType::Membrane) {
            result.template topLeftCorner<9, 9>()
                = HLE::template hessian(mat.getPsi(), FBGetter(plate.de), elementData, /* projectionDisabled = */ !membraneProjection, (weight * mat.getThickness()));
        }

        // Bending energy contribution
        if (etype == EnergyType::Full || etype == EnergyType::Bending)
            plate.accumulateHessian(result, weight, /* projectionMask  = */ false);

        if constexpr (SetLowerTri)
            result.template triangularView<Eigen::Lower>() = result.transpose();

        return result;
    }

    EData elementData;
    PBE plate;
};

#endif /* end of include guard: SHELLELEMENT_HH */
