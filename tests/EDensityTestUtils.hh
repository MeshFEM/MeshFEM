#ifndef EDENSITYTESTUTILS_HH
#define EDENSITYTESTUTILS_HH

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>

template<class A, class B>
void requireApproxEqual(const A &a, const B &b) {
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            REQUIRE(a(i, j) == Approx(b(i, j)));
        }
    }
}

// Note: C-based energies cannot detect inverted elements and thus will
// assign them different energies! Therefore, any energy built using
// `EnergyDensityCBasedFromFBased` will disagree with the underlying
// energy density on inverted elements, and we must avoid these with
// our tests.
// Also, nearly inverted elements will suffer numerical robustness issues due
// to the F^{-1} term appearing in EnergyDensityCBasedFromFBased.
// Lowering the "5e-3" determinant threshold causes the C(F(C)) wrapper
// tests to fail to achieve the requested accuracy.
template<class EigenType>
EigenType getPositiveF() {
    EigenType F(EigenType::Random());
    auto det = [&F]() {
        if (F.rows() == F.cols()) return F.determinant();    // Volumetric case
        return std::sqrt((F.transpose() * F).determinant()); // Membrane case
    };
    while (det() < 5.0e-3) {
        F = EigenType::Random();
    }
    return F;
}

template<class FType, class Func>
void runComparisons(const Func &f) {
    constexpr size_t ntests = 10000;
    for (size_t i = 0; i < ntests; ++i) {
        FType F(getPositiveF<FType>()),
             dF(FType::Random());
        auto  C = (F.transpose() * F).eval();
        auto dC = (F.transpose() * dF + dF.transpose() * F).eval();
        f(F, dF, C, dC);
    }
}

template<class Psi_C, class Psi_F>
void compareEnergies(Psi_C &&psi_C, Psi_F &&psi_F) {
    using FType = typename std::remove_reference_t<Psi_F>::Matrix;
    runComparisons<FType>([&](const auto &F, const auto &dF, const auto &C, const auto &dC) {
        psi_F.setDeformationGradient(F);
        psi_C.setC(C);
        REQUIRE(psi_F.energy() == Approx(psi_C.energy()));

        REQUIRE(psi_F.denergy(dF) == Approx(doubleContract(psi_C.PK2Stress(), 0.5 * dC)));
        requireApproxEqual(psi_F.delta_denergy(dF), dF * psi_C.PK2Stress() + F * psi_C.delta_PK2Stress(dC));

        // Also test the VectorizedShapeFunctionJacobian version of directional derivatives.
        using VSFJ = VectorizedShapeFunctionJacobian<FType::RowsAtCompileTime, decltype(F.row(0).transpose().eval())>;
        VSFJ dF_VSFJ(0, dF.row(0));
        auto dC_VSFJ = (F.transpose() * dF_VSFJ.matrix()).eval();
        dC_VSFJ = (dC_VSFJ + dC_VSFJ.transpose()).eval();

        REQUIRE(doubleContract(psi_F.denergy(), dF_VSFJ) == Approx(doubleContract(psi_C.PK2Stress(), 0.5 * dC_VSFJ)));
        requireApproxEqual(psi_F.delta_denergy(dF_VSFJ), dF_VSFJ.matrix() * psi_C.PK2Stress() + F * psi_C.delta_PK2Stress(dC_VSFJ));
    });
}

template<class Psi_C1, class Psi_C2>
void compareCEnergies(Psi_C1 &&psi_C1, Psi_C2 &&psi_C2) {
    using FType = typename std::remove_reference_t<Psi_C1>::Matrix;
    runComparisons<FType>([&](const auto &/* F */, const auto &/* dF */, const auto &C, const auto &dC) {
        psi_C1.setC(C);
        psi_C2.setC(C);
        REQUIRE(psi_C1.energy() == Approx(psi_C2.energy()));
        requireApproxEqual(psi_C1.PK2Stress(), psi_C2.PK2Stress());
        requireApproxEqual(psi_C1.delta_PK2Stress(dC), psi_C2.delta_PK2Stress(dC));
    });
}

template<class Psi_F1, class Psi_F2>
void compareFEnergies(Psi_F1 &&psi_F1, Psi_F2 &&psi_F2) {
    using FType = typename std::remove_reference_t<Psi_F1>::Matrix;
    runComparisons<FType>([&](const auto &F, const auto &dF, const auto &/* C */, const auto &/* dC */) {
        psi_F1.setDeformationGradient(F);
        psi_F2.setDeformationGradient(F);
        REQUIRE(psi_F1.energy() == Approx(psi_F2.energy()));
        REQUIRE(psi_F1.denergy(dF) == Approx(psi_F2.denergy(dF)));
        requireApproxEqual(psi_F1.delta_denergy(dF), psi_F2.delta_denergy(dF));
    });
}

#endif /* end of include guard: EDENSITYTESTUTILS_HH */
