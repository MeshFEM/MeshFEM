////////////////////////////////////////////////////////////////////////////////
// Stencils.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Stencils representing the local (block) variables of an element.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/22/2023 11:55:53
*///////////////////////////////////////////////////////////////////////////////
#ifndef STENCILS_HH
#define STENCILS_HH
#include <Eigen/Dense>

namespace MeshFEM {

namespace detail {
// Recursion base case.
template<class GlobalVarStructure, size_t Idx, size_t Offset, size_t... VarSizes>
struct ExtractImpl {
    template<class BlockVars, class LocalVars, class Derived>
    static void run(const BlockVars &/* bvars */, LocalVars &/* result */, const Eigen::MatrixBase<Derived> &/* x */, const GlobalVarStructure &/* varStructure */) { }
};

template<class GlobalVarStructure, size_t Idx, size_t Offset, size_t FirstVarSize, size_t... VarSizes>
struct ExtractImpl<GlobalVarStructure, Idx, Offset, FirstVarSize, VarSizes...> {
    template<class BlockVars, class LocalVars, class Derived>
    static void run(const BlockVars &bvars, LocalVars &result, const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) {
        Eigen::Map<VecX_T<typename LocalVars::Scalar>> result_ravel(result.data(), result.size());
        result_ravel.template segment<FirstVarSize>(Offset) = x.template segment<FirstVarSize>(varStructure.offsetForBlock(bvars[Idx]));
        ExtractImpl<GlobalVarStructure, Idx + 1, Offset + FirstVarSize, VarSizes...>::run(bvars, result, x, varStructure);
    }
};
}

template<size_t... BlockDimensions>
struct StaticStencil {
    static constexpr size_t NumLocalBlockVars = sizeof...(BlockDimensions);
    using BlockVars = std::array<int, NumLocalBlockVars>;

    StaticStencil(const BlockVars &bv) : blockVars(bv) { }

    template<class LocalVars, class Derived, class GlobalVarStructure>
    void extract(LocalVars &result, const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) const {
        detail::ExtractImpl<GlobalVarStructure, 0, 0, BlockDimensions...>::run(blockVars, result, x, varStructure);
    }

    template<class LocalVars, class Derived, class GlobalVarStructure>
    LocalVars extract(const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) const {
        LocalVars result;
        extract(result, x, varStructure);
        return result;
    }

    BlockVars blockVars;

protected:
    StaticStencil() { } // Uninitialized!
};

namespace detail {

template<size_t NumVars, size_t BlockDimension, size_t... Dimensions>
struct UniformStencilImpl {
    using type = typename UniformStencilImpl<NumVars - 1, BlockDimension, BlockDimension, Dimensions...>::type;
};

template<size_t BlockDimension, size_t... Dimensions>
struct UniformStencilImpl<0, BlockDimension, Dimensions...> {
    using type = StaticStencil<Dimensions...>;
};

};

// Simpler notation for static stencils with a single block dimension.
template<size_t NumVars, size_t BlockDimension>
using UniformStencil = typename detail::UniformStencilImpl<NumVars, BlockDimension>::type;

struct TriFlapStencil : public UniformStencil<4, 3> {
    using Base = UniformStencil<4, 3>;
    template<class HalfedgeHandle, class VarForVtx>
    TriFlapStencil(const HalfedgeHandle &he, const VarForVtx &var)
        : Base({ var(he.tail().index()),
                 var(he.tip ().index()),
                 var(he.opposite().next().tip().index()),
                 var(he           .next().tip().index()) }) { }
};

template<size_t K, size_t Deg, size_t N>
struct ElementStencil : public UniformStencil<Simplex::numNodes(K, Deg), N> {
    using Base = UniformStencil<Simplex::numNodes(K, Deg), N>;

    template<class Mesh, enable_if_models_concept_t<Concepts::ElementMesh, Mesh, int> = 0>
    ElementStencil(const Mesh &m, size_t ei)
        : Base(m.template elementNodeIndices<int>(ei)) { }

    template<class Mesh, class VarForVtx>
    ElementStencil(const Mesh &m, size_t ei, const VarForVtx &var)
        : ElementStencil(m, ei)
    {
        // Remap the variable indices.
        for (size_t i = 0; i < Base::blockVars.size(); ++i)
            Base::blockVars[i] = var(Base::blockVars[i]);
    }

    // We also allow construction from an element node table stored in an Eigen matrix.
    template<class Derived>
    ElementStencil(const Eigen::MatrixBase<Derived> &E, size_t ei)
        : Base() // Uninitialized!
    {
        static_assert(std::is_integral_v<typename Derived::Scalar>,            "ElementStencil: element node table must have integral type");
        static_assert(Derived::ColsAtCompileTime == Simplex::numNodes(K, Deg), "ElementStencil: element node table has wrong number of columns");
        for (size_t i = 0; i < Base::blockVars.size(); ++i)
            Base::blockVars[i] = E(ei, i);
    }
};

template<class Stencil_>
struct StencilFactory;

template<>
struct StencilFactory<TriFlapStencil> {
    template<class Mesh, class VarForVtx>
    static std::vector<TriFlapStencil> build(const Mesh &m, const VarForVtx &var) {
        static_assert(Mesh::K == 2, "Hinge stencil only supports triangle meshes");
        std::vector<TriFlapStencil> result;
        m.visitInteriorEdges([&](const auto &he, size_t /* ei */) {
            result.emplace_back(he, var);
        });

        return result;
    }

    template<class Mesh>
    static std::vector<TriFlapStencil> build(const Mesh &m) {
        return build(m, [](int vtx) { return vtx; });
    }
};

template<size_t K, size_t Deg, size_t N>
struct StencilFactory<ElementStencil<K, Deg, N>> {
    using S = ElementStencil<K, Deg, N>;
    template<class Mesh, class VarForVtx>
    static std::vector<S> build(const Mesh &m, const VarForVtx &var) {
        std::vector<S> result;
        for (size_t ei = 0; ei < m.numElements(); ++ei)
            result.emplace_back(m, ei, var);
        return result;
    }

    template<class Mesh>
    static std::vector<S> build(const Mesh &m) {
        std::vector<S> result;
        for (size_t ei = 0; ei < m.numElements(); ++ei)
            result.emplace_back(m, ei);
        return result;
    }
};

template<class Stencil_>
struct StencilCollection {
    using S  = Stencil_;
    using SF = StencilFactory<S>;
    template<class Mesh>
    StencilCollection(const Mesh &m) { build(m); }

    template<class Mesh>
    void build(const Mesh &m) { m_stencils = SF::build(m); }

    const S &operator[](size_t i) const { return m_stencils[i]; }
    size_t size() const { return m_stencils.size(); }

private:
    std::vector<S> m_stencils;
};

} // namespace MeshFEM

#endif /* end of include guard: STENCILS_HH */
