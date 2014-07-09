////////////////////////////////////////////////////////////////////////////////
// MaterialOptimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/09/2014 01:34:28
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIALOPTIMIZATION_HH
#define MATERIALOPTIMIZATION_HH

#include "LinearElasticity.hh"
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

namespace MaterialOptimization {

template<size_t _N>
struct IsotropicMaterial {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 2;
    typedef ElasticityTensor<Real, _N> ETensor;

    IsotropicMaterial() { vars[0] = 1.0; vars[1] = 0; }

    void getTensorDerivative(size_t p, ETensor &d) const {
        assert(p == 0 || p == 1);
        assert(false); // TODO: IMPLEMENT
    }

    void getTensor(ETensor &tensor) const {
        tensor.setIsotropic(vars[0], vars[1]);
    }

    Real vars[numVars];
};

// Per-element material field
template<class _Material>
class MaterialField {
public:
    typedef typename _Material::ETensor ETensor;

    MaterialField(size_t numElements,
            const std::vector<size_t> &matIdxForElement = std::vector<size_t>())
    {
        size_t numMat;
        if (matIdxForElement.size() == numElements) {
            m_matIdxForElement = matIdxForElement;
            size_t m = *(std::max_element(matIdxForElement.begin(),
                                          matIdxForElement.end()));
            numMat = m + 1;
            if (numMat > numElements) std::cout << "WARNING: more materials than elements." << std::endl;
            m_elementsForMatIdx.assign(numMat, std::vector<size_t>());
            for (size_t i = 0; i < numElements; ++i) {
                m_elementsForMatIdx[matIdxForElement[i]].push_back(i);
            }
            for (size_t i = 0; i < numMat; ++i) {
                if (m_elementsForMatIdx[i].size() == 0) {
                    std::cout << "WARNING: Material " << i
                              << " unreferenced." << std::endl;
                }
            }
        }
        else {
            // By default, create one material per element
            numMat = numElements;
            assert(matIdxForElement.size() == 0);
            m_matIdxForElement.resize(numElements);
            m_elementsForMatIdx.assign(numMat, std::vector<size_t>(1));
            for (size_t i = 0; i < numElements; ++i) {
                m_matIdxForElement[i] = i;
                m_elementsForMatIdx[i][0] = i;
            }
        }

        m_materials.resize(numMat);
    }

    void getInfluenceRegion(size_t var, std::vector<size_t> &region) const {
        size_t matIdx, param;
        m_variableRole(var, matIdx, param);
        region = m_elementsForMatIdx[matIdx];
    }

    size_t   domainSize() const { return m_matIdxForElement.size(); }
    size_t numMaterials() const { return m_materials.size(); }
    size_t      numVars() const { return _Material::numVars * numMaterials(); }

    template<typename ValueVector>
    void setVars(const ValueVector &vals) {
        for (size_t i = 0; i < numVars(); ++i) {
            size_t matIdx, param;
            m_variableRole(i, matIdx, param);
            m_materials[matIdx].vars[param] = vals[i];
        }
    }

    template<typename ValueVector>
    void getVars(ValueVector &vals) const {
        for (size_t i = 0; i < numVars(); ++i) {
            size_t matIdx, param;
            m_variableRole(i, matIdx, param);
            vals[i] = m_materials[matIdx].vars[param];
        }
    }

    ETensor getElasticityTensor(size_t mi) const {
        assert(mi < m_materials.size());
        ETensor result;
        m_materials[mi].getTensor(result);
        return result;
    }

    // For use in tet/tri Data
    struct MaterialGetter {
        MaterialGetter() : m_field(NULL), m_mat(0) { }
        MaterialGetter(const MaterialField *fld, size_t mat) : m_field(fld), m_mat(mat) { }
        ETensor operator()() const { return m_field->getElasticityTensor(m_mat); }
    private:
        const MaterialField *m_field;
        size_t m_mat;
    };

    MaterialGetter getterForElement(size_t ei) const {
        assert(ei < domainSize());
        return MaterialGetter(this, m_matIdxForElement[ei]);
    }

private:
    std::vector<_Material>            m_materials;
    std::vector<size_t>               m_matIdxForElement;
    std::vector<std::vector<size_t> > m_elementsForMatIdx;

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the role of a variable in the material optimization. This role
    //  comprises the material the variable affects and the parameter it
    //  controls within that material. Currently we only support simple
    //  variables that directly control a single parameter of a single material.
    //  @param[in]  var    variable to query
    //  @param[out] matIdx Index of the material controlled by var
    //  @param[out] param  Parameter of material matIdx controlled by var
    *///////////////////////////////////////////////////////////////////////////
    void m_variableRole(size_t var, size_t &matIdx, size_t &param) const {
        assert(var < numVars());
        matIdx = var / _Material::numVars;
        param  = var % _Material::numVars;
        assert(matIdx < numMaterials());
    }
};

template<class _Material, class _Mesh>
class SimulatorND : public LinearElasticity::SimulatorND<_Mesh>
{
    typedef LinearElasticity::SimulatorND<_Mesh> Base;
    using Base::m_mesh;
public:
    static constexpr size_t N = _Material::N;
    typedef MaterialField<_Material> MField;
    typedef typename Base::VField VField;

    template<typename Elems, typename Vertices>
    SimulatorND(const Elems &elems, const Vertices &vertices,
                std::shared_ptr<const MField> mfield)
        : Base(elems, vertices) {
        attachMaterialField(mfield);
    }

    // Configures each mesh element to its material from mfield.
    // Simulator must obtain (share) ownership of the material field since it
    // may be accessed at any point in Simulator's lifetime.
    void attachMaterialField(std::shared_ptr<const MField> mfield) {
        m_matField = mfield;
        for (size_t i = 0; i < m_mesh.numElements(); ++i)
            m_mesh.element(i)->configure(mfield->getterForElement(i));
    }

    VField solveAdjoint(const VField &u) const {
        // Compute load on the DoFs caused by the adjoint problem's Neuman
        // traction:
        //  u - u_target
        // This traction is defined per-vertex and linearly interpolated over
        // each boundary element, so we can't use the inherited constant
        // per-boundary-element traction load computation. Instead, the load is
        // computed by applying the mass matrix.
        VField dofLoad(m_mesh.numVertices());
        for (size_t bei = 0; bei < m_mesh.numBoundaryElements(); ++bei) {
            auto be = m_mesh.boundaryElement(bei);
            // 2D boundary elements have 2 nodes, 3D have 3 nodes.
            for (size_t j = 0; j < N; ++j) {
                auto dist_j = u(be.vertex(j).volumeVertex().index()) -
                                be.vertex(j)->targetDisplacement();
                for (size_t i = 0; i < N; ++i) {
                    dofLoad(Base::DoF(be.vertex(i).volumeVertex().index())) +=
                                      be->massMatrixContribution(i, j) * dist_j;
                }
            }
        }

        // Adjoint problem looks just like the elastostatic problem, but with
        // the load as computed above.
        return solve(dofLoad);
    }
private:
    std::shared_ptr<const MField> m_matField;
};

template<class _Simulator>
class Optimizer {
public:
    typedef typename _Simulator::VField  VField;
    typedef typename _Simulator::SMatrix SMatrix;
    typedef typename _Simulator::ETensor ETensor;

    template<typename Elems, typename Vertices>
    Optimizer(Elems inElems, Vertices inVertices,
              std::shared_ptr<typename _Simulator::MField> matField)
        : m_sim(inElems, inVertices, matField), m_matField(matField)
    {
    }

    std::vector<Real> objectiveGradient(const VField &u) const {
        auto lambda = m_sim.solveAdjoint(u);
        std::vector<Real> g(m_matField.numVars(), 0);
        std::vector<int> elems;
        for (size_t var = 0; var < m_matField.numVars(); ++var) {
            m_matField.getInfluenceRegion(var, elems);
            ETensor dE;
            m_matField.getDerivative(var, dE);
            for (size_t ei = 0; ei < elems.size(); ++ei) {
                auto e = m_sim.mesh().element(elems[ei]);
                SMatrix e_lambda, e_u;
                m_sim.elementStrain(ei,      u,      e_u);
                m_sim.elementStrain(ei, lambda, e_lambda);
                g[var] += e->volume() * (dE.doubleContract(e_u).doubleContract(e_lambda));
            }
        }

        return g;
    }

private:
    _Simulator m_sim;
    std::shared_ptr<typename _Simulator::MField> m_matField;
};

}

namespace MaterialOptimization2D {

typedef MaterialOptimization::IsotropicMaterial<2> IsotropicMaterial2D;
typedef MaterialOptimization::MaterialField<IsotropicMaterial2D> IsotropicField;

struct BoundaryNodeData : LinearElasticity2D::BoundaryNodeData {
    bool hasTarget;
    Vector2D targetDisplacement;
};

template<class _Material>
using Mesh = LinearElasticity2D::Mesh<LinearFEM2D::NodeData<Point2D>,
            LinearElasticity2D::ElementData<typename
                MaterialOptimization::MaterialField<_Material>::MaterialGetter>,
            BoundaryNodeData>;

template<class _Material>
using Simulator = MaterialOptimization::SimulatorND<_Material, Mesh<_Material> >;

}

#endif /* end of include guard: MATERIALOPTIMIZATION_HH */
