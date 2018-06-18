////////////////////////////////////////////////////////////////////////////////
// MaterialField.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generic parameterized per-element material field. This field comprises
//      one or more distinct materials. Each element is assigned a single one of
//      these materials at construction time (via that matIdxForElement; if no
//      matIdxForElement is provided, a new material is created for each
//      element).
//
//      Elements can share materials, meaning a single variable can affect the
//      material in multiple elements--to get which elements a variable affects
//      use the getInfluenceRegion. However, variables currently can only affect
//      a single material for simplicity.
//
//      The variables parametrizing a material field are a concatentation of
//      each individual material's variables. Currently all materials in a field
//      must have the same number of variables for simplicity, but this
//      limitation could be removed without much difficulty.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/11/2014 15:49:52
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIALFIELD_HH
#define MATERIALFIELD_HH
#include <vector>
#include <set>
#include <string>
#include <stdexcept>

#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/Fields.hh>

// Per-element material field
template<class _Material>
class MaterialField {
public:
    typedef _Material Material;
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

    // Gets dE/dvar
    // Note: assumes variable only affects a single elasticity tensor.
    void getETensorDerivative(size_t var, ETensor &dE) const {
        size_t matIdx, param;
        m_variableRole(var, matIdx, param);
        m_materials[matIdx].getETensorDerivative(param, dE);
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

    size_t materialIndexForElement(size_t ei) const { return m_matIdxForElement.at(ei); }
    const _Material &material(size_t mi) const { return m_materials.at(mi); }
          _Material &material(size_t mi)       { return m_materials.at(mi); }
    const _Material &materialForElement(size_t ei) const { return material(materialIndexForElement(ei)); }
          _Material &materialForElement(size_t ei)       { return material(materialIndexForElement(ei)); }

    ETensor getElasticityTensor(size_t mi) const {
        ETensor result;
        m_materials.at(mi).getTensor(result);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Write a per-element scalar field for each material variable. If "values"
    //  is empty (default), the data written comes from the field variable
    //  values themselves. Otherwise, the data written comes from that vector of
    //  variable values (useful for writing components of a gradient with
    //  respect to the material field variables instead of the material field
    //  variables themselves).
    //
    //  The passed prefix is added to the front of the variable names to
    //  generate the field name.
    *///////////////////////////////////////////////////////////////////////////
    void writeVariableFields(MSHFieldWriter &writer, std::string prefix,
            std::vector<Real> values = std::vector<Real>()) {
        if (values.size() == 0) {
            values.resize(numVars());
            getVars(values);
        }
        if (values.size() != numVars())
            throw std::runtime_error("Variables size mismatch");
        for (size_t vi = 0; vi < _Material::numVars; ++vi) {
            auto name = _Material::variableName(vi);
            size_t numElements = m_matIdxForElement.size();
            ScalarField<Real> varField(numElements);
            for (size_t ei = 0; ei < numElements; ++ei) {
                size_t mi = materialIndexForElement(ei);
                varField[ei] = values[_Material::numVars * mi + vi];
            }
            writer.addField(prefix + name, varField,
                            DomainType::PER_ELEMENT);
        }
    }

    // For use in tet/tri Data
    struct MaterialGetter {
        typedef MaterialField MField;
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

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the material graph adjacencies. Two materials are considered
    //  adjacent if they are assigned to neighboring elements.
    //  @param[in]  mesh   mesh datastructure for element adjacency queries
    //  @param[out] adj    set of adjacent materials for each material.
    *///////////////////////////////////////////////////////////////////////////
    template<class Mesh>
    void materialAdjacencies(const Mesh &mesh,
                             std::vector<std::set<size_t> > &adj) const {
        if (mesh.numElements() != m_matIdxForElement.size())
            throw std::runtime_error("Invalid material adjacency query mesh.");

        adj.assign(numMaterials(), std::set<size_t>());

        // Loop over all "dual eges" of the mesh--if differing materials are
        // assigned to endpoint elements then these materials are adjacent.
        for (size_t i = 0; i < mesh.numElements(); ++i) {
            auto ei = mesh.element(i);
            size_t mati = m_matIdxForElement.at(i);
            for (size_t j = 0; j < ei.numNeighbors(); ++j) {
                auto ej = ei.neighbor(j);
                // For our mesh data structure, a returned "neighbor" may not
                // actually exist--check that we actually have one
                if (!ej) continue;
                size_t matj = m_matIdxForElement.at(ej.index());
                if (mati != matj) {
                    adj.at(mati).insert(matj);
                    adj.at(matj).insert(mati);
                }
            }
        }
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

#endif /* end of include guard: MATERIALFIELD_HH */
