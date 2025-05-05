////////////////////////////////////////////////////////////////////////////////
// ElementBase.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  CRTP base class for elements to to be used, e.g., in a `MeshEnergy`.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/10/2023 12:12:20
*///////////////////////////////////////////////////////////////////////////////
#ifndef ELEMENTBASE_HH
#define ELEMENTBASE_HH

#include "MaterialAssignment.hh"

// Traits class must define `Material` type
template<class Derived>
struct ElementTraits { using Material = MaterialBase; };

template<class Derived>
struct ElementBase {
    using Material       = typename ElementTraits<Derived>::Material;
    using MA             = MaterialAssignment<Material>;
    using MaterialGetter = typename MA::ElementMaterialGetter;
    ElementBase(size_t ei, MA &materials) : m_materialGetter(materials, ei) { }

    const Material &material() const { return m_materialGetter.get(); }
          Material &material()       { return m_materialGetter.get(); }

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<Derived &>(*this); }
private:
    MaterialGetter m_materialGetter;
};

#endif /* end of include guard: ELEMENTBASE_HH */
