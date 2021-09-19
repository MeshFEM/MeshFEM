////////////////////////////////////////////////////////////////////////////////
// BindingInstantiations.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Collects in one place all Mesh binding generation (so corresponding
//  bindings of functions taking these objects are also generated).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/08/2020 19:04:12
////////////////////////////////////////////////////////////////////////////////
#ifndef BINDING_INSTANTIATIONS_HH
#define BINDING_INSTANTIATIONS_HH

#include <utility>

#include <MeshFEM/FEMMesh.hh>

#include <pybind11/pybind11.h>
namespace py = pybind11;

template<class MeshBinder>
void generateMeshSpecificBindings(py::module &m, py::module &detail_module, MeshBinder &&b) {
    using V3d = Eigen::Matrix<double, 3, 1>;
    using V2d = Eigen::Matrix<double, 2, 1>;
    b.template bind<FEMMesh<3, 1, V3d>>(m, detail_module); // linear    tet mesh in 3d
    b.template bind<FEMMesh<3, 2, V3d>>(m, detail_module); // quadratic tet mesh in 3d

    b.template bind<FEMMesh<2, 1, V2d>>(m, detail_module); // linear    tri mesh in 2d
    b.template bind<FEMMesh<2, 2, V2d>>(m, detail_module); // quadratic tri mesh in 2d
    b.template bind<FEMMesh<2, 1, V3d>>(m, detail_module); // linear    tri mesh in 3d
    b.template bind<FEMMesh<2, 2, V3d>>(m, detail_module); // quadratic tri mesh in 3d

#if MESHFEM_BIND_LONG_DOUBLE
    using V3ld = Eigen::Matrix<long double, 3, 1>;
    using V2ld = Eigen::Matrix<long double, 2, 1>;

    b.template bind<FEMMesh<3, 1, V3ld>>(m, detail_module); // linear    tet mesh in 3d
    b.template bind<FEMMesh<3, 2, V3ld>>(m, detail_module); // quadratic tet mesh in 3d

    b.template bind<FEMMesh<2, 1, V2ld>>(m, detail_module); // linear    tri mesh in 2d
    b.template bind<FEMMesh<2, 2, V2ld>>(m, detail_module); // quadratic tri mesh in 2d
    b.template bind<FEMMesh<2, 1, V3ld>>(m, detail_module); // linear    tri mesh in 3d
    b.template bind<FEMMesh<2, 2, V3ld>>(m, detail_module); // quadratic tri mesh in 3d
#endif
}

#endif /* end of include guard: BINDING_INSTANTIATIONS_HH */
