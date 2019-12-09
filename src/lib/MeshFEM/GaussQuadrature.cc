#include "GaussQuadrature.hh"

// We need to provide definitions for the static constexpr `points` memebers to avoid undefined reference linker errors.
// The commented out definitions are for the rules that simply inherit from a lower degree.
   constexpr QPArray<Simplex::Edge,        0> QuadratureTable<Simplex::Edge,        0>::points;
// constexpr QPArray<Simplex::Edge,        1> QuadratureTable<Simplex::Edge,        1>::points;
   constexpr QPArray<Simplex::Edge,        2> QuadratureTable<Simplex::Edge,        2>::points;
// constexpr QPArray<Simplex::Edge,        3> QuadratureTable<Simplex::Edge,        3>::points;
   constexpr QPArray<Simplex::Edge,        4> QuadratureTable<Simplex::Edge,        4>::points;
// constexpr QPArray<Simplex::Edge,        5> QuadratureTable<Simplex::Edge,        5>::points;

   constexpr QPArray<Simplex::Triangle,    0> QuadratureTable<Simplex::Triangle,    0>::points;
// constexpr QPArray<Simplex::Triangle,    1> QuadratureTable<Simplex::Triangle,    1>::points;
   constexpr QPArray<Simplex::Triangle,    2> QuadratureTable<Simplex::Triangle,    2>::points;
   constexpr QPArray<Simplex::Triangle,    3> QuadratureTable<Simplex::Triangle,    3>::points;
   constexpr QPArray<Simplex::Triangle,    4> QuadratureTable<Simplex::Triangle,    4>::points;
   constexpr QPArray<Simplex::Triangle,    5> QuadratureTable<Simplex::Triangle,    5>::points;

   constexpr QPArray<Simplex::Tetrahedron, 0> QuadratureTable<Simplex::Tetrahedron, 0>::points;
// constexpr QPArray<Simplex::Tetrahedron, 1> QuadratureTable<Simplex::Tetrahedron, 1>::points;
   constexpr QPArray<Simplex::Tetrahedron, 2> QuadratureTable<Simplex::Tetrahedron, 2>::points;
   constexpr QPArray<Simplex::Tetrahedron, 3> QuadratureTable<Simplex::Tetrahedron, 3>::points;
   constexpr QPArray<Simplex::Tetrahedron, 4> QuadratureTable<Simplex::Tetrahedron, 4>::points;
