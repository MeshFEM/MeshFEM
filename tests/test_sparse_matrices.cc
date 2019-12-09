#include <MeshFEM/SparseMatrices.hh>
// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

TEST_CASE("sparse matrix format conversions", "[sparse_matrix]" ) {
    TripletMatrix<> A(5, 5);
    A.nz = {{1, 1, 1.5}, {3, 3, 2.5}, {3, 4, 3.5}};

    SuiteSparseMatrix ssMat(A);

    auto B = ssMat.getTripletMatrix();

    REQUIRE(A.nnz() == B.nnz());
    REQUIRE(A.m == B.m);
    REQUIRE(A.n == B.n);

    for (auto &t : B.nz) A.addNZ(t.i, t.j, -t.v);

    A.sumRepeated();
    REQUIRE(A.nnz() == 0); // A - B should be exactly zero
}
