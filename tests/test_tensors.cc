#include <catch2/catch.hpp>
#include <MeshFEM/Flattening.hh>

template<size_t _Dim>
static void test() {
    for (size_t i = 0; i < _Dim; ++i) {
        for (size_t j = i; j < _Dim; ++j) {
            size_t fi = flattenIndices<_Dim>(i, j);
            REQUIRE(unflattenIndex<_Dim>(fi).first  == i);
            REQUIRE(unflattenIndex<_Dim>(fi).second == j);
        }
    }

    for (size_t i = 0; i < flatLen(_Dim); ++i) {
        auto ufi = unflattenIndex<_Dim>(i);
        REQUIRE(flattenIndices<_Dim>(ufi.first, ufi.second) == i);
    }
}

TEST_CASE("tensors", "[tensors]" ) {
    SECTION("Dimension 2 tests") {
        test<2>();
    }
    SECTION("Dimension 3 tests") {
        test<3>();
    }
}
