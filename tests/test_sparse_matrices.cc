#include <MeshFEM/SparseMatrices.hh>
// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>
#include <random>

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

    // Test conversions between full/upper/lower symmetry mode.
    srandom(0);
    const size_t ntests = 100;
    for (size_t test = 0; test < ntests; ++test) {
        size_t matSize   = 1000 + 1000 * (random() % 10);
        size_t ntriplets = 6000 + 6000 * (random() % 10);
        TripletMatrix<> Ctrip(matSize, matSize);
        Ctrip.reserve(2 * ntriplets);
        for (size_t t = 0; t < ntriplets; ++t) {
            size_t i = random() % matSize;
            size_t j = random() % matSize;
            double v = random() / double(RAND_MAX);
            Ctrip.addNZ(i, j, v);
            Ctrip.addNZ(j, i, v);
        }

        SuiteSparseMatrix C(Ctrip);
        auto Cupper = C.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE);
        auto Clower = C.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::LOWER_TRIANGLE);
        REQUIRE(Cupper.nz == Clower.nz);
        REQUIRE(Cupper.nz < C.nz);

        // Ensure that we can reconstruct the full symmetric matrix from each triangle.
        {
            SuiteSparseMatrix Cdiff = C;
            Cdiff.addWithIdenticalSparsity(Cupper.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE), -1.0);
            // if (Cdiff.data().norm() != 0.0)
            //     std::cout << "deviation: " << Cdiff.data().norm() << std::endl;

            REQUIRE(Cdiff.data().norm() / C.data().norm() < 5e-16);

            Cdiff = C;
            Cdiff.addWithIdenticalSparsity(Clower.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE), -1.0);
            REQUIRE(Cdiff.data().norm() / C.data().norm() < 5e-16);
        }

        // Test the matvec implementations all agree.
        const size_t nmatvec_tests = 10;
        for (size_t tt = 0; tt < nmatvec_tests; ++tt) {
            Eigen::VectorXd v = Eigen::VectorXd::Random(matSize);
            Eigen::VectorXd Cv1 = C.apply(v),
                            Cv2 = Cupper.apply(v),
                            Cv3 = Clower.apply(v);
            REQUIRE((Cv2 - Cv1).norm() / Cv1.norm() < 5e-16);
            REQUIRE((Cv3 - Cv2).norm() / Cv1.norm() < 5e-16);
        }
    }

    // Test transpose of asymmetric matrix
    srandom(0);
    for (size_t test = 0; test < ntests; ++test) {
        size_t m   = 1000 + 1000 * (random() % 10);
        size_t n   = 1000 + 1000 * (random() % 10);
        size_t ntriplets = 6000 + 6000 * (random() % 10);
        TripletMatrix<> Ctrip(m, n);
        Ctrip.reserve(ntriplets);
        for (size_t t = 0; t < ntriplets; ++t) {
            size_t i = random() % m;
            size_t j = random() % n;
            double v = random() / double(RAND_MAX);
            Ctrip.addNZ(i, j, v);
        }

        SuiteSparseMatrix C(Ctrip);
        auto C_t = C.transpose();
        REQUIRE(C_t.nz == C.nz);

        // Test the matvec implementations all agree.
        const size_t nmatvec_tests = 10;
        for (size_t tt = 0; tt < nmatvec_tests; ++tt) {
            Eigen::VectorXd v = Eigen::VectorXd::Random(n);
            Eigen::VectorXd Cv1 = C.apply(v),
                            Cv2 = C_t.apply(v, /* transpose = */ true);
            REQUIRE((Cv2 - Cv1).norm() / Cv1.norm() < 5e-16);
        }
    }
}
