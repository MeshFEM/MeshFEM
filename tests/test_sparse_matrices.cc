#include <MeshFEM/SparseMatrices.hh>
// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>
#include <cstdlib>

TEST_CASE("sparse matrix format conversions", "[sparse_matrix]" ) {
    using VXd = Eigen::VectorXd;

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
    srand(0);
    const size_t ntests = 100;
    for (size_t test = 0; test < ntests; ++test) {
        size_t matSize   = 1000 + 1000 * (rand() % 10);
        size_t ntriplets = 6000 + 6000 * (rand() % 10);
        TripletMatrix<> Ctrip(matSize, matSize);
        Ctrip.reserve(2 * ntriplets);
        for (size_t t = 0; t < ntriplets; ++t) {
            size_t i = rand() % matSize;
            size_t j = rand() % matSize;
            double v = rand() / double(RAND_MAX);
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

        // Test that the matvec implementations all agree.
        const size_t nmatvec_tests = 10;
        for (size_t tt = 0; tt < nmatvec_tests; ++tt) {
            VXd v = VXd::Random(matSize);
            VXd Cv1 = C.apply(v),
                Cv2 = Cupper.apply(v),
                Cv3 = Clower.apply(v);
            REQUIRE((Cv2 - Cv1).norm() / Cv1.norm() < 5e-16);
            REQUIRE((Cv3 - Cv2).norm() / Cv1.norm() < 5e-16);
        }
    }

    // Test transpose of asymmetric matrix
    srand(0);
    for (size_t test = 0; test < ntests; ++test) {
        size_t m   = 1000 + 1000 * (rand() % 10);
        size_t n   = 1000 + 1000 * (rand() % 10);
        size_t ntriplets = 6000 + 6000 * (rand() % 10);
        TripletMatrix<> Ctrip(m, n);
        Ctrip.reserve(ntriplets);
        for (size_t t = 0; t < ntriplets; ++t) {
            size_t i = rand() % m;
            size_t j = rand() % n;
            double v = rand() / double(RAND_MAX);
            Ctrip.addNZ(i, j, v);
        }

        SuiteSparseMatrix C(Ctrip);
        auto C_t = C.transpose();
        REQUIRE(C_t.nz == C.nz);

        // Test that the matvec implementations all agree.
        const size_t nmatvec_tests = 10;
        for (size_t tt = 0; tt < nmatvec_tests; ++tt) {
            VXd v = VXd::Random(n);
            VXd Cv1 = C.apply(v),
                Cv2 = C_t.apply(v, /* transpose = */ true),
                Cv3 = C_t.applyTransposeParallel(v);
            REQUIRE((Cv2 - Cv1).norm() / Cv1.norm() < 5e-16);
            REQUIRE((Cv3 - Cv1).norm() / Cv1.norm() < 5e-16);
        }
    }

    // Test block sparse matrix matvec.
    for (size_t test = 0; test < ntests; ++test) {
        using M3d = Eigen::Matrix3d;
        size_t m   = 1000 + 1000 * (rand() % 10);
        size_t n   = 1000 + 1000 * (rand() % 10);
        size_t ntriplets = 6000 + 6000 * (rand() % 10);
        TripletMatrix<> Ctrip(3 * m, 3 * n);
        TripletMatrix<Triplet<M3d>> CtripBlock(m, n);
        Ctrip.reserve(9 * ntriplets);
        CtripBlock.reserve(ntriplets);
        for (size_t t = 0; t < ntriplets; ++t) {
            size_t i = rand() % m;
            size_t j = rand() % n;
            M3d v = M3d::Random();
            CtripBlock.addNZ(i, j, v);
            for (size_t c = 0; c < 3; ++c) {
                for (size_t d = 0; d < 3; ++d)
                    Ctrip.addNZ(3 * i + c, 3 * j + d, v(c, d));
            }
        }

        CSCMatrix<SuiteSparse_long, Real> C(Ctrip);
        CSCMatrix<SuiteSparse_long, M3d>  Cblock(CtripBlock);

        // Test that the scalar and block matvec implementations agree.
        const size_t nmatvec_tests = 10;
        for (size_t tt = 0; tt < nmatvec_tests; ++tt) {
            VXd v = VXd::Random(3 * m); // `m` since we're applying the tranpose.
            // Unflatten block vector into an `m x 3` matrix.
            Eigen::MatrixX3d V = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(v.data(), m, 3);
            // Unflattedn block vector into an m-vector of 3-vectors
            Eigen::Matrix<Eigen::Vector3d, Eigen::Dynamic, 1> v_nested(m);
            for (size_t i = 0; i < m; ++i)
                v_nested[i] = v.segment<3>(3 * i);

            static_assert(isEigenType<decltype(v.segment<3>(0))>(), "argh...");

            VXd Cv1         = C.applyTransposeParallel(v);
            VXd Cv2         = Cblock.applyTransposeParallel(v);
            auto Cv3Mat     = Cblock.applyTransposeParallel(V);
            auto Cv4_nested = Cblock.applyTransposeParallel(v_nested);

            VXd Cv3(Cv3Mat.size());
            // Flatten block vector result
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(Cv3.data(), Cv3Mat.rows(), Cv3Mat.cols()) = Cv3Mat;

            VXd Cv4(Cv4_nested.size() * 3);
            for (int i = 0; i < Cv4_nested.size(); ++i)
                Cv4.segment<3>(3 * i) = Cv4_nested[i];

            REQUIRE((Cv2 - Cv1).norm() / Cv1.norm() < 1e-15);
            REQUIRE((Cv3 - Cv1).norm() / Cv1.norm() < 1e-15);
            REQUIRE((Cv4 - Cv1).norm() / Cv1.norm() < 1e-15);
        }
    }
}
