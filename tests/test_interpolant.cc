#include <MeshFEM/Functions.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>

using namespace Simplex;
using namespace Degree;
using namespace std;

static double randDouble() {
    return ((double)rand() / (double)RAND_MAX);
}

template<size_t K, class F, typename... Args>
typename std::enable_if<(sizeof...(Args) == K + 1), double>::type
static evalAt(const VectorND<K + 1> &/* samplePt */, const F&f, Args&&... args) {
    return f(args...);
}

template<size_t K, class F, typename... Args>
typename std::enable_if<(sizeof...(Args) < K + 1), double>::type
static evalAt(const VectorND<K + 1> &samplePt, const F&f, Args&&... args) {
    return evalAt<K>(samplePt, f, args..., samplePt[sizeof...(Args)]);
}

// Test that all functions in "funcs" up to degree Deg are interpolated exactly by a Deg-degree interpolant.
// Also, ensure that the integrate() methods obtain the same result as Gauss quadrature.
template<size_t K, size_t Deg, typename F>
static void interpolant_test(const vector<vector<F>> &funcs) {
    for (size_t d = 0; d <= Deg; ++d) {
        for (const auto &f : funcs.at(d)) {
            auto interp = Interpolation<K, Deg>::interpolant(f);
            // check interpolation by randomly sampling at many points.
            size_t samples = 20000;
            for (size_t i = 0; i < samples; ++i) {
                VectorND<K + 1> samplePt;
                // Don't bother staying within the simplex to test
                // interpolation--the polynomials should match on all of R^K.
                for (size_t c = 0; c < K; ++c)
                    samplePt[c] = randDouble();
                samplePt[K] = 0;
                samplePt[K] = 1 - samplePt.sum();
                double diff = evalAt<K>(samplePt, interp) - evalAt<K>(samplePt, f);
                if (std::abs(diff) > 1e-13) {
                    std::cout << "interp sample:"   << evalAt<K>(samplePt, interp) << std::endl;
                    std::cout << "function sample:" << evalAt<K>(samplePt,      f) << std::endl;
                    std::cout << "sample pt:" << samplePt.transpose() << std::endl;
                    std::cout << "Interpolation error of " + std::to_string(diff) + " in <"
                            + std::to_string(K) + ", " + std::to_string(Deg) + "> interpolant of deg "
                            + std::to_string(d) + " function." << std::endl;
                }
                REQUIRE(std::abs(diff) <= 1e-13);
            }
            double diff = interp.integrate(1.0) - Quadrature<K, Deg>::integrate(f, 1.0);
            if (std::abs(diff) > 1e-16) {
                std::cout << "Quadrature error of " + std::to_string(diff) + " in <"
                        + std::to_string(K) + ", " + std::to_string(Deg) + "> interpolant of deg "
                        + std::to_string(d) + " function." << std::endl;
            }
            REQUIRE(std::abs(diff) <= 1e-16);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

// The monomials systematically tested below generally don't include all
// variables; silence the resulting warnings on GCC and Clang.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

TEST_CASE("interpolant functions", "[quadrature]" ) {

    cout << std::setprecision(16);
    // Interpolant<Real, 2, 1> f(vector<Real>({1.0, 2.0, 3.0}));
    Interpolant<Real, Triangle, Linear> f(0.5, 2.0, 3.0);
    cout << f(1/3., 1/3., 1/3.) << endl;

    EvalPt<2> center{{1/3., 1/3., 1/3.}};
    cout << f(center) << endl;

    Interpolant<Real, Triangle, Constant> fConst(1.0);
    cout << fConst() << endl;
    cout << fConst(1.0, 2.0, 3.0) << endl;

    std::array<Real, 2> vals = {{1/3., 1/3.}};
    Interpolant<Real, Edge, Linear> efLin(0.5, 1.0);
    Interpolant<Real, Edge, Linear> efLin2(vals);
    cout << efLin(0.5, 0.5) << endl;
    cout << efLin2(0.5, 0.5) << endl;
    cout << efLin(0.0, 1.0) << endl;
    cout << efLin(1, 0) << endl;

    // cout << efLin.integrate(1.0) << endl;

    Interpolant<Real, Triangle, Quadratic> fQuad(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    cout << fQuad.integrate(1.0) << endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    cout << fQuadTet.integrate(1.0) << endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet2(fQuadTet);
    fQuadTet2 *= 2;
    fQuadTet2 += fQuadTet2;
    cout << fQuadTet2.integrate(1.0) << endl;
    cout << (2 * (fQuadTet + fQuadTet)).integrate(1.0) << endl;

    Interpolant<Real, Triangle, Quadratic> fPromote(f);
    cout << "linear integral: " << f.integrate(1.0) << endl;
    cout << "quadratic integral: " << fPromote.integrate(1.0) << endl;

    Interpolant<Real, Triangle, Quadratic> fPromoteFromConst(fConst);
    cout << "quadratic integral of const: " << fPromoteFromConst.integrate(1.0) << endl;
    cout << "integral of sum: " << (fPromoteFromConst + f + 1).integrate(1.0) << endl;

    Interpolant<Real, Tetrahedron, Constant> fConstTet(1.0);
    fQuadTet2 = 99;
    cout << "fQuadTet2 = 99, integral: " << (fQuadTet2 + fConstTet).integrate(1.0) << endl;

    SECTION("Multiple runs") {
      auto assert_eq = [&](size_t ln, double a, double b) -> bool {
          double err = std::abs(a - b) / std::abs(b);
          bool eq = err < 1e-9; // Interpolation can have cancellation error, so be lenient on it.
          if (!eq) {
            cout <<  "Line " << ln <<  " ERROR: " << err << ", ABS ERROR: " << std::abs(a - b) << endl;
          }
          REQUIRE(eq);
          return eq;
      };

      auto ecfi = Interpolation<Simplex::Edge, 0>::interpolant([] (Real, Real) { return 1.0; });
      assert_eq(__LINE__, ecfi.integrate(1.0), 1.0);
      size_t runs = 200000;

      for (size_t i = 0; i < runs; ++i) {
          Interpolant<Real, Edge,    Linear> efl(randDouble(), randDouble());
          Interpolant<Real, Edge, Quadratic> efa(randDouble(), randDouble(), randDouble());
          Interpolant<Real, Triangle,    Linear> tfl(randDouble(), randDouble(), randDouble());
          Interpolant<Real, Triangle, Quadratic> tfa(randDouble(), randDouble(), randDouble(),
                                                     randDouble(), randDouble(), randDouble());
          Interpolant<Real, Tetrahedron,    Linear> tetfl(randDouble(), randDouble(), randDouble(), randDouble());
          Interpolant<Real, Tetrahedron, Quadratic> tetfa(randDouble(), randDouble(), randDouble(), randDouble(),
                                                          randDouble(), randDouble(), randDouble(),
                                                          randDouble(), randDouble(), randDouble());

          double l0 = randDouble();
          VectorND<2> edgeSample(l0, 1 - l0);
          l0 = randDouble(); double l1 = randDouble();
          VectorND<3> triSample(l0, l1, 1 - (l0 + l1));
          l0 = randDouble(); l1 = randDouble(); double l2 = randDouble();
          VectorND<4> tetSample(l0, l1, l2, 1 - (l0 + l1 + l2));
  #if 0
          // Compare versions of interpolation
          assert_eq(__LINE__, efl(edgeSample), efl(edgeSample[0], edgeSample[1]));
          assert_eq(__LINE__, efa(edgeSample), efa(edgeSample[0], edgeSample[1]));

          assert_eq(__LINE__, tfl(triSample), tfl(triSample[0], triSample[1], triSample[2]));
          assert_eq(__LINE__, tfa(triSample), tfa(triSample[0], triSample[1], triSample[2]));

          assert_eq(__LINE__, tetfl(tetSample), tetfl(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
          assert_eq(__LINE__, tetfa(tetSample), tetfa(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
  #endif

          // Compare versions of integration
          assert_eq(__LINE__, integrate_edge<1>([&] (Real a, Real b) { return efl(a, b); }), efl.integrate(1.0));
          assert_eq(__LINE__, integrate_edge<2>([&] (Real a, Real b) { return efa(a, b); }), efa.integrate(1.0));
          assert_eq(__LINE__, integrate_edge<3>([&] (Real a, Real b) { return efa(a, b); }), efa.integrate(1.0));

          assert_eq(__LINE__, integrate_tri<1>([&] (Real a, Real b, Real c) { return tfl(a, b, c); }), tfl.integrate(1.0));
          assert_eq(__LINE__, integrate_tri<2>([&] (Real a, Real b, Real c) { return tfa(a, b, c); }), tfa.integrate(1.0));
          assert_eq(__LINE__, integrate_tri<3>([&] (Real a, Real b, Real c) { return tfa(a, b, c); }), tfa.integrate(1.0));

          assert_eq(__LINE__, integrate_tet<1>([&] (Real a, Real b, Real c, Real d) { return tetfl(a, b, c, d); }), tetfl.integrate(1.0));
          assert_eq(__LINE__, integrate_tet<2>([&] (Real a, Real b, Real c, Real d) { return tetfa(a, b, c, d); }), tetfa.integrate(1.0));
          assert_eq(__LINE__, integrate_tet<3>([&] (Real a, Real b, Real c, Real d) { return tetfa(a, b, c, d); }), tetfa.integrate(1.0));

          // Test expression interpolants
          Interpolant<Real, Edge,    Constant> efc(randDouble());
          auto edgeExpr = [&] (Real a, Real b) { return efc(a, b) + efl(a, b) + efa(a, b); };
          assert_eq(__LINE__, edgeExpr(edgeSample[0], edgeSample[1]), Interpolation<Edge, Quadratic>::interpolant(edgeExpr)(edgeSample[0], edgeSample[1]));

          Interpolant<Real, Triangle,    Constant> tfc(randDouble());
          auto triExpr = [&] (Real a, Real b, Real c) { return tfc(a, b, c) + tfl(a, b, c) + tfa(a, b, c); };
          auto triExprInterp = Interpolation<Triangle, Quadratic>::interpolant(triExpr);
          if (!assert_eq(__LINE__, triExpr(triSample[0], triSample[1], triSample[2]), triExprInterp(triSample[0], triSample[1], triSample[2]))) {
              cout << tfc << tfl << tfa << triExprInterp;
              cout << "sample at:\t" << triSample[0] << ", " << triSample[1] << ", " << triSample[2] << endl;
              cout << "true val: " << triExpr(triSample[0], triSample[1], triSample[2]) << endl;
              cout << "interp val: " << triExprInterp(triSample[0], triSample[1], triSample[2]) << endl;
          }

          Interpolant<Real, Tetrahedron, Constant> tetfc(randDouble());
          auto tetExpr = [&] (Real a, Real b, Real c, Real d) { return tetfc(a, b, c, d) + tetfl(a, b, c, d) + tetfa(a, b, c, d); };
          assert_eq(__LINE__, tetExpr(tetSample[0], tetSample[1], tetSample[2], tetSample[3]), Interpolation<Tetrahedron, Quadratic>::interpolant(tetExpr)(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
      }
    }

    // 1D functions up to degree 4
    vector<vector<function<Real(Real, Real)>>> functions1D =
            {{[](Real u, Real other) { return 1; }},
             {[](Real u, Real other) { return u; }},
             {[](Real u, Real other) { return u*u; }},
             {[](Real u, Real other) { return u*u*u; }},
             {[](Real u, Real other) { return u*u*u*u; }}};

    // 2D functions up to degree 5
    vector<vector<function<Real(Real, Real, Real)>>> functions2D =
        {{[](Real u, Real v, Real other) { return 1; }},
         {[](Real u, Real v, Real other) { return v; },
          [](Real u, Real v, Real other) { return u; }},
         {[](Real u, Real v, Real other) { return v*v; },
          [](Real u, Real v, Real other) { return u*v; },
          [](Real u, Real v, Real other) { return u*u; }},
         {[](Real u, Real v, Real other) { return v*v*v; },
          [](Real u, Real v, Real other) { return u*(v*v); },
          [](Real u, Real v, Real other) { return v*(u*u); },
          [](Real u, Real v, Real other) { return u*u*u; }},
         {[](Real u, Real v, Real other) { return v*v*v*v; },
          [](Real u, Real v, Real other) { return u*(v*v*v); },
          [](Real u, Real v, Real other) { return u*u*(v*v); },
          [](Real u, Real v, Real other) { return v*(u*u*u); },
          [](Real u, Real v, Real other) { return u*u*u*u; }},
         {[](Real u, Real v, Real other) { return v*v*v*v*v; },
          [](Real u, Real v, Real other) { return u*(v*v*v*v); },
          [](Real u, Real v, Real other) { return u*u*(v*v*v); },
          [](Real u, Real v, Real other) { return u*u*u*(v*v); },
          [](Real u, Real v, Real other) { return v*(u*u*u*u); },
          [](Real u, Real v, Real other) { return u*u*u*u*u; }}};

    // 3D functions up to degree 4
    vector<vector<function<Real(Real, Real, Real, Real)>>> functions3D =
        {{[](Real u, Real v, Real w, Real other) { return 1; }},
         {[](Real u, Real v, Real w, Real other) { return w; },
          [](Real u, Real v, Real w, Real other) { return v; },
          [](Real u, Real v, Real w, Real other) { return u; }},
         {[](Real u, Real v, Real w, Real other) { return w*w; },
          [](Real u, Real v, Real w, Real other) { return v*w; },
          [](Real u, Real v, Real w, Real other) { return v*v; },
          [](Real u, Real v, Real w, Real other) { return u*w; },
          [](Real u, Real v, Real w, Real other) { return u*v; },
          [](Real u, Real v, Real w, Real other) { return u*u; }},
         {[](Real u, Real v, Real w, Real other) { return w*w*w; },
          [](Real u, Real v, Real w, Real other) { return v*(w*w); },
          [](Real u, Real v, Real w, Real other) { return w*(v*v); },
          [](Real u, Real v, Real w, Real other) { return v*v*v; },
          [](Real u, Real v, Real w, Real other) { return u*(w*w); },
          [](Real u, Real v, Real w, Real other) { return u*v*w; },
          [](Real u, Real v, Real w, Real other) { return u*(v*v); },
          [](Real u, Real v, Real w, Real other) { return w*(u*u); },
          [](Real u, Real v, Real w, Real other) { return v*(u*u); },
          [](Real u, Real v, Real w, Real other) { return u*u*u; }},
         {[](Real u, Real v, Real w, Real other) { return w*w*w*w; },
          [](Real u, Real v, Real w, Real other) { return v*(w*w*w); },
          [](Real u, Real v, Real w, Real other) { return v*v*(w*w); },
          [](Real u, Real v, Real w, Real other) { return w*(v*v*v); },
          [](Real u, Real v, Real w, Real other) { return v*v*v*v; },
          [](Real u, Real v, Real w, Real other) { return u*(w*w*w); },
          [](Real u, Real v, Real w, Real other) { return u*v*(w*w); },
          [](Real u, Real v, Real w, Real other) { return u*w*(v*v); },
          [](Real u, Real v, Real w, Real other) { return u*(v*v*v); },
          [](Real u, Real v, Real w, Real other) { return u*u*(w*w); },
          [](Real u, Real v, Real w, Real other) { return v*w*(u*u); },
          [](Real u, Real v, Real w, Real other) { return u*u*(v*v); },
          [](Real u, Real v, Real w, Real other) { return w*(u*u*u); },
          [](Real u, Real v, Real w, Real other) { return v*(u*u*u); },
          [](Real u, Real v, Real w, Real other) { return u*u*u*u; }}};

    SECTION("Degree 0 tests") {
      interpolant_test<1, 0>(functions1D);
      interpolant_test<2, 0>(functions2D);
      interpolant_test<3, 0>(functions3D);
    }

    SECTION("Degree 1 tests") {
      interpolant_test<1, 1>(functions1D);
      interpolant_test<2, 1>(functions2D);
      interpolant_test<3, 1>(functions3D);
    }

    SECTION("Degree 2 tests") {
      interpolant_test<1, 2>(functions1D);
      interpolant_test<2, 2>(functions2D);
      interpolant_test<3, 2>(functions3D);
    }

    SECTION("Degree 3 tests") {
      // (only cubic triangles implemented)
      interpolant_test<2, 3>(functions2D);
    }

    SECTION("Degree 4 tests") {
      // (only quartic triangles implemented)
      // interpolant_test<1, 4>(functions1D);
      interpolant_test<2, 4>(functions2D);
      // interpolant_test<3, 4>(functions3D);
    }
}

#pragma GCC diagnostic pop
