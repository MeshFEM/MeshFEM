#include "Functions.hh"
#include "GaussQuadrature.hh"
#include <iostream>
#include <iomanip>

using namespace Simplex;
using namespace Degree;
using namespace std;

double randDouble() {
    return double(random()) / 2147483647.0;
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    cout << std::setprecision(16);
    // Interpolant<Real, 2, 1> f(vector<Real>({1.0, 2.0, 3.0}));
    Interpolant<Real, Triangle, Linear> f(0.5, 2.0, 3.0);
    cout << f(1/3., 1/3., 1/3.) << endl;

    VectorND<3> center(1/3., 1/3., 1/3.);
    cout << f(center) << endl;;

    Interpolant<Real, Triangle, Constant> fConst(1.0);
    cout << fConst() << endl;
    cout << fConst(1.0, 2.0, 3.0) << endl;

    VectorND<2> vals(1/3., 1/3.);
    Interpolant<Real, Edge, Linear> efLin(0.5, 1.0);
    Interpolant<Real, Edge, Linear> efLin2(vals);
    cout << efLin(0.5, 0.5) << endl;
    cout << efLin2(0.5, 0.5) << endl;
    cout << efLin(0.0, 1.0) << endl;
    cout << efLin(1, 0) << endl;

    // cout << efLin.integrate() << endl;

    Interpolant<Real, Triangle, Quadratic> fQuad(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    cout << fQuad.integrate() << endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    cout << fQuadTet.integrate() << endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet2(fQuadTet);
    fQuadTet2 *= 2;
    fQuadTet2 += fQuadTet2;
    cout << fQuadTet2.integrate() << endl;
    cout << (2 * (fQuadTet + fQuadTet)).integrate() << endl;

    Interpolant<Real, Triangle, Quadratic> fPromote(f);
    cout << "linear integral: " << f.integrate() << endl;
    cout << "quadratic integral: " << fPromote.integrate() << endl;

    Interpolant<Real, Triangle, Quadratic> fPromoteFromConst(fConst);
    cout << "quadratic integral of const: " << fPromoteFromConst.integrate() << endl;
    cout << "integral of sum: " << (fPromoteFromConst + f + 1).integrate() << endl;

    Interpolant<Real, Tetrahedron, Constant> fConstTet(1.0);
    fQuadTet2 = 99;
    cout << "fQuadTet2 = 99, integral: " << (fQuadTet2 + fConstTet).integrate() << endl;


    size_t numPass = 0, numTests = 0;
    auto assert_eq = [&](size_t ln, double a, double b) -> bool {
        double err = std::abs(a - b) / std::abs(b);
        bool eq = err < 1e-9; // Interpolation can have cancellation error, so be lenient on it.
        if (!eq) cout <<  "Line " << ln <<  " ERROR: " << err << ", ABS ERROR: " << std::abs(a - b) << endl;
        else ++numPass;
        ++numTests;
        return eq;
    };

    assert_eq(__LINE__, integrate_tet<0>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_tet<1>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_tet<2>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(__LINE__, integrate_tri<0>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_tri<1>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_tri<2>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(__LINE__, integrate_edge<0>([] (Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_edge<1>([] (Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_edge<2>([] (Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(__LINE__, integrate_edge<0>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_edge<1>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);
    assert_eq(__LINE__, integrate_edge<2>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);

    assert_eq(__LINE__, Quadrature<Simplex::Edge, 0>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);
    assert_eq(__LINE__, Quadrature<Simplex::Edge, 1>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);
    assert_eq(__LINE__, Quadrature<Simplex::Edge, 2>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);

    assert_eq(__LINE__, Quadrature<Simplex::Edge, 0>::integrate([] (Real, Real) { return 1.0; }), 1.0);
    assert_eq(__LINE__, Quadrature<Simplex::Edge, 1>::integrate([] (Real, Real) { return 1.0; }), 1.0);
    assert_eq(__LINE__, Quadrature<Simplex::Edge, 2>::integrate([] (Real, Real) { return 1.0; }), 1.0);

    auto ecfi = Interpolation<Simplex::Edge, 0>::interpolant([] (Real, Real) { return 1.0; });
    assert_eq(__LINE__, ecfi.integrate(), 1.0);
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
        
        // Compare versions of interpolation
        double l0 = randDouble();
        VectorND<2> edgeSample(l0, 1 - l0);
        assert_eq(__LINE__, efl(edgeSample), efl(edgeSample[0], edgeSample[1]));
        assert_eq(__LINE__, efa(edgeSample), efa(edgeSample[0], edgeSample[1]));

        l0 = randDouble(); double l1 = randDouble();
        VectorND<3> triSample(l0, l1, 1 - (l0 + l1));
        assert_eq(__LINE__, tfl(triSample), tfl(triSample[0], triSample[1], triSample[2]));
        assert_eq(__LINE__, tfa(triSample), tfa(triSample[0], triSample[1], triSample[2]));

        l0 = randDouble(); l1 = randDouble(); double l2 = randDouble();
        VectorND<4> tetSample(l0, l1, l2, 1 - (l0 + l1 + l2));
        assert_eq(__LINE__, tetfl(tetSample), tetfl(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
        assert_eq(__LINE__, tetfa(tetSample), tetfa(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));

        // Compare versions of integration
        assert_eq(__LINE__, integrate_edge<1>([&] (Real a, Real b) { return efl(a, b); }), efl.integrate());
        assert_eq(__LINE__, integrate_edge<2>([&] (Real a, Real b) { return efa(a, b); }), efa.integrate());

        assert_eq(__LINE__, integrate_tri<1>([&] (Real a, Real b, Real c) { return tfl(a, b, c); }), tfl.integrate());
        assert_eq(__LINE__, integrate_tri<2>([&] (Real a, Real b, Real c) { return tfa(a, b, c); }), tfa.integrate());

        assert_eq(__LINE__, integrate_tet<1>([&] (Real a, Real b, Real c, Real d) { return tetfl(a, b, c, d); }), tetfl.integrate());
        assert_eq(__LINE__, integrate_tet<2>([&] (Real a, Real b, Real c, Real d) { return tetfa(a, b, c, d); }), tetfa.integrate());


        // Test expression interpolants
        Interpolant<Real, Edge,    Constant> efc(randDouble());
        auto edgeExpr = [&] (Real a, Real b) { return efc(a, b) + efl(a, b) + efa(a, b); };
        assert_eq(__LINE__, edgeExpr(edgeSample[0], edgeSample[1]), Interpolation<Edge, Quadratic>::interpolant(edgeExpr)(edgeSample));

        Interpolant<Real, Triangle,    Constant> tfc(randDouble());
        auto triExpr = [&] (Real a, Real b, Real c) { return tfc(a, b, c) + tfl(a, b, c) + tfa(a, b, c); };
        auto triExprInterp = Interpolation<Triangle, Quadratic>::interpolant(triExpr);
        if (!assert_eq(__LINE__, triExpr(triSample[0], triSample[1], triSample[2]), triExprInterp(triSample))) {
            cout << tfc << tfl << tfa << triExprInterp;
            cout << "sample at:\t" << triSample[0] << ", " << triSample[1] << ", " << triSample[2] << endl;
            cout << "true val: " << triExpr(triSample[0], triSample[1], triSample[2]) << endl;
            cout << "interp val: " << triExprInterp(triSample) << endl;
        }

        Interpolant<Real, Tetrahedron, Constant> tetfc(randDouble());
        auto tetExpr = [&] (Real a, Real b, Real c, Real d) { return tetfc(a, b, c, d) + tetfl(a, b, c, d) + tetfa(a, b, c, d); };
        assert_eq(__LINE__, tetExpr(tetSample[0], tetSample[1], tetSample[2], tetSample[3]), Interpolation<Tetrahedron, Quadratic>::interpolant(tetExpr)(tetSample));
    }

    cout << numPass << " / " << numTests << " passed" << endl;

    return 0;
}
