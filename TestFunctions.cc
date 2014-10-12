#include "Functions.hh"
#include "GaussQuadrature.hh"
#include <iostream>

using namespace Simplex;
using namespace Degree;
using namespace std;

double randDouble() {
    return double(random()) / numeric_limits<long>::max();
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
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
    auto assert_eq = [&](double a, double b) -> bool {
        bool eq = abs(a - b) < 1e-25;
        if (!eq) cout <<  "TEST ERROR: " << abs(a - b) << endl;
        else ++numPass;
        ++numTests;
        return !eq;
    };

    assert_eq(integrate_tet<0>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_tet<1>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_tet<2>([] (Real, Real, Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(integrate_tri<0>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_tri<1>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_tri<2>([] (Real, Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(integrate_edge<0>([] (Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_edge<1>([] (Real, Real) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_edge<2>([] (Real, Real) { return 1.0; }, 1.0), 1.0);

    assert_eq(integrate_edge<0>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_edge<1>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);
    assert_eq(integrate_edge<2>([] (const Point2D &p) { return 1.0; }, 1.0), 1.0);

    assert_eq(Quadrature<Simplex::Edge, 0>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);
    assert_eq(Quadrature<Simplex::Edge, 1>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);
    assert_eq(Quadrature<Simplex::Edge, 2>::integrate([] (const Point2D &p) { return 1.0; }), 1.0);

    assert_eq(Quadrature<Simplex::Edge, 0>::integrate([] (Real, Real) { return 1.0; }), 1.0);
    assert_eq(Quadrature<Simplex::Edge, 1>::integrate([] (Real, Real) { return 1.0; }), 1.0);
    assert_eq(Quadrature<Simplex::Edge, 2>::integrate([] (Real, Real) { return 1.0; }), 1.0);

    auto ecfi = Interpolation<Simplex::Edge, 0>::interpolant([] (Real, Real) { return 1.0; });
    assert_eq(ecfi.integrate(), 1.0);
    // Compare versions of interpolation
    size_t runs = 10;
    size_t fails = 0, tests = runs * 6;
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
        
        VectorND<2> edgeSample(randDouble(), randDouble());
        assert_eq(efl(edgeSample), efl(edgeSample[0], edgeSample[1]));
        assert_eq(efa(edgeSample), efa(edgeSample[0], edgeSample[1]));

        VectorND<3> triSample(randDouble(), randDouble(), randDouble());
        assert_eq(tfl(triSample), tfl(triSample[0], triSample[1], triSample[2]));
        assert_eq(tfa(triSample), tfa(triSample[0], triSample[1], triSample[2]));

        VectorND<4> tetSample(randDouble(), randDouble(), randDouble(), randDouble());
        assert_eq(tetfl(tetSample), tetfl(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
        assert_eq(tetfa(tetSample), tetfa(tetSample[0], tetSample[1], tetSample[2], tetSample[3]));
    }

    cout << numPass << " / " << numTests << " passed" << endl;

    return 0;
}
