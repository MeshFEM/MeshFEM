#include "Functions.hh"
#include <iostream>

using namespace Simplex;
using namespace Degree;

double randDouble() {
    return double(random()) / std::numeric_limits<long>::max();
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    // Interpolant<Real, 2, 1> f(std::vector<Real>({1.0, 2.0, 3.0}));
    Interpolant<Real, Triangle, Linear> f(0.5, 2.0, 3.0);
    std::cout << f(1/3., 1/3., 1/3.) << std::endl;

    VectorND<3> center(1/3., 1/3., 1/3.);
    std::cout << f(center) << std::endl;;

    Interpolant<Real, Triangle, Constant> fConst(1.0);
    std::cout << fConst() << std::endl;
    std::cout << fConst(1.0, 2.0, 3.0) << std::endl;

    VectorND<2> vals(1/3., 1/3.);
    Interpolant<Real, Edge, Linear> efLin(0.5, 1.0);
    Interpolant<Real, Edge, Linear> efLin2(vals);
    std::cout << efLin(0.5, 0.5) << std::endl;
    std::cout << efLin2(0.5, 0.5) << std::endl;
    std::cout << efLin(0.0, 1.0) << std::endl;
    std::cout << efLin(1, 0) << std::endl;

    // std::cout << efLin.integrate() << std::endl;

    Interpolant<Real, Triangle, Quadratic> fQuad(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    std::cout << fQuad.integrate() << std::endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    std::cout << fQuadTet.integrate() << std::endl;

    Interpolant<Real, Tetrahedron, Quadratic> fQuadTet2(fQuadTet);
    fQuadTet2 *= 2;
    fQuadTet2 += fQuadTet2;
    std::cout << fQuadTet2.integrate() << std::endl;
    std::cout << (2 * (fQuadTet + fQuadTet)).integrate() << std::endl;

    Interpolant<Real, Triangle, Quadratic> fPromote(f);
    std::cout << "linear integral: " << f.integrate() << std::endl;
    std::cout << "quadratic integral: " << fPromote.integrate() << std::endl;

    Interpolant<Real, Triangle, Quadratic> fPromoteFromConst(fConst);
    std::cout << "quadratic integral of const: " << fPromoteFromConst.integrate() << std::endl;
    std::cout << "integral of sum: " << (fPromoteFromConst + f + 1).integrate() << std::endl;

    Expression<Real, Triangle, Quadratic> expr(fPromoteFromConst);
    std::cout << "expr: " << expr.interpolant<Linear>()(center) << std::endl;

    Interpolant<Real, Tetrahedron, Constant> fConstTet(1.0);
    fQuadTet2 = 99;
    std::cout << "fQuadTet2 = 99, integral: " << (fQuadTet2 + fConstTet).integrate() << std::endl;

    std::cout << integrate_tet<Real, 0>([] (Real, Real, Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_tet<Real, 1>([] (Real, Real, Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_tet<Real, 2>([] (Real, Real, Real, Real) { return 1.0; }, 1.0) << std::endl;

    std::cout << integrate_tri<Real, 0>([] (Real, Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_tri<Real, 1>([] (Real, Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_tri<Real, 2>([] (Real, Real, Real) { return 1.0; }, 1.0) << std::endl;

    std::cout << integrate_edge<Real, 0>([] (Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_edge<Real, 1>([] (Real, Real) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_edge<Real, 2>([] (Real, Real) { return 1.0; }, 1.0) << std::endl;

    std::cout << integrate_edge<Real, 0>([] (const Point2D &p) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_edge<Real, 1>([] (const Point2D &p) { return 1.0; }, 1.0) << std::endl;
    std::cout << integrate_edge<Real, 2>([] (const Point2D &p) { return 1.0; }, 1.0) << std::endl;


    // Compare versions of interpolation
    size_t runs = 10;
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
    }

    return 0;
}
