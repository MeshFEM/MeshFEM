////////////////////////////////////////////////////////////////////////////////
// GaussQuadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Gaussian quadrature rules for edges, triangles, and tetrahedra for
//      degrees up to 4.
//
//      These routines work both on functions with K + 1 Real parameters (where
//      K + 1 is the number of nodes of the K simplex) and functions with a
//      single EvalPt parameter.
//
//      SFINAE is used to "overload" the integration routines to work in both of
//      these cases.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/10/2014 17:13:25
////////////////////////////////////////////////////////////////////////////////
#ifndef GAUSSQUADRATURE_HH
#define GAUSSQUADRATURE_HH
#include <MeshFEM/Types.hh>
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/Functions.hh>
#include <array>

#include <MeshFEM_export.h>

template<size_t _K, size_t _Deg>
struct MESHFEM_EXPORT QuadratureTable {
    static constexpr size_t numPoints = 0;
    inline static constexpr std::array<EvalPt<_K>, numPoints> points{};
    inline static constexpr std::array<double,     numPoints> weights{};
};

// Edge function (1D)
// 1 point quadrature for const and linear, 2 point for quadratic and cubic, 3 for quartic and quintic
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 2) && (_Deg <= 5), int>::type = 0>
typename function_traits<F>::result_type integrate_edge(const F &f, Real vol = 1.0) {
    if constexpr (_Deg <= 1) { return vol * f(0.5, 0.5); }
    if constexpr ((_Deg == 2) || (_Deg == 3)) {
        constexpr double c0 = 0.78867513459481288225; // (3 + sqrt(3)) / 6
        constexpr double c1 = 0.21132486540518711775; // (3 - sqrt(3)) / 6
        typename function_traits<F>::result_type result(f(c0, c1));
        result += f(c1, c0);
        result *= vol / 2.0;
        return result;
    }
    if constexpr ((_Deg == 4) || (_Deg == 5)) {
        constexpr double c0 = 0.11270166537925831148; // (1 - sqrt(3/5)) / 2
        constexpr double c1 = 0.88729833462074168852; // (1 + sqrt(3/5)) / 2
        typename function_traits<F>::result_type result(f(c0, c1));
        result += f(c1, c0);
        result *= 5.0 / 18.0;
        result += (4.0 / 9.0) * f(0.5, 0.5);
        result *= vol;
        return result;
    }
    assert(false);
}

template<size_t _K, size_t _Deg>
using QPArray = std::array<EvalPt<_K>, QuadratureTable<_K, _Deg>::numPoints>;

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 0> {
    static constexpr size_t numPoints = 1;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0.5, 0.5}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 1.0 }};
};

// Linear rule is the same as constant
template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 1> : public QuadratureTable<Simplex::Edge, 0> { };

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 2> {
    static constexpr size_t numPoints = 2;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0.78867513459481288225, 0.21132486540518711775}},
        {{0.21132486540518711775, 0.78867513459481288225}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 0.5, 0.5 }};
};

// Cubic rule is the same as quadratic
template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 3> : public QuadratureTable<Simplex::Edge, 2> { };

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 4> {
    static constexpr size_t numPoints = 3;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0.11270166537925831148, 0.88729833462074168852}},
        {{0.88729833462074168852, 0.11270166537925831148}},
        {{0.5, 0.5}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 5/18., 5/18., 4/9. }};
};

// Degree 5 rule is the same as degree 4
template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Edge, 5> : public QuadratureTable<Simplex::Edge, 4> { };

template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_edge(const F &f, Real vol = 1.0) {
    return integrate_edge<_Deg>([&](Real p0, Real p1) { return f(EvalPt<1>{{p0, p1}}); }, vol); }

// Triangle function (2D)
// 1 point quadrature for const and linear, 3 for quadratic, 4 for cubic, and 6
// for quartic
// For efficiency, a negative weight rule is used for cubic
// integrals (the nonnegative weight rule would use 6 points)
// This means that the rule should not be used for stiffness matrix construction
// to avoid ruining positive semidefiniteness (This is only a problem for FEM
// degree 3+, which is not currently implemented)
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 3) && (_Deg <= 5), int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    if constexpr (_Deg <= 1) { return vol * f(1 / 3.0, 1 / 3.0, 1 / 3.0); }
    if constexpr (_Deg == 2) {
        constexpr double c0 = 2 / 3.0;
        constexpr double c1 = 1 / 6.0;
#if 0
        typename function_traits<F>::result_type result(f(c0, c1, c1));
        result += f(c1, c0, c1);
        result += f(c1, c1, c0);
        result *= vol / 3.0;
        return result;
#else // This version seems faster...
        return (vol / 3.0) * (f(c0, c1, c1) + f(c1, c0, c1) + f(c1, c1, c0));
#endif
    }
    if constexpr (_Deg == 3) {
#if 0 // This rule has nice symmetry but negative weights...
        constexpr double c0 = 3 / 5.0;
        constexpr double c1 = 1 / 5.0;
        typename function_traits<F>::result_type result(f(c0, c1, c1));
        result += f(c1, c0, c1);
        result += f(c1, c1, c0);
        result *= (25.0 / 48);
        result += (-9.0 / 16) * f(1 / 3.0, 1 / 3.0, 1 / 3.0); // NEGATIVE WEIGHT
#else
        // The following 4-pt rule with *positive weights* is originally from
        // [Hillion 1977: Numerical Integration on a Triangle]. It's the same
        // rule used in PolyFEM, which in turn got the rule from `quadpy`.
        constexpr double c0 = 0.178558728263616461884311092944699339568614959716796875;
        constexpr double c1 = 0.155051025721682111946364557297783903777599334716796875;
        constexpr double c2 = 0.66639024601470142616932434975751675665378570556640625; // 1 - (c0 + c1)
        constexpr double c3 = 0.0750311102226081383381739442484104074537754058837890625;
        constexpr double c4 = 0.644948974278317876951405196450650691986083984375;
        constexpr double c5 = 0.28001991549907401246599647492985241115093231201171875; // 1 - (c3 + c4)
        typename function_traits<F>::result_type result(f(c2, c0, c1));
        result += f(c0, c2, c1);
        result *= 0.31804138174397722504949115318595431745052337646484375;
        result += 0.181958618256022719439357615556218661367893218994140625 * (f(c5, c3, c4) + f(c3, c5, c4));
#endif
        result *= vol;
        return result;
    }
    if constexpr (_Deg == 4) {
        // The analytic expressions of these weights are complicated...
        // See Derivations/TriangleGaussFelippa.nb
        // (From the Mathematica code in:
        // http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch24.d/IFEM.Ch24.pdf )
        constexpr double w_0 =  0.22338158967801146570;
        constexpr double c0_0 = 0.10810301816807022736;
        constexpr double c1_0 = 0.44594849091596488632;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0);
        tmp *= w_0;

        constexpr double w_1 =  0.10995174365532186764;
        constexpr double c0_1 = 0.81684757298045851308;
        constexpr double c1_1 = 0.09157621350977074346;
        typename function_traits<F>::result_type result(f(c0_1, c1_1, c1_1));
        result += f(c1_1, c0_1, c1_1);
        result += f(c1_1, c1_1, c0_1);
        result *= w_1;

        result += tmp;
        result *= vol;
        return result;
    }
    if constexpr (_Deg == 5) {
        // The analytic expressions of these weights are complicated...
        // See Derivations/TriangleGaussFelippa.nb
        // (From the Mathematica code in:
        // http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch24.d/IFEM.Ch24.pdf )
        constexpr double w_0 =  0.12593918054482715260;
        constexpr double c0_0 = 0.79742698535308732240;
        constexpr double c1_0 = 0.10128650732345633880;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0);
        tmp *= w_0;

        constexpr double w_1 =  0.13239415278850618074;
        constexpr double c0_1 = 0.059715871789769820459;
        constexpr double c1_1 = 0.47014206410511508977;
        typename function_traits<F>::result_type result(f(c0_1, c1_1, c1_1));
        result += f(c1_1, c0_1, c1_1);
        result += f(c1_1, c1_1, c0_1);
        result *= w_1;

        result += tmp;
        result += (9.0 / 40) * f(1.0 / 3, 1.0 / 3, 1.0 / 3);

        result *= vol;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    return integrate_tri<_Deg>([&](Real p0, Real p1, Real p2) { return f(EvalPt<2>{{p0, p1, p2}}); }, vol);
}

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 0> {
    static constexpr size_t numPoints = 1;
    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 1.0 }};
};

// Linear rule is the same as constant
template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 1> : public QuadratureTable<Simplex::Triangle, 0> { };

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 2> {
    static constexpr size_t numPoints = 3;

    static constexpr double c0 = 2 / 3.0, c1 = 1 / 6.0;
    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{c0, c1, c1}},
        {{c1, c0, c1}},
        {{c1, c1, c0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 1/3., 1/3., 1/3. }};
};

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 3> {
    static constexpr size_t numPoints = 4;
#if 0 // Old negative-weight rule.
    static constexpr double c0 = 3 / 5.0,
                            c1 = 1 / 5.0;
    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{c0, c1, c1}},
        {{c1, c0, c1}},
        {{c1, c1, c0}},
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 25/48., 25/48.,  25/48.,  -9.0/16 }};
#endif
    // See discussion of origin above.
    static constexpr double c0 = 0.178558728263616461884311092944699339568614959716796875;
    static constexpr double c1 = 0.155051025721682111946364557297783903777599334716796875;
    static constexpr double c2 = 0.66639024601470142616932434975751675665378570556640625; // 1 - (c0 + c1)
    static constexpr double c3 = 0.0750311102226081383381739442484104074537754058837890625;
    static constexpr double c4 = 0.644948974278317876951405196450650691986083984375;
    static constexpr double c5 = 0.28001991549907401246599647492985241115093231201171875; // 1 - (c3 + c4)
    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{c2, c0, c1}},
        {{c0, c2, c1}},
        {{c5, c3, c4}},
        {{c3, c5, c4}}
    }};

    static constexpr double w0 = 0.31804138174397722504949115318595431745052337646484375;
    static constexpr double w1 = 0.181958618256022719439357615556218661367893218994140625;

    inline static constexpr std::array<double, numPoints> weights{{ w0, w0, w1, w1 }};
};

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 4> {
    static constexpr size_t numPoints = 6;

    static constexpr double c0_0 = 0.10810301816807022736,
                            c1_0 = 0.44594849091596488632,
                            c0_1 = 0.81684757298045851308,
                            c1_1 = 0.09157621350977074346;

    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{c0_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c0_0}},
        {{c0_1, c1_1, c1_1}},
        {{c1_1, c0_1, c1_1}},
        {{c1_1, c1_1, c0_1}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{
        0.22338158967801146570, 0.22338158967801146570, 0.22338158967801146570,
        0.10995174365532186764, 0.10995174365532186764, 0.10995174365532186764
    }};
};

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Triangle, 5> {
    static constexpr size_t numPoints = 7;

    static constexpr double c0_0 = 0.79742698535308732240,
                            c1_0 = 0.10128650732345633880,
                            c0_1 = 0.059715871789769820459,
                            c1_1 = 0.47014206410511508977;

    inline static constexpr std::array<EvalPt<Simplex::Triangle>, numPoints> points{{
        {{c0_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c0_0}},
        {{c0_1, c1_1, c1_1}},
        {{c1_1, c0_1, c1_1}},
        {{c1_1, c1_1, c0_1}},
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{
        0.12593918054482715260, 0.12593918054482715260, 0.12593918054482715260,
        0.13239415278850618074, 0.13239415278850618074, 0.13239415278850618074,
        9/40.
    }};
};

// Tet function (3D)
// 1 point quadrature for const and linear, 4 point for quadratic, 5 for cubic,
// and 11 for quartic.
// For efficiency, negative weight rules are used for cubic and quartic
// integrals (the nonnegative weight rules use 8 and 16 points respectively).
// This means that those rules should not be used for stiffness matrix
// construction to avoid ruining positive semidefiniteness (This is only a
// problem for FEM degree 3+, which is not currently implemented)
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 4) && (_Deg <= 4), int>::type = 0>
typename function_traits<F>::result_type integrate_tet(const F &f, Real vol = 1.0) {
    if constexpr (_Deg <= 1) { return vol * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); }
    if constexpr (_Deg == 2) {
        constexpr double c0 = 0.58541019662496845446; // (5 + 3 sqrt(5)) / 20
        constexpr double c1 = 0.13819660112501051518; // (5 -   sqrt(5)) / 20
#if 0
        typename function_traits<F>::result_type result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= vol / 4;
        return result;
#else // This version seems faster...
        return (0.25 * vol) * (f(c0, c1, c1, c1) + f(c1, c0, c1, c1) + f(c1, c1, c0, c1) + f(c1, c1, c1, c0));
#endif
    }
    if constexpr (_Deg == 3) {
#if 0 // The following nice symmetric 5-pt rule has negative weights...
        // http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
        constexpr double c0 = 0.5;
        constexpr double c1 = 1 / 6.0;

        typename function_traits<F>::result_type result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= 0.45;
        result += (-0.8) * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); // NEGATIVE WEIGHT
        result *= vol;
        return result;
#else
        return vol * (0.12232200275734507466385281304610543884336948394775390625 * f(0.6414297914956963442278947695740498602390289306640625, 0.16200149169852445796280449030746240168809890747070313,    0.1838503504920977471570608940965030342340469360351563,      0.01271836631368145065223984602198470383882522583007813 )
                    + 0.12806641271074692411957585136406123638153076171875       * f(0.345444155719730736087136335754621541127562522888    , 0.01090521221118924410919959200327866710722446441650390625, 0.2815238021235462184677089680917561054229736328125,         0.362126829945533801335955104150343686342239379882813   )
                    + 0.1325680271444452384965728697352460585534572601318359375  * f(0.439858947649275040109317913561426394153386354446411 , 0.190117002439283921955137657278100959956645965576171875,   0.011403329444557169097818061231919273268431425094604492188, 0.358620720466883868837726367928553372621536254882813   )
                    + 0.14062440966040323786501176073215901851654052734375       * f(0.0378716317823570014500234037768677808344364166259766, 0.1708169251649890030275713570517837069928646087646484375,  0.1528181430909273386120617033157031983137130737304688,      0.63849329996172665691034353585564531385898590087890625 )
                    + 0.224415166917557418191364604354021139442920684814453125   * f(0.12480486216524716569509223518252838402986526489258  , 0.158685163227440584332583739524125121533870697021484375,   0.585662805655215779054856284346897155046463012695313,       0.130847168952096470917467740946449339389801025390625   )
                    + 0.252003980809502314830439217985258437693119049072265625   * f(0.1414827519695045499048546844278462231159210205078125, 0.5712260521491151488149284887185785919427871704101563,     0.146918390087169586921689301561855245381593704223632813,    0.140372805794210714358527525291719939559698104858398438));
#endif
    }
    if constexpr (_Deg == 4) {
#if 0
        // This rule is from
        // http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
        // but the weights there are off by a factor of 6!
        typename function_traits<F>::result_type result(f(0.25, 0.25, 0.25, 0.25));
        result *= -148.0 / 1875.0; // NEGATIVE WEIGHT

        constexpr double c0_0 = 11.0 / 14.0;
        constexpr double c1_0 =  1.0 / 14.0;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c1_0, c0_0);
        tmp *= 343.0 / 7500.0;
        result += tmp;

        constexpr double c0_1 = 0.39940357616679920500; // (14 + sqrt(70)) / 56
        constexpr double c1_1 = 0.10059642383320079500; // (14 - sqrt(70)) / 56
        tmp  = f(c0_1, c0_1, c1_1, c1_1);
        tmp += f(c0_1, c1_1, c0_1, c1_1);
        tmp += f(c0_1, c1_1, c1_1, c0_1);
        tmp += f(c1_1, c0_1, c0_1, c1_1);
        tmp += f(c1_1, c0_1, c1_1, c0_1);
        tmp += f(c1_1, c1_1, c0_1, c0_1);
        tmp *= 56.0 / 375.0;
        result += tmp;
        result *= vol;
        return result;
#else
        // Asymmetric 11-pt positive-weight rule used in PolyFEM
        return vol * (0.03925109092483995698596999091023462824523448944091796875  * f(0.1746940586972305468893562618859505164436995983123779,   0.04049050672759042790449512949635391123592853546142578125, 0.0135607018798028812478495552795720868743956089019775390625, 0.7712547326953761439582990533381234854459762573242188   )
                    + 0.055273369155936898089453990223773871548473834991455078125 * f(0.0814049184028592387463163504435215145349502563476562,   0.7525085070096549921814244044071529060602188110351563,     0.06809937093820665754417831294631469063460826873779296875,   0.0979872036492791115280809322030108887702226638793945313)
                    + 0.055393798871576367670588325609060120768845081329345703125 * f(0.741228882093622601368032576374389464035630226135253906, 0.0672232948933833979188179341690556611865758895874023438,  0.0351839297735987155402170856177690438926219940185546875,    0.156363893239395285172932403838785830885171890258789063 )
                    + 0.05993318514655952833347640762440278194844722747802734375  * f(0.053341239535745252342557876090722857043147087097168,    0.41926631387951301954686300632602069526910781860351563,    0.04778143555908666295639619647772633470594882965087890625,   0.4796110110256550651541829211055301129817962646484375   )
                    + 0.06946996593763536675947278808962437324225902557373046875  * f(0.4329534904813556739355817626346834003925323486328125,   0.4507658760912768292072883014043327420949935913085938,     0.05945661629943382875396196141082327812910079956054688,      0.056824017127933668103167974550160579383373260498046875 )
                    + 0.07616271524555835725767138910669018514454364776611328125  * f(0.5380072039161857555544798970004194416105747222900,      0.129411373788910405435714778832334559410810470581054688,   0.3301904148374644742958139431721065193414688110351563,       0.00239100745743936471399138099513947963714599609375     )
                    + 0.0794266800680253071131886599687277339398860931396484375   * f(0.00899126009333582609794888185206218622624874114990,     0.1215419913339278040753654863692645449191331863403320313,  0.306493988429690278341155362795689143240451812744140625,     0.56297276014304609148553026898298412561416625976563     )
                    + 0.10646803415549009608209729549344046972692012786865234375  * f(0.10660417256199361535351499696844257414340972900391,     0.0972046445875832665350912975554820150136947631835938,     0.684390415453040024118536166497506201267242431640625,        0.11180076739738309399285753897856920957565307617188     )
                    + 0.11023423242849765546491624945701914839446544647216796875  * f(0.32923295974264682461907227661868091672658920288086,     0.02956949520647961238140055684198159724473953247070313,    0.31790356021339460923513797752093523740768432617188,         0.323293984837478953764389189018402248620986938476563    )
                    + 0.1549761160162460849054610889652394689619541168212890625   * f(0.1038441164109931147407905882573686540126800537109,      0.43271023904776856339182700139645021408796310424804688,    0.35382323920929709126781403938366565853357315063476563,      0.10962240533194123059956837096251547336578369140625     )
                    + 0.193410812049634450726642853624070994555950164794921875    * f(0.30444840243449691752353203355596633628010749816894531,  0.240276664928072619664689568708126898854970932006835938,   0.126801725915392016208471659410861320793628692626953125,     0.32847320672203844660330673832504544407129287719726563  )) ;
#endif
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tet(const F &f, Real vol = 1.0) {
    return integrate_tet<_Deg>([&](Real p0, Real p1, Real p2, Real p3) { return f(EvalPt<3>{{p0, p1, p2, p3}}); }, vol);
}

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Tetrahedron, 0> {
    static constexpr size_t numPoints = 1;
    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 1.0 }};

};

// Linear rule is the same as constant
template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Tetrahedron, 1> : public QuadratureTable<Simplex::Tetrahedron, 0> { };

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Tetrahedron, 2> {
    static constexpr size_t numPoints = 4;
    static constexpr double c0 = 0.58541019662496845446, // (5 + 3 sqrt(5)) / 20
                            c1 = 0.13819660112501051518; // (5 -   sqrt(5)) / 20
    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{c0, c1, c1, c1}},
        {{c1, c0, c1, c1}},
        {{c1, c1, c0, c1}},
        {{c1, c1, c1, c0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 0.25, 0.25, 0.25, 0.25 }};
};

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Tetrahedron, 3> {
#if 0 // Symmetric egative weight rule
    static constexpr size_t numPoints = 5;
    static constexpr double c0 = 0.5,
                            c1 = 1 / 6.0;
    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{c0, c1, c1, c1}},
        {{c1, c0, c1, c1}},
        {{c1, c1, c0, c1}},
        {{c1, c1, c1, c0}},
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{
        0.45, 0.45, 0.45, 0.45,
        -0.8
    }};
#else // Asymmetric 6-pt positive-weight rule used in PolyFEM
    static constexpr size_t numPoints = 6;
    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{ 0.6414297914956963442278947695740498602390289306640625, 0.16200149169852445796280449030746240168809890747070313,    0.1838503504920977471570608940965030342340469360351563,      0.01271836631368145065223984602198470383882522583007813  }},
        {{ 0.345444155719730736087136335754621541127562522888,     0.01090521221118924410919959200327866710722446441650390625, 0.2815238021235462184677089680917561054229736328125,         0.362126829945533801335955104150343686342239379882813    }},
        {{ 0.439858947649275040109317913561426394153386354446411,  0.190117002439283921955137657278100959956645965576171875,   0.011403329444557169097818061231919273268431425094604492188, 0.358620720466883868837726367928553372621536254882813    }},
        {{ 0.0378716317823570014500234037768677808344364166259766, 0.1708169251649890030275713570517837069928646087646484375,  0.1528181430909273386120617033157031983137130737304688,      0.63849329996172665691034353585564531385898590087890625  }},
        {{ 0.12480486216524716569509223518252838402986526489258,   0.158685163227440584332583739524125121533870697021484375,   0.585662805655215779054856284346897155046463012695313,       0.130847168952096470917467740946449339389801025390625    }},
        {{ 0.1414827519695045499048546844278462231159210205078125, 0.5712260521491151488149284887185785919427871704101563,     0.146918390087169586921689301561855245381593704223632813,    0.140372805794210714358527525291719939559698104858398438 }}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 0.12232200275734507466385281304610543884336948394775390625, 0.12806641271074692411957585136406123638153076171875, 0.1325680271444452384965728697352460585534572601318359375, 0.14062440966040323786501176073215901851654052734375, 0.224415166917557418191364604354021139442920684814453125, 0.252003980809502314830439217985258437693119049072265625 }};
#endif
};

template<>
struct MESHFEM_EXPORT QuadratureTable<Simplex::Tetrahedron, 4> {
#if 0 // Symmetric negative weight rule
    static constexpr size_t numPoints = 11;

    static constexpr double c0_0 = 11.0 / 14.0,
                            c1_0 = 1.0 / 14.0,
                            c0_1 = 0.39940357616679920500, // (14 + sqrt(70)) / 56
                            c1_1 = 0.10059642383320079500; // (14 - sqrt(70)) / 56

    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}},
        {{c0_0, c1_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0, c1_0}},
        {{c1_0, c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c1_0, c0_0}},
        {{c0_1, c0_1, c1_1, c1_1}},
        {{c0_1, c1_1, c0_1, c1_1}},
        {{c0_1, c1_1, c1_1, c0_1}},
        {{c1_1, c0_1, c0_1, c1_1}},
        {{c1_1, c0_1, c1_1, c0_1}},
        {{c1_1, c1_1, c0_1, c0_1}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{
        -148/1875.,
        343 / 7500., 343 / 7500., 343 / 7500., 343 / 7500.,
        56 / 375., 56 / 375., 56 / 375., 56 / 375., 56 / 375., 56 / 375.
    }};
#else // Asymmetric 11-pt positive-weight rule used in PolyFEM
    static constexpr size_t numPoints = 11;
    inline static constexpr std::array<EvalPt<Simplex::Tetrahedron>, numPoints> points{{
        {{0.1746940586972305468893562618859505164436995983123779,   0.04049050672759042790449512949635391123592853546142578125, 0.0135607018798028812478495552795720868743956089019775390625, 0.7712547326953761439582990533381234854459762573242188    }},
        {{0.0814049184028592387463163504435215145349502563476562,   0.7525085070096549921814244044071529060602188110351563,     0.06809937093820665754417831294631469063460826873779296875,   0.0979872036492791115280809322030108887702226638793945313 }},
        {{0.741228882093622601368032576374389464035630226135253906, 0.0672232948933833979188179341690556611865758895874023438,  0.0351839297735987155402170856177690438926219940185546875,    0.156363893239395285172932403838785830885171890258789063  }},
        {{0.053341239535745252342557876090722857043147087097168,    0.41926631387951301954686300632602069526910781860351563,    0.04778143555908666295639619647772633470594882965087890625,   0.4796110110256550651541829211055301129817962646484375    }},
        {{0.4329534904813556739355817626346834003925323486328125,   0.4507658760912768292072883014043327420949935913085938,     0.05945661629943382875396196141082327812910079956054688,      0.056824017127933668103167974550160579383373260498046875  }},
        {{0.5380072039161857555544798970004194416105747222900,      0.129411373788910405435714778832334559410810470581054688,   0.3301904148374644742958139431721065193414688110351563,       0.00239100745743936471399138099513947963714599609375      }},
        {{0.00899126009333582609794888185206218622624874114990,     0.1215419913339278040753654863692645449191331863403320313,  0.306493988429690278341155362795689143240451812744140625,     0.56297276014304609148553026898298412561416625976563      }},
        {{0.10660417256199361535351499696844257414340972900391,     0.0972046445875832665350912975554820150136947631835938,     0.684390415453040024118536166497506201267242431640625,        0.11180076739738309399285753897856920957565307617188      }},
        {{0.32923295974264682461907227661868091672658920288086,     0.02956949520647961238140055684198159724473953247070313,    0.31790356021339460923513797752093523740768432617188,         0.323293984837478953764389189018402248620986938476563     }},
        {{0.1038441164109931147407905882573686540126800537109,      0.43271023904776856339182700139645021408796310424804688,    0.35382323920929709126781403938366565853357315063476563,      0.10962240533194123059956837096251547336578369140625      }},
        {{0.30444840243449691752353203355596633628010749816894531,  0.240276664928072619664689568708126898854970932006835938,   0.126801725915392016208471659410861320793628692626953125,     0.32847320672203844660330673832504544407129287719726563   }},
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 0.03925109092483995698596999091023462824523448944091796875, 0.055273369155936898089453990223773871548473834991455078125, 0.055393798871576367670588325609060120768845081329345703125, 0.05993318514655952833347640762440278194844722747802734375, 0.06946996593763536675947278808962437324225902557373046875, 0.07616271524555835725767138910669018514454364776611328125, 0.0794266800680253071131886599687277339398860931396484375, 0.10646803415549009608209729549344046972692012786865234375, 0.11023423242849765546491624945701914839446544647216796875, 0.1549761160162460849054610889652394689619541168212890625, 0.193410812049634450726642853624070994555950164794921875 }};
#endif
};

// Integration on a _K simplex (runs the implementations above).
// Usage:
// Quadrature<Simplex::{Edge,Triangle,Tetrahedron}, Degree>::integrate(f);
template<size_t _K, size_t _Deg>
struct Quadrature { };

template<size_t _K, size_t _Deg>
struct TableBasedQuadrature : public QuadratureTable<_K, _Deg> {
    using QT = QuadratureTable<_K, _Deg>;
    template<class F>
    static void foreach(const F &f, Real vol = 1.0) {
        for (size_t i = 0; i < QT::numPoints; ++i)
            f(QT::points[i], QT::weights[i] * vol);
    }
};

template<size_t _Deg> struct Quadrature<Simplex::Edge,        _Deg> : public TableBasedQuadrature<Simplex::Edge,        _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_edge<_Deg>(f)) { return integrate_edge<_Deg>(f, vol); } };
template<size_t _Deg> struct Quadrature<Simplex::Triangle,    _Deg> : public TableBasedQuadrature<Simplex::Triangle,    _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_tri <_Deg>(f)) { return integrate_tri< _Deg>(f, vol); } };
template<size_t _Deg> struct Quadrature<Simplex::Tetrahedron, _Deg> : public TableBasedQuadrature<Simplex::Tetrahedron, _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_tet <_Deg>(f)) { return integrate_tet< _Deg>(f, vol); } };

// Convenience function for calculating the integral of the FEM shape functions
// over an element of volume 1.
template<size_t _Deg, size_t _K>
Eigen::Matrix<Real, Simplex::numNodes(_K, _Deg), 1>
integratedShapeFunctions() {
    return Quadrature<_K, _Deg>::integrate([](const EvalPt<_K> &x) { return shapeFunctions<_Deg, _K>(x); });
}

#endif /* end of include guard: GAUSSQUADRATURE_HH */
