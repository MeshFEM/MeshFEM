////////////////////////////////////////////////////////////////////////////////
// dense_newton.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A simple Newton optimizer for small, dense problems using a brute-force
//  Hessian regularization based on an Eigendecomposition.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/17/2020 12:50:37
////////////////////////////////////////////////////////////////////////////////
#ifndef DENSE_NEWTON_HH
#define DENSE_NEWTON_HH

template<class Problem>
void dense_newton(Problem &prob, size_t maxIter = 100, double gradTol = 1e-14, bool verbose=false) {
    using HType = typename Problem::HessType;
    using VType = typename Problem::VarType;
    using Real_  = typename Problem::Real;

    if (verbose) std::cout.precision(19);

    auto printReport = [](size_t it, Real e, const VType &g, bool indefinite) {
        std::cout << it
                  << '\t' << e
                  << '\t' << g.norm()
                  << '\t' << indefinite
                  << std::endl;
    };

    bool isIndefinite = false;
    size_t it;
    for (it = 0; it <= maxIter; ++it) {
        auto vars = prob.getVars();
        auto g = prob.gradient();

        // Robust eigendecomposition-based Hessian modification + solver.
        auto H_es = Eigen::SelfAdjointEigenSolver<HType>(prob.hessian());
        auto Hinv_lambda = H_es.eigenvalues();
        // std::cout << "Hessian: " << prob.hessian() << std::endl;
        // std::cout << "Hessian eigenvalues: " << H_es.eigenvalues().transpose() << std::endl;
        for (int i = 0; i < Hinv_lambda.rows(); ++i) {
            if (Hinv_lambda[i] < 0) {
                isIndefinite = true;
                Hinv_lambda[i] *= -1;
            }
            if (Hinv_lambda[i] > 1e-10) // Avoid division by zero in nearly singular case
                Hinv_lambda[i] = 1.0 / Hinv_lambda[i];
        }

        if ((!isIndefinite) && (g.norm() < gradTol)) break;
        const Real_ currEnergy = prob.energy();

        if (verbose) printReport(it, currEnergy, g, isIndefinite);

        auto step = H_es.eigenvectors() * (Hinv_lambda.asDiagonal() * (H_es.eigenvectors().transpose() * -g));
        Real_ directionalDerivative = g.dot(step);

        Real_ alpha = 1.0;
        const Real_ c_1 = 1e-4;
        const size_t nbacktrack_iter = 15;
        size_t bit;
        Eigen::VectorXd steppedVars;
        for (bit = 0; bit < nbacktrack_iter; ++bit) {
            steppedVars = vars + alpha * step;
            prob.setVars(steppedVars);
            Real_ steppedEnergy = prob.energy();

            const Real_ sufficientDecrease = -c_1 * alpha * directionalDerivative;
            const Real_ decrease = currEnergy - steppedEnergy;
            // Terminate line search successfully if a sufficient decrease is achieved
            // (or if we cannot expect to evaluate the energy decrease accurately
            // enough to measure a sufficient decrease--and the energy does not
            // increase significantly)
            if  ((decrease >= sufficientDecrease)
                    || (std::abs(sufficientDecrease) < 1e-10 * std::abs(currEnergy)
                            && (decrease > -1e-16 * std::abs(currEnergy)))) {
                break;
            }
            alpha *= 0.5;
        }

        if (bit == nbacktrack_iter) {
            if (verbose) std::cout << "Backtracking failed\n";
            prob.setVars(vars);
            break;
        }
    }

    if (verbose) printReport(it + 1, prob.energy(), prob.gradient(), isIndefinite);
}

#endif /* end of include guard: DENSE_NEWTON_HH */
