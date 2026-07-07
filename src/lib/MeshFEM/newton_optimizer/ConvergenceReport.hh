#ifndef CONVERGENCE_REPORT_HH
#define CONVERGENCE_REPORT_HH
#include <map>
#include <string>

namespace MeshFEM {

struct ConvergenceReport {
    bool success = false;
    bool backtracking_failure = false;

    // Entries for iterations 0..numIters inclusive (numIters + 1 entries in total)
    std::vector<Real> energy,
                      freeGradientNorm, // norm of "free components" of gradient
                      stepLength;       // step length chosen by this iteration's line search (only numIters meaningful entries; last is duplicated)
    std::vector<bool> indefinite;       // whether the Hessian is indefinite                  (only numIters meaningful entries; last is duplicated)
    std::vector<bool> hessianProjected; // whether the Hessian was projected to be positive definite (only numIters meaningful entries; last is duplicated)
    std::vector<std::map<std::string, Real>> customData;

    void addEntry(Real e, Real gfn, Real alpha, bool indef, bool proj) {
        energy.push_back(e);
        freeGradientNorm.push_back(gfn);
        stepLength.push_back(alpha);
        indefinite.push_back(indef);
        hessianProjected.push_back(proj);
    }
    void addCustomData(const std::map<std::string, Real> &data) {
        customData.push_back(data);
    }

    size_t numIters() const { return energy.size() ? energy.size() - 1 : 0; }
    void printEntry(size_t entry = std::numeric_limits<size_t>::max()) const {
        entry = std::min(entry, numIters());
        if (entry < energy.size()) {
            std::cout << energy[entry]
                      << '\t' << freeGradientNorm[entry]
                      << '\t' << stepLength[entry] << '\t' << indefinite[entry]
                      << '\t' << hessianProjected[entry]
                      << '\n';
        }
    }
};


} // namespace MeshFEM

#endif /* end of include guard: CONVERGENCE_REPORT_HH */
