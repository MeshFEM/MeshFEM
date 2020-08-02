#ifndef CONVERGENCE_REPORT_HH
#define CONVERGENCE_REPORT_HH
#include <map>
#include <string>

struct ConvergenceReport {
    bool success = false;
    bool backtracking_failure = false;

    // Entries for iterations 0..numIters inclusive (numIters + 1 entries in total)
    std::vector<Real> energy, gradientNorm,
                      freeGradientNorm, // norm of "free components" of gradient
                      stepLength;       // step length chosen by this iteration's line search (only numIters meaningful entries; last is duplicated)
    std::vector<bool> indefinite;       // whether the Hessian is indefinite                  (only numIters meaningful entries; last is duplicated)
    std::vector<std::map<std::string, Real>> customData;

    void addEntry(Real e, Real gn, Real gfn, Real alpha, bool indef) {
        energy.push_back(e);
        gradientNorm.push_back(gn);
        freeGradientNorm.push_back(gfn);
        stepLength.push_back(alpha);
        indefinite.push_back(indef);
    }
    void addCustomData(const std::map<std::string, Real> &data) {
        customData.push_back(data);
    }

    size_t numIters() const { return energy.size() ? energy.size() - 1 : 0; }
    void printEntry(size_t entry = std::numeric_limits<size_t>::max()) const {
        entry = std::min(entry, numIters());
        if (entry < energy.size()) {
            std::cout << energy[entry]
                      << '\t' << gradientNorm[entry]
                      << '\t' << freeGradientNorm[entry]
                      << '\t' << stepLength[entry] << '\t' << indefinite[entry]
                      << std::endl;
        }
    }
};

#endif /* end of include guard: CONVERGENCE_REPORT_HH */
