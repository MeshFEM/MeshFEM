#ifndef FEASIBLESTEPLENGTHCOMPUTER_HH
#define FEASIBLESTEPLENGTHCOMPUTER_HH

#include <MeshFEM/Types.hh>

// Abstract base class for implementing custom feasible step length
// calculations (e.g., a flip-free step for parametrization).
struct FeasibleStepLengthComputer {
    using Real = double;
    using VXd = Eigen::VectorXd;
    virtual Real eval(const VXd &vars, const VXd &step) const = 0;
    virtual ~FeasibleStepLengthComputer() = default;
};

#endif /* end of include guard: FEASIBLESTEPLENGTHCOMPUTER_HH */
