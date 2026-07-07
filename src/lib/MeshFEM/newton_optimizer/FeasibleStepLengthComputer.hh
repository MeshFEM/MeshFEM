#ifndef FEASIBLESTEPLENGTHCOMPUTER_HH
#define FEASIBLESTEPLENGTHCOMPUTER_HH

#include <MeshFEMCore/Types.hh>


// Abstract base class for implementing custom feasible step length
// calculations (e.g., a flip-free step for parametrization).
namespace MeshFEM {

struct FeasibleStepLengthComputer {
    using Real = double;
    using VXd = Eigen::VectorXd;
    virtual Real eval(const VXd &vars, const VXd &step) const = 0;
    virtual ~FeasibleStepLengthComputer() = default;
};


} // namespace MeshFEM

#endif /* end of include guard: FEASIBLESTEPLENGTHCOMPUTER_HH */
