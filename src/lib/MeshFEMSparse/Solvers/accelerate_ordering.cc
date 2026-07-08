#include "accelerate_ordering.hh"
#include "AccelerateFactorizer.hh"

namespace MeshFEM {

VecX_T<int> compute_accelerate_ordering(const SuiteSparseMatrix &A, AccelerateSparseOrder order) {
    AccelerateFactorizer af;
    af.orderingMethod = (order == AccelerateSparseOrder::AMD) ? AccelerateFactorizer::OrderingMethod::AMD : AccelerateFactorizer::OrderingMethod::Metis;
    af.storeOrdering = true;
    af.factorizeSymbolic(A, std::vector<size_t>());
    VecX_T<int> perm = af.getPermutation();
    if (perm.size() != A.m) throw std::runtime_error("Accelerate returned permutation of wrong size");
    return perm;
}

} // namespace MeshFEM
