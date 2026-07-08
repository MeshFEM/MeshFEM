#include "pardiso_ordering.hh"
#include "PardisoFactorizer.hh"

namespace MeshFEM {

VecX_T<int> compute_pardiso_ordering(const SuiteSparseMatrix &A, PardisoSparseOrder order) {
    PardisoFactorizer pf;

    if      (order == PardisoSparseOrder::AMD)           pf.orderingMethod = PardisoFactorizer::OrderingMethod::AMD;
    else if (order == PardisoSparseOrder::Metis)         pf.orderingMethod = PardisoFactorizer::OrderingMethod::Metis;
    else if (order == PardisoSparseOrder::ParallelMetis) pf.orderingMethod = PardisoFactorizer::OrderingMethod::ParallelMetis;
    else throw std::runtime_error("Unexpected ordering method");

    pf.storeOrdering = true;
    pf.factorizeSymbolic(A, std::vector<size_t>());
    VecX_T<int> perm = pf.getPermutation();
    if (perm.size() != A.m) throw std::runtime_error("Accelerate returned permutation of wrong size");
    return perm;
}

} // namespace MeshFEM
