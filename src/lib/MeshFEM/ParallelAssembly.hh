#ifndef PARALLELASSEMBLY_HH
#define PARALLELASSEMBLY_HH

#include "SparseMatrices.hh"

#if MESHFEM_WITH_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

template<typename Real_>
struct HessianAssemblerData {
    CSCMatrix<SuiteSparse_long, Real_> H;
    bool constructed = false;
};

template<typename Real_>
using HALocalData = tbb::enumerable_thread_specific<HessianAssemblerData<Real_>>;

template<typename F, typename Real_>
struct HessianAssembler {
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    HessianAssembler(F &f, const CSCMat &H, HALocalData<Real_> &locals) : Hsp(H), m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        HessianAssemblerData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.H.zeros_like(Hsp); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data.H); }
    }

    const CSCMat &Hsp; // sparsity pattern for H
private:
    F &m_f;
    HALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
HessianAssembler<F, Real_> make_hessian_assembler(F &f, const CSCMatrix<SuiteSparse_long, Real_> &H, HALocalData<Real_> &locals) {
    return HessianAssembler<F, Real_>(f, H, locals);
}

// Assemble a Hessian in parallel
template<typename PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, CSCMatrix<SuiteSparse_long, Real_> &H, const size_t numElems) {
    HALocalData<Real_> haLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                      make_hessian_assembler(assembler, H, haLocalData));

    for (HessianAssemblerData<Real> &data : haLocalData)
        H.addWithIdenticalSparsity(data.H);
}

#else

// Fallback to serial assembly.
template<typename PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, CSCMatrix<SuiteSparse_long, Real_> &H, const size_t numElems) {
    for (size_t ei = 0; ei < numElems; ++ei)
        assembler(ei, H);
}
#endif

#endif /* end of include guard: PARALLELASSEMBLY_HH */
