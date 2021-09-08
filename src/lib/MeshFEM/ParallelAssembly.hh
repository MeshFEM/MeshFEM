#ifndef PARALLELASSEMBLY_HH
#define PARALLELASSEMBLY_HH

#include "SparseMatrices.hh"
#include "Eigen/Dense"
template<typename Real_> using VecX_T = Eigen::Matrix<Real_, -1, 1>;

#include <MeshFEM/Parallelism.hh>

// Support custom thread-local data.
struct CustomThreadLocalData {
    void construct() { } // called once for each thread's copy.
};

struct CTLDEmpty : public CustomThreadLocalData { };

#if MESHFEM_WITH_TBB
// Energy summation

template<typename F, typename Real_>
struct SummandEvaluator {
    SummandEvaluator(F& f, const size_t nvars, VecX_T<Real_>& summands) : m_f(f), m_nvars(nvars), m_summands(summands) { }

    void operator()(const tbb::blocked_range<size_t>& r) const {
        for (size_t i = r.begin(); i < r.end(); ++i) { m_summands[i] = m_f(i); }
    }
private:
    F& m_f;
    size_t m_nvars;
    VecX_T<Real_>& m_summands;
};

template<typename F, typename Real_>
SummandEvaluator<F, Real_> make_summand_evaluator(F& f, size_t nvars, VecX_T<Real_>& summands) {
    return SummandEvaluator<F, Real_>(f, nvars, summands);
}

template<typename Real_, typename PerElemSummand>
Real_ summation_parallel(const PerElemSummand& summand, const size_t numElems) {
    VecX_T<Real_> summands(numElems);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
        make_summand_evaluator(summand, numElems, summands));

    return summands.sum();
}

// Dense vector/matrix assembly (e.g., for gradient)

template<class DenseMatrixType>
struct DenseAssemblerData {
    DenseMatrixType A;
    bool needs_reset = true;
};

template<class DenseMatrixType>
using DALocalData = tbb::enumerable_thread_specific<DenseAssemblerData<DenseMatrixType>>;

template<typename F, class DenseMatrixType>
struct DenseAssembler {
    template<class Derived>
    DenseAssembler(F &f, Eigen::MatrixBase<Derived> &A, DALocalData<DenseMatrixType>& locals)
        : m_f(f), m_nrows(A.rows()), m_ncols(A.cols()), m_locals(locals), m_A(A.derived()) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        // First thread accumulates directly to m_A
        if (tbb::this_task_arena::current_thread_index() == 0) {
            for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, m_A); }
            return;
        }

        // Other threads accumulate to thread-local storage
        DenseAssemblerData<DenseMatrixType> &data = m_locals.local();
        if (data.needs_reset) {
            data.A.setZero(m_nrows, m_ncols);
            data.needs_reset = false;
        }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data.A); }
    }
private:
    F &m_f;
    size_t m_nrows, m_ncols;
    DALocalData<DenseMatrixType> &m_locals;
    DenseMatrixType &m_A;
};

template<typename F, class DenseMatrixType, class Derived>
DenseAssembler<F, DenseMatrixType> make_dense_assembler(F &f, Eigen::MatrixBase<Derived> &A, DALocalData<DenseMatrixType> &locals) {
    return DenseAssembler<F, DenseMatrixType>(f, A, locals);
}

template<typename PerElemAssembler, class Derived, class Locals>
auto assemble_parallel(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems, Locals &localData) {
    for (auto &d : localData)
        d.needs_reset = true;

    get_gradient_assembly_arena().execute([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                          make_dense_assembler(assembler, A, localData));
    });

    for (const auto &d : localData)
        if (!d.needs_reset) A += d.A; // this if statement will skip thread 0's unused storage.
}

// Returns thread local storage collection so that it might be re-used.
template<typename PerElemAssembler, class Derived>
auto assemble_parallel(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems) {
    using DenseMatrixType = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options>;
    auto daLocalData = std::make_unique<DALocalData<DenseMatrixType>>();
    assemble_parallel(assembler, A, numElems, *daLocalData);
    return daLocalData;
}

////////////////////////////////////////////////////////////////////////////////
// Hessian assembly
////////////////////////////////////////////////////////////////////////////////

template<typename Real_, class CustomData_ = CTLDEmpty>
struct HessianAssemblerData {
    CSCMatrix<SuiteSparse_long, Real_> H;
    bool constructed = false;
    CustomData_ customData;
};

template<typename Real_, class CustomData_ = CTLDEmpty>
using HALocalData = tbb::enumerable_thread_specific<HessianAssemblerData<Real_, CustomData_>>;

template<class CustomData_>
struct HAFunctionCaller {
    template<class F, class HAD>
    static void run(F &f, size_t si, HAD &data) {
        f(si, data.H, data.customData);
    }
};

// Without custom data, the per-element assembler takes only the element index
// and the (thread-local) sparse Hessian to contribute to.
template<>
struct HAFunctionCaller<CTLDEmpty> {
    template<class F, class HAD>
    static void run(F &f, size_t si, HAD &data) {
        f(si, data.H);
    }
};

template<class CustomData_, class F, typename Real_>
struct HessianAssembler {
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    HessianAssembler(F &f, const CSCMat &H, HALocalData<Real_, CustomData_> &locals) : Hsp(H), m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        HessianAssemblerData<Real_, CustomData_> &data = m_locals.local();
        if (!data.constructed) { data.H.zeros_like(Hsp); data.customData.construct(); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { HAFunctionCaller<CustomData_>::run(m_f, si, data); }
    }

    const CSCMat &Hsp; // sparsity pattern for H
private:
    F &m_f;
    HALocalData<Real_, CustomData_> &m_locals;
};

template<class CustomData_, class F, typename Real_>
HessianAssembler<CustomData_, F, Real_> make_hessian_assembler(F &f, const CSCMatrix<SuiteSparse_long, Real_> &H, HALocalData<Real_, CustomData_> &locals) {
    return HessianAssembler<CustomData_, F, Real_>(f, H, locals);
}

// Assemble a Hessian in parallel
template<class CustomData_ = CTLDEmpty, class PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, CSCMatrix<SuiteSparse_long, Real_> &H, const size_t numElems) {
    HALocalData<Real_, CustomData_> haLocalData;
    get_hessian_assembly_arena().execute([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                          make_hessian_assembler<CustomData_>(assembler, H, haLocalData));
    });

    for (const auto &data : haLocalData)
        H.addWithIdenticalSparsity(data.H);
}

// Assemble a Hessian consisting of two distinct element types (e.g., membrane + hinge energies),
// for which the caller provides assembly routines assembler1 and assembler2.
template<class CustomData_ = CTLDEmpty, class PerElemAssembler1, class PerElemAssembler2, typename Real_>
void assemble_parallel(const PerElemAssembler1 &assembler1, const size_t numElems1,
                       const PerElemAssembler2 &assembler2, const size_t numElems2,
                       CSCMatrix<SuiteSparse_long, Real_> &H,
                       const std::string benchmarkTimerName1 = std::string(),
                       const std::string benchmarkTimerName2 = std::string()) {
    HALocalData<Real_, CustomData_> haLocalData;
    get_hessian_assembly_arena().execute([&]() {
        if (numElems1 > 0) {
            if (!benchmarkTimerName1.empty()) BENCHMARK_START_TIMER(benchmarkTimerName1);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems1),
                              make_hessian_assembler<CustomData_>(assembler1, H, haLocalData));
            if (!benchmarkTimerName1.empty()) BENCHMARK_STOP_TIMER(benchmarkTimerName1);
        }

        if (numElems2 > 0) {
            if (!benchmarkTimerName2.empty()) BENCHMARK_START_TIMER(benchmarkTimerName2);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems2),
                              make_hessian_assembler<CustomData_>(assembler2, H, haLocalData));
            if (!benchmarkTimerName2.empty()) BENCHMARK_STOP_TIMER(benchmarkTimerName2);
        }
    });

    for (const auto &data : haLocalData)
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
