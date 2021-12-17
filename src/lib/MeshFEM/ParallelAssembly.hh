#ifndef PARALLELASSEMBLY_HH
#define PARALLELASSEMBLY_HH

#include "SparseMatrices.hh"
#include "Eigen/Dense"

#include <MeshFEM/Parallelism.hh>

// Support custom thread-local data.
struct CustomThreadLocalData {
    void construct() { } // called once for each thread's copy.
};

struct CTLDEmpty : public CustomThreadLocalData { };

#if MESHFEM_WITH_TBB
// Energy summation
template<typename T>
struct SummationData {
    T v;
    bool constructed = false;
};

template<typename T>
using SumLocalData = tbb::enumerable_thread_specific<SummationData<T>>;

template<class F, typename Real_>
struct SummandEvaluator {
    SummandEvaluator(const F &f, SumLocalData<Real_> &locals) : m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        auto &data = m_locals.local();
        if (!data.constructed) data.v = 0.0;
        for (size_t i = r.begin(); i < r.end(); ++i) { data.v += m_f(i); }
    }
private:
    const F &m_f;
    SumLocalData<Real_> &m_locals;
};

template<typename Real_, typename PerElemSummand>
Real_ summation_parallel(const PerElemSummand &summand, const size_t numElems) {
    SumLocalData<Real_> localData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                      SummandEvaluator<PerElemSummand, Real_>(summand, localData));

    Real_ result = 0;
    for (const auto &d : localData)
        result += d.v; // this if statement will skip thread 0's unused storage.
    return result;
}

// Dense vector/matrix assembly (e.g., for gradient)

template<class DenseMatrixType>
struct DenseAssemblerData {
    DenseMatrixType A;
    bool needs_reset = true;
};

template<class DenseMatrixType>
using DALocalData = tbb::enumerable_thread_specific<DenseAssemblerData<DenseMatrixType>>;

template<class F, class DenseMatrixType>
struct DenseAssembler {
    template<class Derived>
    DenseAssembler(const F &f, Eigen::MatrixBase<Derived> &A, DALocalData<DenseMatrixType> &locals)
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
    const F &m_f;
    size_t m_nrows, m_ncols;
    DALocalData<DenseMatrixType> &m_locals;
    DenseMatrixType &m_A;
};

template<typename PerElemAssembler, class Derived, class DenseMatrixType>
auto assemble_parallel(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems, DALocalData<DenseMatrixType> &localData) {
    for (auto &d : localData)
        d.needs_reset = true;

    get_gradient_assembly_arena().execute([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                          DenseAssembler<PerElemAssembler, DenseMatrixType>(assembler, A, localData));
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
template<typename T, class PerElemAssembler, class Enable = void>
struct SPMatType {
    using type = CSCMatrix<SuiteSparse_long, T>;
};

// For assemblers that can accept a matrix with a reference-type sparsity pattern,
// share the sparsity pattern accross all instances.
template<typename T, class PerElemAssembler>
struct SPMatType<T, PerElemAssembler, decltype(std::declval<PerElemAssembler>()(0, std::declval<CSCMatrix<SuiteSparse_long, T, const std::vector<SuiteSparse_long> &> & /* without & we get an rvalue reference... */>()))>
{
    using type = CSCMatrix<SuiteSparse_long, T, const std::vector<SuiteSparse_long> &>;
};

template<typename T, class PerElemAssembler>
using SPMatType_t = typename SPMatType<T, PerElemAssembler>::type;

template<class SPMat_, class CustomData_ = CTLDEmpty>
struct HessianAssemblerData {
    using SPMat = SPMat_;
    std::unique_ptr<SPMat> H;
    bool constructed = false;
    CustomData_ customData;
};

template<class SPMat_, class CustomData_ = CTLDEmpty>
using HALocalData = tbb::enumerable_thread_specific<HessianAssemblerData<SPMat_, CustomData_>>;

template<class CustomData_>
struct HAFunctionCaller {
    template<class F, class SPMat, class HAD>
    static void run(F &f, size_t si, SPMat &H, HAD &data) {
        f(si, H, data.customData);
    }
};

// Without custom data, the per-element assembler takes only the element index
// and the (thread-local) sparse Hessian to contribute to.
template<>
struct HAFunctionCaller<CTLDEmpty> {
    template<class F, class SPMat, class HAD>
    static void run(F &f, size_t si, SPMat &H, HAD &/* data */) {
        f(si, H);
    }
};

template<class CustomData_, class F, typename Real_>
struct HessianAssembler {
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    using HAD    = HessianAssemblerData<SPMatType_t<Real_, F>, CustomData_>;
    using HALD   = HALocalData         <SPMatType_t<Real_, F>, CustomData_>;
    using SPMat  = typename HAD::SPMat;
    HessianAssembler(const F &f, CSCMat &H, HALD &locals) : m_H(H), m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        using FC = HAFunctionCaller<CustomData_>;
        // First thread accumulates directly to m_H
        if (tbb::this_task_arena::current_thread_index() == 0) {
            HAD &data = m_locals.local();
            if (!data.constructed) { /* don't allocate data.H */ data.customData.construct(); data.constructed = true; }
            for (size_t si = r.begin(); si < r.end(); ++si) { FC::run(m_f, si, m_H, data); }
            return;
        }

        HAD &data = m_locals.local();
        if (!data.constructed) {
            // Construct with a copy/reference to the sparsity pattern of `m_H`
            data.H = std::make_unique<SPMat>(m_H.m, m_H.n, m_H.Ap, m_H.Ai);
            // Arithmetic types are already zero-ed out by the constructor, but
            // custom types need to be explicitly set to zero.
            if (!std::is_arithmetic<Real_>::value)
                data.H->template setZero<false>();
            data.customData.construct();
            data.constructed = true;
        }
        SPMat &H = *(data.H);
        for (size_t si = r.begin(); si < r.end(); ++si) { FC::run(m_f, si, H, data); }
    }

    CSCMat &m_H;
private:
    const F &m_f;
    HALD &m_locals;
};

// Assemble a Hessian in parallel
template<class CustomData_ = CTLDEmpty, class PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, CSCMatrix<SuiteSparse_long, Real_> &H, const size_t numElems) {
    HALocalData<SPMatType_t<Real_, PerElemAssembler>, CustomData_> haLocalData;
    get_hessian_assembly_arena().execute([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems),
                          HessianAssembler<CustomData_, PerElemAssembler, Real_>(assembler, H, haLocalData));
    });

    for (const auto &data : haLocalData) {
        if (data.H != nullptr)
            H.addWithIdenticalSparsity(*(data.H));
    }
}

// Assemble a Hessian consisting of two distinct element types (e.g., membrane + hinge energies),
// for which the caller provides assembly routines assembler1 and assembler2.
template<class CustomData_ = CTLDEmpty, class PerElemAssembler1, class PerElemAssembler2, typename Real_>
void assemble_parallel(const PerElemAssembler1 &assembler1, const size_t numElems1,
                       const PerElemAssembler2 &assembler2, const size_t numElems2,
                       CSCMatrix<SuiteSparse_long, Real_> &H,
                       const std::string benchmarkTimerName1 = std::string(),
                       const std::string benchmarkTimerName2 = std::string()) {
    HALocalData<SPMatType_t<Real_, PerElemAssembler1>, CustomData_> haLocalData;
    get_hessian_assembly_arena().execute([&]() {
        if (numElems1 > 0) {
            if (!benchmarkTimerName1.empty()) BENCHMARK_START_TIMER(benchmarkTimerName1);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems1),
                              HessianAssembler<CustomData_, PerElemAssembler1, Real_>(assembler1, H, haLocalData));
            if (!benchmarkTimerName1.empty()) BENCHMARK_STOP_TIMER(benchmarkTimerName1);
        }

        if (numElems2 > 0) {
            if (!benchmarkTimerName2.empty()) BENCHMARK_START_TIMER(benchmarkTimerName2);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems2),
                              HessianAssembler<CustomData_, PerElemAssembler2, Real_>(assembler2, H, haLocalData));
            if (!benchmarkTimerName2.empty()) BENCHMARK_STOP_TIMER(benchmarkTimerName2);
        }
    });

    for (const auto &data : haLocalData) {
        if (data.H != nullptr)
            H.addWithIdenticalSparsity(*(data.H));
    }
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
