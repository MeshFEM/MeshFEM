#ifndef PARALLELASSEMBLY_HH
#define PARALLELASSEMBLY_HH

#include "Eigen/Dense"

#include <MeshFEMCore/Parallelism.hh>
#include <MeshFEMCore/ParallelVectorOps.hh>
#include <MeshFEMCore/function_traits.hh>
#include <MeshFEMSparse/SparseMatrices.hh>

namespace MeshFEM {

// Support custom thread-local data.
struct CustomThreadLocalData {
    void construct() { } // called once for each thread's copy.
};

struct CTLDEmpty : public CustomThreadLocalData { };

#if MESHFEM_WITH_TBB
// Scalar summation (e.g., for per-element energies)
template<typename T>
struct SummationData {
    T v;
    bool constructed = false;
};

template<typename T>
using SumLocalData = tbb::enumerable_thread_specific<SummationData<T>>;

template<class F, typename Real_, typename enabled = void>
struct SummandEvaluator;

// Evaluator for loop bodies that take an individual index
template<class F, typename Real_>
struct SummandEvaluator<F, Real_, std::enable_if_t<std::is_integral<typename function_traits<F>::template arg<0>::type>::value>> {
    SummandEvaluator(const F &f, SumLocalData<Real_> &locals) : m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        auto &data = m_locals.local();
        if (!data.constructed) { data.v = 0.0; data.constructed = true; }
        for (size_t i = r.begin(); i < r.end(); ++i) { data.v += m_f(i); }
    }
private:
    const F &m_f;
    SumLocalData<Real_> &m_locals;
};

// Evaluator for loop bodies that take an entire `tbb::blocked_range`.
template<class F, typename Real_>
struct SummandEvaluator<F, Real_, std::enable_if_t<!std::is_integral<typename function_traits<F>::template arg<0>::type>::value>> {
    SummandEvaluator(const F &f, SumLocalData<Real_> &locals) : m_f(f), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        auto &data = m_locals.local();
        if (!data.constructed) { data.v = 0.0; data.constructed = true; }
        data.v += m_f(r);
    }
private:
    const F &m_f;
    SumLocalData<Real_> &m_locals;
};

template<typename PerElemSummand>
return_type<PerElemSummand> summation_parallel(const PerElemSummand &summand, const size_t numElems, size_t grain_size = 24) {
    using Real_ = return_type<PerElemSummand>;
    SumLocalData<Real_> localData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems, grain_size),
                      SummandEvaluator<PerElemSummand, Real_>(summand, localData));

    Real_ result;
    result = 0; // Not used as initialization above because that would break the Eigen::Array type...
    for (const auto &d : localData)
        result += d.v;
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

template<class F>
struct accepts_integer_index { static constexpr bool value = std::is_integral<typename function_traits<F>::template arg<0>::type>::value; };

// Note: `DenseMatrixType` is a storage-backed version of `DestinationType`
// (needed for cases, e.g., where `DestinationType` is really an Eigen::Map).
template<class F, class DestinationType, class DenseMatrixType>
struct DenseAssembler {
    DenseAssembler(const F &f, DestinationType &A, DALocalData<DenseMatrixType> &locals)
        : m_f(f), m_nrows(A.rows()), m_ncols(A.cols()), m_locals(locals), m_A(A.derived()) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        // First thread accumulates directly to m_A
        if (tbb::this_task_arena::current_thread_index() == 0) {
            if constexpr (accepts_integer_index<F>::value) { for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, m_A.derived()); } }
            else { m_f(r, m_A.derived()); }
            return;
        }

        // Other threads accumulate to thread-local storage
        DenseAssemblerData<DenseMatrixType> &data = m_locals.local();
        if (data.needs_reset) {
            data.A.setZero(m_nrows, m_ncols);
            data.needs_reset = false;
        }
        if constexpr (accepts_integer_index<F>::value) { for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data.A); } }
        else { m_f(r, data.A); }
    }
private:
    const F &m_f;
    size_t m_nrows, m_ncols;
    DALocalData<DenseMatrixType> &m_locals;
    DestinationType &m_A;
};

template<typename PerElemAssembler, class Derived, class DenseMatrixType>
void assemble_parallel_noarena(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems, DALocalData<DenseMatrixType> &localData) {
    const size_t parallelism_threshold = 256;
    if ((numElems < parallelism_threshold) || (get_max_num_tbb_threads() == 1)) {
        if constexpr (accepts_integer_index<PerElemAssembler>::value) {
            for (size_t i = 0; i < numElems; ++i)
                assembler(i, A.derived());
        } else {
            assembler(tbb::blocked_range<size_t>(0, numElems), A.derived());
        }
        return;
    }

    for (auto &d : localData)
        d.needs_reset = true;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numElems, /* grain_size = */ 100),
                      DenseAssembler<PerElemAssembler, Eigen::MatrixBase<Derived>, DenseMatrixType>(assembler, A, localData));

    if (A.rows() < 10 * 1024) { // Threshold for parallel reduction of per-thread results.
        for (const auto &d : localData)
            if (!d.needs_reset) A += d.A;
        return;
    }

    std::vector<const DenseMatrixType *> toAdd;
    toAdd.reserve(localData.size());
    for (const auto &d : localData)
        if (!d.needs_reset) toAdd.push_back(&d.A); // skips unused storage (e.g., if number of threads was reduced after localData was constructed)
    if (toAdd.empty()) return;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Combine per-thread dense results");
    // WARNING: using `tbb::affinity_partitioner` for this loop appears to cause
    // a memory bug on macOS (EXC_BAD_ACCESS in tbb code, hangs in scheduler, etc.)
    // This is most likely a tbb bug :(
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.rows(), 1024),
                      [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = 0; i < toAdd.size(); ++i)
                A.middleRows(r.begin(), r.size()) += toAdd[i]->middleRows(r.begin(), r.size());
        });
}

template<typename PerElemAssembler, class Derived>
auto assemble_parallel_noarena(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems) {
    using DenseMatrixType = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options>;
    auto daLocalData = std::make_unique<DALocalData<DenseMatrixType>>();
    assemble_parallel_noarena(assembler, A, numElems, *daLocalData);
    return daLocalData;
}

template<typename PerElemAssembler, class Derived, class DenseMatrixType>
void assemble_parallel(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems, DALocalData<DenseMatrixType> &localData) {
    get_gradient_assembly_arena().execute([&]() { assemble_parallel_noarena(assembler, A, numElems, localData); });
}

// Returns thread local storage collection so that it might be re-used.
template<typename PerElemAssembler, class Derived>
auto assemble_parallel(const PerElemAssembler &assembler, Eigen::MatrixBase<Derived> &A, const size_t numElems) {
    using DenseMatrixType = std::decay_t<decltype(A.eval())>;
        // Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options>;
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
            // We need to propagate `symmetry_mode` in case the assembly
            // routine calls `addNZBlock` or another symmetry-dependent method!
            data.H->symmetry_mode = static_cast<typename SPMat::SymmetryMode>(m_H.symmetry_mode);
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

    BENCHMARK_SCOPED_TIMER_SECTION timer("Combine per-thread matrices");
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

// Hessian (sparsity pattern) assembly for triplet matrix. Temporary stub implementation that operates serially.
template<class CustomData_ = CTLDEmpty, typename PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, TripletMatrix<Triplet<Real_>> &H, const size_t numElems) {
    CustomData_ customData;
    for (size_t ei = 0; ei < numElems; ++ei)
        assembler(ei, H, customData);
}

#else

// Fallback to serial assembly.
template<typename PerElemAssembler, typename Real_>
void assemble_parallel(const PerElemAssembler &assembler, CSCMatrix<SuiteSparse_long, Real_> &H, const size_t numElems) {
    for (size_t ei = 0; ei < numElems; ++ei)
        assembler(ei, H);
}
#endif

} // namespace MeshFEM

#endif /* end of include guard: PARALLELASSEMBLY_HH */
