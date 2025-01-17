////////////////////////////////////////////////////////////////////////////////
// NewtonOptions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Options for the Newton optimizer.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEWTONOPTIONS_HH
#define NEWTONOPTIONS_HH

#include <MeshFEM/Types.hh>
#include "HessianProjectionController.hh"
#include "HessianUpdateController.hh"
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>

struct NewtonOptimizerOptionsBase {
    Real gradTol = 2e-8,
         beta = 1e-8;
    bool hessianScaledBeta = true;
    size_t niter = 100;                        // Maximum number of newton iterations
    bool useIdentityMetric = false;            // Whether to force the use of the identity matrix for Hessian modification (instead of the problem's custom metric)
    bool useNegativeCurvatureDirection = true; // Whether to compute and move in negative curvature directions to escape from saddle points.
    bool feasibilitySolve = true;              // Whether to solve for a feasible starting point or rely on the problem to jump to feasible parameters.
    int verbose = 1;
    bool verboseNonPosDef = false;             // Print CHOLMOD warning for non-pos-def matrices
    int stdoutFlushInterval = 1;               // How often to flush stdout (e.g., for immediate updates in Jupyter notebook or for reduced disk i/o when redirecting to a file in a HPC setting)
    bool writeIterateFiles = false;
    // Warning: the following fields are NOT serialized for reasons of backwards compatibility
    size_t nbacktrack_iter = 25;               // Number of backtracking iterations to run before giving up on the linesearch
    size_t ngd_fallback_steps = 3;             // Total number of "fall-backs iterations" trying the neg gradient instead of the Newton direction
    int  verboseWorkingSet = 0;                // Whether to report changes to the working set (>0) and the contents of nonempty working sets upon termination (>1).
    CholeskyProvider factorizer = get_default_cholesky_provider();
    std::string matrixRecordDir;               // If nonempty, all Hessian sparsity patterns and values encountered during the optimization will be recorded to this directory.
};

// The part of the optimizer interface that is not trivially copyable.
struct MESHFEM_EXPORT NewtonOptimizerOptions : public NewtonOptimizerOptionsBase {
    NewtonOptimizerOptions() = default;
    NewtonOptimizerOptions(const NewtonOptimizerOptions &b)
        : NewtonOptimizerOptionsBase(b),
          m_hessianProjectionController(b.m_hessianProjectionController->clone()),
          m_hessianUpdateController(b.m_hessianUpdateController->clone())
    { }

    NewtonOptimizerOptions &operator=(const NewtonOptimizerOptions &b) {
        NewtonOptimizerOptionsBase::operator=(b);
        m_hessianProjectionController = b.m_hessianProjectionController->clone();
        m_hessianUpdateController     = b.m_hessianUpdateController->clone();
        return *this;
    }

    HessianProjectionController &getHessianProjectionController() const { return *m_hessianProjectionController; }
    void setHessianProjectionController(const HessianProjectionController &hpc) { m_hessianProjectionController = hpc.clone(); }

    HessianUpdateController &getHessianUpdateController() const { return *m_hessianUpdateController; }
    void setHessianUpdateController(const HessianUpdateController &huc) { m_hessianUpdateController = huc.clone(); }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<Real, Real, bool, size_t, bool, bool, bool, int, bool, bool, std::shared_ptr<HessianProjectionController>, std::shared_ptr<HessianUpdateController>, size_t, size_t, CholeskyProvider>;
    using StateBackwardCompat = std::tuple<Real, Real, bool, size_t, bool, bool, bool, int, bool, bool, std::shared_ptr<HessianProjectionController>, std::shared_ptr<HessianUpdateController>>; // before nbacktrack_iter and ngd_fallback_steps were added
    using StateBackwardCompat2 = std::tuple<Real, Real, bool, size_t, bool, bool, bool, int, bool, bool, std::shared_ptr<HessianProjectionController>, std::shared_ptr<HessianUpdateController>, size_t, size_t>; // before CholeskyProvider was added
    static State serialize(const NewtonOptimizerOptions &opts) {
        return std::make_tuple(opts.gradTol,  opts.beta,
                               opts.hessianScaledBeta, opts.niter, opts.useIdentityMetric,
                               opts.useNegativeCurvatureDirection, opts.feasibilitySolve,
                               opts.verbose, opts.writeIterateFiles, opts.verboseNonPosDef,
                               opts.m_hessianProjectionController, opts.m_hessianUpdateController,
                               opts.nbacktrack_iter, opts.ngd_fallback_steps, opts.factorizer);
    }
    template<typename State_>
    static std::unique_ptr<NewtonOptimizerOptions> deserialize_(const State_ &state) {
        auto opts = std::make_unique<NewtonOptimizerOptions>();
        opts->gradTol                       = std::get<0 >(state);
        opts->beta                          = std::get<1 >(state);
        opts->hessianScaledBeta             = std::get<2 >(state);
        opts->niter                         = std::get<3 >(state);
        opts->useIdentityMetric             = std::get<4 >(state);
        opts->useNegativeCurvatureDirection = std::get<5 >(state);
        opts->feasibilitySolve              = std::get<6 >(state);
        opts->verbose                       = std::get<7 >(state);
        opts->writeIterateFiles             = std::get<8 >(state);
        opts->verboseNonPosDef              = std::get<9 >(state);
        opts->m_hessianProjectionController = std::get<10>(state);
        opts->m_hessianUpdateController     = std::get<11>(state);
        return opts;
    }
    static std::unique_ptr<NewtonOptimizerOptions> deserialize(const StateBackwardCompat &state) { return deserialize_(state); }
    static std::unique_ptr<NewtonOptimizerOptions> deserialize(const StateBackwardCompat2 &state) {
        auto opts = deserialize_(state);
        opts->nbacktrack_iter    = std::get<12>(state);
        opts->ngd_fallback_steps = std::get<13>(state);
        return opts;
    }
    static std::unique_ptr<NewtonOptimizerOptions> deserialize(const State &state) {
        auto opts = deserialize_(state);
        opts->nbacktrack_iter    = std::get<12>(state);
        opts->ngd_fallback_steps = std::get<13>(state);
        opts->factorizer         = std::get<14>(state);
        return opts;
    }
    std::unique_ptr<NewtonOptimizerOptions> clone() { return deserialize(serialize(*this)); }

protected:
    // `shared_ptr` to support pickling
    std::shared_ptr<HessianProjectionController> m_hessianProjectionController = std::make_shared<HessianProjectionAdaptive>();
    std::shared_ptr<HessianUpdateController>     m_hessianUpdateController     = std::make_shared<HessianUpdateAlways>();
};

#endif /* end of include guard: NEWTONOPTIONS_HH */
