////////////////////////////////////////////////////////////////////////////////
// DynamicSimulator.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Dynamic elasticity simulation implemented as a NewtonMultiobjectiveProblem.
//
//  Author:  Haleh Mohammadian (halehOssadat), haleh.mohammadian@gmail.com
//  Created:  07/10/2023 15:43:10
//  Modified: 01/15/2024 11:35:40
*///////////////////////////////////////////////////////////////////////////////
#ifndef DYNAMICSIMULATOR_HH
#define DYNAMICSIMULATOR_HH
#include "newton_optimizer/newton_optimizer.hh"
#include "Loads/Inertia.hh"
#include "ElasticObject.hh"

#include <MeshFEM/Solvers/CholeskyFactorizerBase.hh>
#include <MeshFEM/GlobalBenchmark.hh>

enum class TimesteppingMethod { BackwardEuler, ImplicitNewmark };

template<typename _Real>
using LoadCollection = std::vector<std::shared_ptr<Loads::Load<_Real>>>;

// Base class for terms that can limit the timestep (e.g., `IPCObjectiveTerm`).
struct MESHFEM_EXPORT TimestepLimiter {
    TimestepLimiter() { };
    virtual double getTimestepLength (double t, double dt) { return 1.0; }
    virtual void initialBarrierStiffness(double w, const Eigen::VectorXd &primaryPotentialGradient) = 0;
    virtual ~TimestepLimiter() { };
    void setAdaptiveTimestep(bool flag) { if (flag) throw std::runtime_error("Adaptive timestep is disabled until further testing"); m_useAdaptiveTimestep = flag; }
    bool useAdaptiveTimestep() { return m_useAdaptiveTimestep; }
private:
    bool m_useAdaptiveTimestep = false; // disabled for now
};

template<class _Real>
struct DynamicSimulator {
    using Real = _Real;
    using EO = ElasticObject<Real>;
    using LC = LoadCollection<Real>;
    using VXd = typename EO::VXd;
    using MXd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using NewtonTermPtr = std::shared_ptr<NewtonObjectiveTermBase>;

    using TimestepCallback = std::function<bool(DynamicSimulator &, size_t)>;
    using NewtonCallback = typename NewtonMultiobjectiveProblem::CallbackFunction;

    DynamicSimulator(const std::shared_ptr<EO> &eo, std::vector<NewtonTermPtr> &terms, bool useLumpedMass, double dt_)
        : dt(dt_), m_obj(eo), m_noninertiaTerms(terms)
    {
        v.setZero(m_obj->numVars());

        // Insert the elastic object at the beginning so that its
        // customFeasibleStepLength is called first, enabling it to cut
        // the CCD step short. This works around an OOM crash in the hashgrid
        // broad phase of `ipc_toolkit` that can apparently be triggered by
        // element inversions.
        m_noninertiaTerms.insert(m_noninertiaTerms.begin(), eo);

        m_inertiaLoad = std::make_shared<Loads::Inertia<EO>>(eo, useLumpedMass);
        m_noninertiaTerms.push_back(m_inertiaLoad); // Include the inertia term in the equilibrium problem loads.

        m_prob = std::make_shared<NewtonMultiobjectiveProblem>(m_obj, m_noninertiaTerms);

        // Drop the inertia load from our copy of the loads.
        m_noninertiaTerms.pop_back();

        m_opt = std::make_shared<NewtonOptimizer>(m_prob);

        // Set good defaults
        HessianProjectionAdaptive hpc;
        hpc.startWithProjectionActive = false;             // Dynamics problems are highly regularized by the inertia term and therefore usually do not need projection
        hpc.numConsecutiveIndefiniteStepsBeforeEnable = 0; // Enable projection immediately if the Hessian is indefinite
        hpc.numProjectionStepsBeforeDisable = 1;           // Disable projection for the following step
        m_opt->options.setHessianProjectionController(hpc);
    }

    VXd getVars() const { return m_obj->getVars(); }
    void setVars(const VXd &vars) { m_obj->setVars(vars); }

    void setInitVelocity (const VXd &v0) {
        if (v.size() != v0.size())  std::runtime_error("Size Mismatch.");
        v = v0;
    }

    void setXhat(const VXd &x){ m_inertiaLoad->setXhat(x); }
    VXd  getXhat(){ return m_inertiaLoad->xhat; }

    NewtonOptimizer &getOptimizer() const { return *m_opt; }

    VXd computeNoninertiaForces() const {
        // Computing total forces, external and potential forces
        // excluding inertia forces
        VXd f;
        f.setZero(m_obj->numVars());
        for (const auto &term : m_noninertiaTerms)
            term->accumulateGradient(-1.0, f);

        return f;
    }

    const Loads::Inertia<EO> &inertiaLoad() const { return *m_inertiaLoad; }

    CholeskyFactorizerBase &massMatrixFactorization() {
        if (m_inertiaLoad->usingLumpedMass())
            throw std::runtime_error("Lumped mass matrix does not require a Cholesky factorization; call applyMinv instead.");

        if (m_massCholesky.second && (m_massCholesky.first == m_inertiaLoad->getMassMatrixID()))
            return *(m_massCholesky.second);

        m_massCholesky.first  = m_inertiaLoad->getMassMatrixID();
        m_massCholesky.second = make_cholesky_factorizer(m_opt->options.factorizer);
        try {
            auto M_scalar = m_inertiaLoad->M_full.toScalar();
            m_massCholesky.second->factorize(M_scalar, m_prob->fixedVars());
        }
        catch (const std::exception &e) {
            std::cout << "Exception encountered when factorizing Mass matrix: " << e.what() << std::endl;
            std::cout << "Warning: lumped mass matrix is not positive definite for quadratic FEM" << std::endl;
            m_inertiaLoad->M_full.dump("failed_mass_matrix.nh");
            throw e;
        }
        return *(m_massCholesky.second);
    }

    VXd applyMinv(const VXd &b) {
        VXd result;
        if (m_inertiaLoad->usingLumpedMass()) {
            result = b.array() / m_inertiaLoad->M_lumped.array();
            for (size_t fv : m_prob->fixedVars())
                result[fv] = 0.0;
        } else {
            result = massMatrixFactorization().solve(b);
        }
        return result;
    }

    VXd configureInertiaForTimeStep(Real alpha = 1){
        VXd xt = getVars();
        VXd f_xt;
        Real alpha_dt = alpha * dt;
        // Set weights and update xhat based on the method
        if (method == TimesteppingMethod::BackwardEuler) {
            m_inertiaLoad->weight = 1.0 / (alpha_dt * alpha_dt);
            m_inertiaLoad->xhat = xt + alpha_dt * v;
        }
        else if (method == TimesteppingMethod::ImplicitNewmark) {
            m_inertiaLoad->weight = 1.0 / (beta * alpha_dt * alpha_dt);
            f_xt = computeNoninertiaForces();
            m_inertiaLoad->xhat = xt + alpha_dt * v
                                    + (alpha_dt * alpha_dt * (1.0 - 2.0 * beta) / 2.0) * applyMinv(f_xt);
        }
        else throw std::runtime_error("Method is not implemented");

        return f_xt;
    }

    void timeStep(Real alpha = 1.0) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("DynamicSimulator.timeStep");
        VXd xt = getVars();

        Real alpha_dt = alpha * dt;

        VXd f_xt = configureInertiaForTimeStep(alpha);

        // Initialize Barrier Stiffness in every time step
        for (size_t i = 0; i < m_noninertiaTerms.size(); i++) {
            const auto &term = m_noninertiaTerms[i];
            auto derivedObj = std::dynamic_pointer_cast<TimestepLimiter>(term);
            if (derivedObj != nullptr){
                derivedObj->initialBarrierStiffness(dt * dt /** m_prob->weight(i)*/, primaryPotentialGradient()); // Need weight = dt^2 to match IPC formulation
            }
        }
   
        // std::cout << "Inertia term: " << m_inertiaLoad->energy() << std::endl;
        m_crs.push_back(m_opt->optimize());

        if (method == TimesteppingMethod::BackwardEuler) {
            v = (getVars() - xt) / alpha_dt;
            // Debugging: compare `(x^{t + 1} - x^t) / dt` with `v^t + dt * M^{-1} f(x^{t + 1})`.
            // VXd v_recompute_from_accel = v + dt * applyMinv(computeForces());
            // std::cout << "v_manual norm: " << v_recompute_from_accel.norm() << std::endl;
            // std::cout << "v norm: " <<  v.norm() << std::endl;
            // std::cout << "v_manual relative error: " << (v_recompute_from_accel - v).norm() / v.norm() << std::endl;
        }
        else if (method == TimesteppingMethod::ImplicitNewmark) {
            VXd b = alpha_dt * ((1.0 - gamma) * f_xt + gamma * computeNoninertiaForces());
            v += applyMinv(b);
        }

        m_kineticEnergy.push_back(kineticEnergy());
        m_potentialEnergy.push_back(potentialEnergy());
    }

    Real kineticEnergy() const { return 0.5 * m_inertiaLoad->evalQuadraticForm(v); }
    Real potentialEnergy() const {
        Real result = 0.0;//m_obj->energy();
        for (const auto &term : m_noninertiaTerms)
            result += term->objective();
        return result;
    }

    const std::vector<Real> &  kineticEnergies() const { return   m_kineticEnergy; }
    const std::vector<Real> &potentialEnergies() const { return m_potentialEnergy; }

    void setPostTimestepCallback(const TimestepCallback &cb) { m_postTimestepCallback = cb; }
    void  setPreTimestepCallback(const TimestepCallback &cb) { m_preTimestepCallback = cb; }
    void       setNewtonCallback(const   NewtonCallback &cb) { m_prob->setCustomIterationCallback(cb); }

    void setFixedVars(const std::vector<size_t> &fixedVars) { m_prob->setFixedVars(fixedVars); m_massCholesky.second.reset(); }
    const std::vector<size_t> &fixedVars() const { return m_prob->fixedVars(); }

    Eigen::VectorXd primaryPotentialGradient() const {
        Eigen::VectorXd grad;
        grad.setZero(m_obj->numVars());
        
        for (size_t i = 0; i < m_noninertiaTerms.size(); i++) {
            const auto &term = m_noninertiaTerms[i];
            if (std::dynamic_pointer_cast<TimestepLimiter>(term)) continue; // Skip contact term
            term->accumulateGradient(m_prob->weight(i), grad);
        }
        // grad miss inertia term
        grad += m_inertiaLoad->grad_x();

        return grad;
    }

    std::vector<ConvergenceReport> run(const double initTime, const double finalTime) {
        if (initTime >= finalTime) std::runtime_error("Time mismatch: initTime >= finalTime.");

        double time = 0.0;
        m_kineticEnergy  .assign(1,   kineticEnergy());
        m_potentialEnergy.assign(1, potentialEnergy());
        m_crs.clear();
        const double numTimeSteps = std::ceil((finalTime - initTime) / dt);
        m_crs.reserve(numTimeSteps);
        for (size_t t = 0; t < numTimeSteps; t++) {
            Real alpha = 1.0;

            if (m_iterationCallback(t, m_preTimestepCallback)) break;

            // Adaptive time stepping for preventing collision of obstacle and elastic object
            for (size_t i = 0; i < m_noninertiaTerms.size(); i++) {
                const auto &term = m_noninertiaTerms[i];
                auto derivedObj = std::dynamic_pointer_cast<TimestepLimiter>(term);
                if (derivedObj != nullptr){
                    if (derivedObj->useAdaptiveTimestep()) {
                        alpha = derivedObj->getTimestepLength(time, dt);
                    }
                }
            }

            timeStep(alpha);
            time += alpha * dt;
            {
                // BENCHMARK_SCOPED_TIMER_SECTION timer("DynamicSimulator.postTimestepCallback"); // When collecting per-timestep benchmarks, we use a callback that resets the timer stack. This will cause problems if it's nested in a scoped timer...
                if (m_iterationCallback(t, m_postTimestepCallback)) {
                    break;
                }
            }
        }
        return m_crs;
    }

    std::shared_ptr<NewtonMultiobjectiveProblem> getProblem() const { return m_prob; }

    double dt = 0.1;  // Time step size
    
    TimesteppingMethod method = TimesteppingMethod::BackwardEuler;

    // Parameters of implicit Newmark integration
    Real beta = 0.25, gamma = 0.5;

    VXd v;

private:
    bool m_iterationCallback(size_t i, TimestepCallback &customCallback) {
        if (customCallback) return customCallback(*this, i);
        return false; // don't exit early
    }

    std::shared_ptr<EO> m_obj;
    std::vector<NewtonTermPtr> m_noninertiaTerms;
    std::shared_ptr<Loads::Inertia<EO>> m_inertiaLoad;
    std::shared_ptr<NewtonMultiobjectiveProblem> m_prob;
    std::shared_ptr<NewtonOptimizer> m_opt;

    TimestepCallback m_postTimestepCallback;
    TimestepCallback m_preTimestepCallback;

    // (Mass Matrix ID, Cholesky Factorization)
    std::pair<size_t, std::shared_ptr<CholeskyFactorizerBase>> m_massCholesky;

    // Per-timestep statistics
    std::vector<ConvergenceReport> m_crs; // Newton solver convergence report
    std::vector<Real> m_kineticEnergy, m_potentialEnergy;
};

#endif
