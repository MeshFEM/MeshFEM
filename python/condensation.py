import numpy as np
import py_newton_optimizer
import benchmark

class CondensedElasticObject:
    """
    Implements a statically condensed elastic object. A subset of the degrees of
    freedom of the elastic object are condensed out via nonlinear optimization.
    Then gradients and dense Hessians of the energy with respect to the
    remaining degrees of freedom are computed.
    """
    def __init__(self, elastic_object, free_variable_indices, pinned_variable_indices = [], opts = None):
        """
        :param elastic_object:        The elastic object to be condensed.
                                      Note: condensation is applied directly to this object
                                      instance, meaning its its vars will be modified!
        :param free_variable_indices: The indices of the degrees of freedom to be kept;
                                      all others will be condensed out (or pinned).
        :param pinned_variable_indices: The indices of the degrees of freedom to be pinned during condensation.
        """
        self.object = elastic_object

        free_variable_indices = np.unique(free_variable_indices)
        self.free_variable_indices = free_variable_indices
        self.condensed_variable_indices = np.setdiff1d(np.arange(len(self.object.getVars())), free_variable_indices)
        self.pinned_variable_indices = pinned_variable_indices

        if self.numCondensedVars() == 0:
            self.eq = None
            self.object_leq = None
            return

        if opts is None:
            opts = py_newton_optimizer.NewtonOptimizerOptions()
            opts.niter = 100
            opts.verbose = False

        self.eq = self.object.equilibriumOptimizer(fixedVars=list(self.free_variable_indices) + self.pinned_variable_indices, opts=opts)

        # If the elastic energy is nonlinear, we construct an auxiliary
        # linear elastic object to compute a reasonable initial guess for the
        # nonlinear optimization.
        eo_name = self.object.__class__.__name__
        if 'LinearElasticity' not in eo_name and 'ElasticSolid' in eo_name:
            import elastic_solid, energy
            self.object_leq = elastic_solid.ElasticSolid(self.object.mesh(), energy.IsotropicLinearElastic(2, 1, 0.3))
            self.le_eq = self.object_leq.equilibriumOptimizer(fixedVars=list(self.free_variable_indices) + self.pinned_variable_indices, opts=opts)
        else: self.object_leq = None

        self.setVars(elastic_object.getVars()[free_variable_indices])

    def numVars(self): return len(self.free_variable_indices)
    def getVars(self): return self.object.getVars()[self.free_variable_indices]
    def numCondensedVars(self): return len(self.object.getVars()) - len(self.free_variable_indices)

    def setVars(self, v):
        # TODO: better initialization strategy
        x_guess = self.object.getVars()
        x_guess[self.free_variable_indices] = v

        # Initialize the condensed variables' positions using
        # linear elasticity.
        if self.object_leq is not None:
            self.object_leq.setVars(x_guess)
            self.le_eq.optimize()
            x_guess = self.object_leq.getVars()

        self.object.setVars(x_guess)

        self.factorizations_up_to_date = False

        if self.eq is not None:
            self.eq.optimize()

    def energy(self): return self.object.energy()

    def gradient(self):
        # Sensitivity of the energy with respect to the condensed variables
        # can be neglected due to the envelope theorem.
        return self.object.gradient()[self.free_variable_indices]

    def hessian(self):
        if self.numCondensedVars() == 0:
            return self.object.hessian()

        benchmark.start_timer_section('CondensedElasticObject.hessian')

        if not self.factorizations_up_to_date:
            self.eq.update_factorizations()
            self.factorizations_up_to_date = True

        # Let H = [H_ff H_fc]
        #         [H_cf H_cc]
        # where `f` are the free variables and `c` are the condensed variables.
        # The Hessian with respect to the free variables is given by the
        # Schur complement `H_ff - H_fc * H_cc^-1 * H_cf`
        H = self.object.hessian().H_ss.toScalar()
        H_scipy = H.toSymmetryMode(H.symmetry_mode.NONE).toSciPy()

        fvi = self.  free_variable_indices
        pvi = self.pinned_variable_indices

        H__f = H_scipy[:, fvi].todense()
        H_ff = H__f[fvi, :].copy()

        benchmark.start_timer_section('extract_blocks')
        H__f[fvi, :] = 0
        H__f[pvi, :] = 0
        H_cf = H__f[self.condensed_variable_indices, :]
        benchmark.stop_timer_section('extract_blocks')

        # TODO: bind and use solveMultiRHS...
        H_cc_inv_H_cf = np.column_stack([self.eq.hessian_factorization.solve(H__f[:, col])[self.condensed_variable_indices] for col in range(H__f.shape[1])])

        benchmark.start_timer_section('form_schur_complement')
        H_ff -= H_cf.transpose() @ H_cc_inv_H_cf
        benchmark.stop_timer_section('form_schur_complement')

        benchmark.stop_timer_section('CondensedElasticObject.hessian')

        self.H_cf = H_cf
        self.H_cc_inv_H_cf = H_cc_inv_H_cf

        return H_ff
