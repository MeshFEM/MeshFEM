# %%
import MeshFEM
import mesh, mesh_energy
import numpy as np
import symdir_test
import parametrization


# %%
m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/cone.obj')
optVars = mesh_energy.NodalVars(m, 2)
optVars.setVars(parametrization.lscm(m).ravel())

# %%
sde_ad = symdir_test.param_sym_dirichlet_edensity_ad(m, optVars)
# sde = symdir_test.param_sym_dirichlet_edensity(m, optVars)
sde_elem = symdir_test.param_sym_dirichlet_element(m, optVars)
# de_elem_ad = dirichlet_demo.param_dirichlet_element_ad(m, optVars)
# sde_elem_ad = dirichlet_demo.param_symdirichlet_element_ad(m, optVars)

# %%
# Finite difference validations of gradient and Hessian
import fd_validation, py_newton_optimizer
prob = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [sde_elem])

fd_validation.gradConvergencePlot(prob)
fd_validation.hessConvergencePlot(prob)


# %%
