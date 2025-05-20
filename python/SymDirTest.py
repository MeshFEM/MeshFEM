# %%
import MeshFEM
import mesh, mesh_energy
import numpy as np
import symdir_test
import parametrization
import fd_validation, py_newton_optimizer

# %%
m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/lilium.msh',embeddingDimension = 3)
optVars = mesh_energy.NodalVars(m, 2)
optVars.setVars(parametrization.lscm(m).ravel())

# %%
sde_ad = symdir_test.param_sym_dirichlet_edensity_ad(m, optVars)
sde_elem = symdir_test.param_sym_dirichlet_element(m, optVars)
#%%
print(sde_elem.elementEnergy(0))
print(np.round(np.reshape(sde_elem.gradient(), (3,2)), 6))
print(np.round(sde_elem.hessian().toSciPy().toarray(), 6))

print(sde_ad.elementEnergy(0))
print(np.round(np.reshape(sde_ad.gradient(), (3,2)), 6))
print(np.round(sde_ad.hessian().toSciPy().toarray(), 6))
# %%
# Finite difference validations of gradient and Hessian
prob_elem = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [sde_elem])

fd_validation.gradConvergencePlot(prob_elem)
fd_validation.hessConvergencePlot(prob_elem)
#%%
prob_ad = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [sde_ad])

fd_validation.gradConvergencePlot(prob_ad)
fd_validation.hessConvergencePlot(prob_ad)


# %%
