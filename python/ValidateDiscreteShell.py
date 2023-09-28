#%%
import MeshFEM, mesh
import discrete_shell, loads, sim_utils
import numpy as np
import vedo as vd 
from matplotlib import pyplot as plt
vd.settings.default_backend='vtk'
m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/lilium.msh', embeddingDimension=3)
# %%
def func(evt):                       ### called every time mouse moves!
    msh = evt.actor
    if not msh:
        return                       # mouse hits nothing, return.
    pt = evt.picked3d                # 3d coords of point under mouse

    pid = msh.closest_point(pt, return_point_id=True)
    n = msh.normal_at(pid)         # compute normals at pid
    arw = vd.Arrow(pt, pt + n, s=0.001, c='orange5')
    if len(plt.actors) > 3:
        plt.pop()                    # remove the old flagpole

    plt.add(arw).render()        # add Arrow and the new flagpole

vdmesh = vd.Mesh([m.vertices(), m.elements()])
vdmesh.computeNormals()
plt = vd.Plotter(axes=1, bg2='lightblue')

plt.add_callback('mouse move', func) # add the callback function
plt.add_callback('keyboard', lambda evt: plt.remove(plt.actors[3:]).render())

plt.show(vdmesh, __doc__, viewup='z')
# %%
ds = discrete_shell.DiscreteShell(m,youngModulus = 1)
#g = discrete_shell.Gravity(ds, rho=1e-1, g=[0, -9.81, 0]) # mass density in kg/mm^3, gravitational acceleration in N/kg
fixedVars = sim_utils.getBBoxVars(ds, sim_utils.BBoxFace.MIN_Z, tol=1e-2)

#%%
ds.computeEquilibrium(fixedVars=fixedVars)
#%%
ds.setVars(ds.getVars() + 1e-3 * np.random.normal(size=ds.numVars()))
#%%
import fd_validation
fd_validation.gradConvergencePlot(ds)
fd_validation.hessConvergencePlot(ds)
H = ds.hessian()
plt.spy(H.toSciPy(), markersize=1)
# %%
