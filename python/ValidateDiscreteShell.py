#%%
import MeshFEM, mesh    
import discrete_shell, loads, sim_utils
import numpy as np
import vedo as vd 
import py_newton_optimizer
import benchmark
vd.settings.default_backend='vtk'

#%%
m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/lilium.msh', embeddingDimension=3)
ds = discrete_shell.DiscreteShell(m,youngModulus = 1)
g = discrete_shell.Gravity(ds, rho=1e-1, g=[0, -9.81, 0]) # mass density in kg/mm^3, gravitational acceleration in N/kg
fixedVars = sim_utils.getBBoxVars(ds, sim_utils.BBoxFace.MIN_Z, tol=1e-2)
attachmentPoints = [loads.AttachmentPointCoordinate([i], [1]) for i in range(ds.numVars())]
targets = [loads.AttachmentPointCoordinate(v) for v in m.vertices().ravel()]
s=discrete_shell.Springs(ds, attachmentPoints, targets, 1.0)
# %%
def func(evt):
    ds.computeEquilibrium(loads = [g,s], fixedVars=fixedVars,opts=nopts)
    v = ds.getVars()
    vdmesh.points(ds.getVars().reshape((-1, 3))) 
    #vdmesh = vd.Mesh([m.vertices(), m.elements()])

vdmesh = vd.Mesh([m.vertices(), m.elements()])
plt = vd.Plotter(axes=1, bg2='lightblue')
nopts = py_newton_optimizer.NewtonOptimizerOptions()
nopts.niter = 1
plt.addCallback('mouseclick', func)
#plt.addCallback('keypress', lambda evt: plt.close() if evt.key == 'q' else None)
plt.addCallback('keypress', func)
plt.show(vdmesh, viewup='z')

#%%
vdmesh = vd.Mesh([m.vertices(), m.elements()])
vd.plot(vdmesh)
#%%
ds.setVars(ds.getVars() + 1e-3 * np.random.normal(size=ds.numVars()))
#%%
import fd_validation
fd_validation.gradConvergencePlot(ds)
fd_validation.hessConvergencePlot(ds)
H = ds.hessian()
plt.spy(H.toSciPy(), markersize=1)
# %%
def func(evt):                   
    msh = evt.actor
    if not msh:
        return                   
    pt = evt.picked3d            
    vdmesh = vd.Mesh([m.vertices(), m.elements()])
    
vdmesh = vd.Mesh([m.vertices(), m.elements()])
plt = vd.Plotter(axes=1, bg2='lightblue')

plt.show(vdmesh, viewup='z',interactive=False)

nopts = py_newton_optimizer.NewtonOptimizerOptions()
nopts.niter = 100
benchmark.reset()
ds.computeEquilibrium(loads = [g,s], fixedVars=fixedVars,opts=nopts,cb = func)
benchmark.report()
