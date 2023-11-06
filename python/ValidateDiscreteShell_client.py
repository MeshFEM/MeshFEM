#%%
import threading
import MeshFEM, mesh
import numpy as np
import pyvista
pyvista.set_jupyter_backend('client')
from client_udp import Client, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()
# %%

def func(reply_message):
    print(reply_message)
    #vdmesh.points(ds.getVars().reshape((-1, 3))) 
    #vdmesh = vd.Mesh([m.vertices(), m.elements()])

client = Client('localhost', 12345, func)
await client.connect()

m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/ball.msh', embeddingDimension=3)
m=m.boundaryMesh()
vf = {"V":m.vertices().astype(np.float32),"F":m.elements()}
await client.send_single_message(Message("initialize",vf))

# %%
def update(evt):
    asyncio.run(client.send_single_message("start"))

def start_stop_server():
    if bu.status == 0:
        asyncio.run(client.send_single_message("start"))
    else:1244
        asyncio.run(client.send_single_message("stop"))

m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/lilium.msh', embeddingDimension=3)
mesh = pyvista.PolyData(m.vertices(), np.hstack((np.full((m.numTris(),1), 3), m.elements())))
plt = pyvista.Plotter()
plt.add_mesh(mesh)

plt.add_checkbox_button_widget(toggle_vis, value=True)


#plt.addCallback('mouseclick', func)
#plt.addCallback('keypress', lambda evt: plt.close() if evt.key == 'q' else None)
plt.add_callback('keypress', update)
plt.show(vdmesh, viewup='z').close()
print("done")

# %%
