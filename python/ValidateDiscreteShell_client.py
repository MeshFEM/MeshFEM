#%%
import threading
import MeshFEM, mesh
import numpy as np
import pyvista, pyvistaqt
pyvista.set_jupyter_backend('client')
from client_udp import Client, Message
import asyncio
import nest_asyncio
import time
from threading import Thread
nest_asyncio.apply()
# %%
def emit(message,data=None):
    return asyncio.run(client.send_single_message(Message(message,data)))

def func(reply_message):
    print(reply_message)
    #vdmesh.points(ds.getVars().reshape((-1, 3))) 
    #vdmesh = vd.Mesh([m.vertices(), m.elements()])

client = Client('localhost', 12345, func)
asyncio.run(client.connect())

m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/ball.msh', embeddingDimension=3)
m=m.boundaryMesh()
vf = {"V":m.vertices().astype(np.float32),"F":m.elements()}
emit("initialize",vf)

# %%
def start_stop_server(flag):
    if flag:
        mes = emit("start")
        print(mes.string)
    else:
        mes = emit("stop")
        print(mes.string)

vistamesh = pyvista.PolyData(m.vertices(), np.hstack((np.full((m.numTris(),1), 3), m.elements())))
plt = pyvistaqt.BackgroundPlotter()
plt.add_checkbox_button_widget(start_stop_server, value = False)
plt.add_mesh(vistamesh)
plt.show()

print("done")

wait = input("PRESS ENTER TO CONTINUE.")



# # %%
# flag = True
# def update():
#     global flag
#     while flag:
#         mess = emit("get")
#         vistamesh.points = mess.data
#         time.sleep(0.5)

# thread = Thread(target=update)
# thread.start()
# %%
