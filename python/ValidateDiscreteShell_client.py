#%%
import MeshFEM, mesh
import numpy as np
from client_udp import Client, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()
import pyvista, pyvistaqt
pyvista.set_jupyter_backend('trame')
from qasync import QEventLoop
#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# %%
# start the viewer and get the event loop
plt = pyvistaqt.BackgroundPlotter()
loop = QEventLoop(plt.app)

client = Client('localhost', 12345)
client.connect(loop)
#%%
def add_mesh_to_viewer(V,F):
    vmesh = pyvista.PolyData(V, np.hstack((np.full((F.shape[0],1), 3), F)))
    plt.add_mesh(vmesh)
    return vmesh

def load_mesh_from_server():
    V = client.emit("get").data
    F = client.emit("get_faces").data
    if V is None or F is None:
        print("no data")
        return None
    return add_mesh_to_viewer(V,F)

def update_mesh_vertices():
    mes = client.emit("get")
    if mes is None:
        return
    vf = {"V":mes.data.astype(np.float32),"F":m.elements()}
    plt.update_coordinates(vf)
    plt.render()

def set_mesh_from_file(filename):
    m = mesh.Mesh(filename, embeddingDimension=3)
    m=m.boundaryMesh()
    V=m.vertices()
    F=m.elements()
    vf = {"V":V.astype(np.float32),"F":F}
    client.emit("initialize",vf)
    return add_mesh_to_viewer(V,F)
    

def start_stop_server(flag):
    if flag:
        mes = client.emit("start")
        print(mes.string)
    else:
        mes = client.emit("stop")
        print(mes.string)


# %%
# from PyQt5.QtCore import pyqtSlot, QTimer
# @pyqtSlot()
# def mycallback():
#     print("Timer fired!")

# from IPython.lib import backgroundjobs as bg
# jobs = bg.BackgroundJobManager()
# jobs.new(loop.run_until_complete,mycallback())
set_mesh_from_file("../3rdparty/MeshFEM/misc/examples/meshes/ball.msh")

#%%
a=1
async def atest():
    global a
    a+=1
    await asyncio.sleep(1)
    a+=1

def test(flag):
    global a
    if flag:
        a+=1
        loop.run_until_complete(atest)
    else:
        print("stop")

btn = plt.add_checkbox_button_widget(test, value = False)

input("press enter to exit")



# %%
# plt.add_callback(update_mesh_vertices,interval=1000)
# flag = True
# async def start_stop_update():
#     global flag
#     while flag:
#         mes = emit("get")
#         if mes is None:
#             break
#         vistamesh.points=mes.data
#         plt.render()
#         await asyncio.sleep(0.1)

# await start_stop_update()
# # %%
# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("asyncio").setLevel(logging.DEBUG)
# loop = QEventLoop(plt.app)
# flag = True
# async def test():
#     #do 5 times
#     for i in range(5):
#         print("wait")
#         await asyncio.sleep(1)
#         print("done")
# loop.create_task(test())

# # %%
