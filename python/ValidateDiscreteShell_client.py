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

# %%
# start the viewer and get the event loop
plt = pyvistaqt.BackgroundPlotter()
loop = QEventLoop(plt.app)

client = Client('localhost', 12345)
loop.run_until_complete(client.connect())
#%%
def emit(message,data=None):
    return asyncio.run(client.send_single_message(Message(message,data)))

m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/ball.msh', embeddingDimension=3)
m=m.boundaryMesh()
vf = {"V":m.vertices().astype(np.float32),"F":m.elements()}
emit("initialize",vf)

def start_stop_server(flag):
    if flag:
        mes = emit("start")
        print(mes.string)
    else:
        mes = emit("stop")
        print(mes.string)
# %%
vistamesh = pyvista.PolyData(m.vertices(), np.hstack((np.full((m.numTris(),1), 3), m.elements())))

plt.add_checkbox_button_widget(start_stop_server, value = False)
plt.add_mesh(vistamesh)
plt.show()

print("done")
# %%
flag = True
async def start_stop_update():
    global flag
    while flag:
        mes = emit("get")
        if mes is None:
            break
        vistamesh.points=mes.data
        plt.render()
        await asyncio.sleep(0.1)

await start_stop_update()
# %%
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("asyncio").setLevel(logging.DEBUG)
loop = QEventLoop(plt.app)
flag = True
async def test():
    print("wait")
    await asyncio.sleep(0.5)
    print("done")
loop.run_until_complete(test())

# %%
