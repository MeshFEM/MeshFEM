#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#%%
import MeshFEM, mesh
import numpy as np
from udp_com import Client, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()
import vedo

vedo.settings.default_backend = 'vtk'


# %%
# start the viewer and get the event loop
plt = vedo.Plotter()

client = Client('localhost', 12345)
client.connect()
#%%
def add_mesh_to_viewer(V,F):
    vmesh = vedo.Mesh([V, F])
    plt.add(vmesh)
    return vmesh

def load_mesh_from_server():
    V = client.emit("get").data
    F = client.emit("get_faces").data
    if V is None or F is None:
        print("no data")
        return None
    return add_mesh_to_viewer(V,F)

def update_mesh_vertices(obj, ename):
    mes = client.emit("get")
    vmesh.points(mes.data)
    plt.render()

def set_mesh_from_file(filename):
    m = mesh.Mesh(filename, embeddingDimension=3)
    m=m.boundaryMesh()
    V=m.vertices()
    F=m.elements()
    vf = {"V":V.astype(np.float32),"F":F}
    client.emit("initialize",vf)
    return add_mesh_to_viewer(V,F)
    

def start_stop_server(obj, ename):
    if btn_startstop.status() == "start":
        mes = client.emit("start")
        print(mes.string)
    else:
        mes = client.emit("stop")
        print(mes.string)
    btn_startstop.switch()

def start_stop_update_timer(obj, ename):
    if btn_timer.status() == "start":
        ida = plt.timer_callback("start", dt=1000, timer_id=timer_id, one_shot=True)
    else:
        ida = plt.timer_callback("stop", timer_id=timer_id)
    btn_timer.switch()

vmesh = set_mesh_from_file("../3rdparty/MeshFEM/misc/examples/meshes/ball.msh")

def add_button(func,states):
    btn = plt.add_button(
        func,
        states=states,  # text for each state
    )
    add_button.offset +=0.05
    btn.pos((0.0, add_button.offset),justify="left-bottom")
    return btn
add_button.offset = 0

btn_startstop = add_button(start_stop_server, states=["start", "stop"])
btn_timer = add_button(start_stop_update_timer, states=["start timer", "stop timer"])
btn_update_mesh_once = add_button(update_mesh_vertices, states=["update mesh once"])
btn_reducedelta = add_button(lambda obj, ename: client.emit('run_command','self.panelization_obj.materialForElement(0).delta /= 2'), states=["reduce delta"])
timer_id = plt.add_callback("timer", update_mesh_vertices)

# %%
plt.show()

#%%
msg = client.emit('run_command','self.panelization_obj.materialForElement(0).delta')

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
