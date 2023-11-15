#%%
import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#%%
import MeshFEM, mesh
import numpy as np
from udp_com import Client, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()
import vedo
import multiprocessing as mp

vedo.settings.default_backend = 'vtk'
# %%
from IPython.core.magic import register_line_magic

@register_line_magic
def return_from_server(line):
    """
    A magic function that runs the contents of the cell on the server.
    """
    msg = client.emit("run_command", "a=" + line)
    return msg.data[1]['a']

@register_line_magic
def run_on_server(line):
    """
    A magic function that runs the contents of the cell on the server.
    """
    msg = client.emit("run_command", line)
    return msg.data[1]

# To be able to use our new magic command,
# we need to explicitly load it:

#get_ipython().register_magic_function(run_on_server, magic_kind='line')

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

def update_mesh_vertices(obj, ename = None):
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
    if btn_startstop_opt.status() == "start":
        mes = client.emit("start")
        print(mes.string)
    else:
        mes = client.emit("stop")
        print(mes.string)
    btn_startstop_opt.switch()

timer_id = None
def start_stop_update_timer(obj, ename):
    global timer_id
    if btn_startstop_timer.status() == "update view":
        timer_id = plt.timer_callback("start", dt=100)
        print(timer_id)
    else:
        print(timer_id)
        ida = plt.timer_callback("stop", timer_id=timer_id)
    btn_startstop_timer.switch()

def start_update_timer(obj, ename):
    global timer_id
    timer_id = plt.timer_callback("start", dt=100)
    print(f"started: {timer_id}")

def stop_update_timer(obj, ename):
    global timer_id
    print(f"stopped: {timer_id}")
    plt.timer_callback("stop", timer_id=timer_id)

timer_cb_id = plt.add_callback("timer", update_mesh_vertices, enable_picking=False)

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

btn_startstop_opt = add_button(start_stop_server, states=["start", "stop"])
btn_startstop_timer = add_button(start_stop_update_timer, states=["update view", "stop update view"])
btn_update_mesh_once = add_button(update_mesh_vertices, states=["update view once"])
btn_reducedelta = add_button(lambda obj, ename: client.emit('run_command','self.panelization_obj.materialForElement(0).delta /= 2'), states=["reduce delta"])

# %%
plt.show()
