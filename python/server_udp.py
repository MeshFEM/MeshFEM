import asyncio
import random
import pickle
from jetblack_datagram import start_udp_server
import sys
import io
from contextlib import redirect_stdout

# note: if using IPython, add the following lines to the top of the file:
# import nest_asyncio
# nest_asyncio.apply()

class Message:
    def __init__(self, string, data=None):
        self.string = string
        self.data = data

class Server:
    def __init__(self, one_iteration):
        self.last_result = None
        self.callbacks = {"run_command": self.onRunCommand,
                          "start": self.onStart,
                          "stop": self.onStop,
                          "get": self.onGet,
                          "quit": self.onQuit,
                          "set_data": self.onSetData}
        self.one_iteration = one_iteration
        self._processing_enabled = False
        self._server_task = None
    
    def on(self, name, callback):
        self.callbacks[name] = callback

    async def process(self):
        while True:
            if self._processing_enabled:
                self.last_result = self.one_iteration()
            await asyncio.sleep(0.001)
    def onRunCommand(self, command):
        try:
            with redirect_stdout(io.StringIO()) as f:
                exec(command,{'self':self})
        except Exception as e:
            return Message(f"error running command {command}", str(e))
        return Message(f"ran command: {command}", f.getvalue())
    
    def onStart(self):
        self._processing_enabled = True
        return Message("process started")

    def onStop(self):
        self._processing_enabled = False
        return Message("process stopped")
    
    def onGet(self):
        return Message("sent last result" , self.last_result)
    
    def onQuit(self):
        self._server_task.cancel()
        return Message("server stopped")
    
    def onSetData(self, data):
        self.last_result = data
        return Message("loaded data")
    
    async def handle_client(self):
        while True:
            buffer, addr = await self.server.recvfrom()
            message = pickle.loads(buffer)
            reply = self.callbacks[message.string](message.data) if message.data is not None else self.callbacks[message.string]()
            buf = pickle.dumps(reply)
            self.server.sendto(buf,addr)
            await asyncio.sleep(0.001)
            
    async def start_server(self):
        host = 'localhost'
        port = 12345
        
        self.server = await start_udp_server((host, port))
        print(f"Server listening on {host}:{port}...")
        
        # Start the task for generating numbers
        t1 = asyncio.create_task(self.process())
        t2 = asyncio.create_task(self.handle_client())
        await asyncio.gather(t1,t2)
        # Create the server
        
        

def random_number_generator():
    return random.randint(0, 100)

if __name__ == "__main__":
    server = Server(random_number_generator)
    # Run the server
    loop = asyncio.get_event_loop()
    task = loop.create_task(server.start_server())
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
    finally:
        loop.close()
