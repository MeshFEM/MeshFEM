import asyncio
import pickle
from jetblack_datagram import open_udp_connection

class Message:
    def __init__(self, string, data=None):
        self.string = string
        self.data = data

class Client:
    def __init__(self, host, port, func):
        self.host = host
        self.port = port
        self.func = func

    async def connect_and_send_single_message(self, message):
        try:
            # Connect to the server
            self.client = await open_udp_connection((self.host, self.port))
            self.send_single_message(message)
            self.client.close()
        except ConnectionRefusedError:
            return
        return message
    
    async def send_single_message(self, message):
            if isinstance(message, str):
                message = Message(message)
            try:
                # Prompt user for a message to send
                self.client.send(pickle.dumps(message))
                data = await asyncio.wait_for(self.client.recv(), timeout=5)
                # data = await self.client.recv()
                message = pickle.loads(data)
                print(f"Message recieved: {message.string}")
                print(f"Data recieved: {message.data!r}")
            except asyncio.TimeoutError:
                print("Timed out waiting for response")
                return None
            except ConnectionRefusedError:
                print("Connection refused")
                return None
            return message
    
    async def connect(self):
        try:
            # Connect to the server
            self.client = await open_udp_connection((self.host, self.port))
        except ConnectionRefusedError:
            return
        return True
    
    async def close(self):
        self.client.close()
        return True
    
    def __del__(self):
        self.client.close()
        return True

if __name__ == "__main__":
    def func(reply_message):
        print(f"Received reply: {reply_message}")

    loop = asyncio.get_event_loop()
    
    client = Client('localhost', 12345, func)
    loop.create_task(client.event_loop())
    client.send_event("Event 1")
    # wait for key press
    