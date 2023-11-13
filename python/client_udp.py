import asyncio
import pickle
from jetblack_datagram import open_udp_connection
import logging


class Message:
    def __init__(self, string, data=None):
        self.string = string
        self.data = data
class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.event_loop = None

    async def connect_and_send_single_message(self, message):
        try:
            # Connect to the server
            self.client = await open_udp_connection((self.host, self.port))
            self.send_single_message(message)
            self.client.close()
        except ConnectionRefusedError:
            print("Connection refused")
            return
        return message
    
    async def send_single_message(self, message):
            if isinstance(message, str):
                message = Message(message)
            try:
                # Prompt user for a message to send
                self.client.send(pickle.dumps(message))
                data = await asyncio.wait_for(self.client.recv(), timeout=5)
                message = pickle.loads(data)
                logging.debug(f"Message recieved: {message.string}")
                logging.debug(f"Data recieved: {message.data!r}")
            except asyncio.TimeoutError:
                print("Timed out waiting for response")
                return None
            except ConnectionRefusedError:
                print("Connection refused")
                return None
            return message
    
    def connect(self, loop=None):
        self.event_loop = loop
        if self.event_loop is None:
            self.client = asyncio.run(open_udp_connection((self.host, self.port)))
        else:
            self.client = self.event_loop.run_until_complete(open_udp_connection((self.host, self.port)))
        logging.debug("Connection successful" if self.client else "Connection refused")
    
    def emit(self,message,data=None) -> Message:
        if self.event_loop is None:
            return asyncio.run(self.send_single_message(Message(message,data)))
        else:
            return self.event_loop.run_until_complete(self.send_single_message(Message(message,data)))

    def emit_from_qt(self,message,data=None) -> Message:
        return

    async def close(self):
        self.client.close()
        logging.debug("Connection closed")
        return True
    
    
    def __del__(self):
        self.client.close()
        return True

if __name__ == "__main__":
    client = Client('localhost', 12345)
    asyncio.run(client.connect())
    # wait for key press
    