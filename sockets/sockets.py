from socketio import Client

class InitalizeSocket:
    def __init__(self):
        self.socket: Client = None

    def initialize(self):
        if self.socket is None:
            self.socket = Client()
        return self.socket

    def getSocket(self) -> Client:
        if self.socket is None:
            raise RuntimeError("Socket not initialized. Call initialize() first.")
        return self.socket

# Singleton instance
socketinstance = InitalizeSocket()
