# coding=utf-8

from .config import logger
from ..common import TypedWebSocket, WebSocketState


class WebSocketManager:
    """ WebSocket manager"""

    def __init__(self):
        self.active_connections: list[TypedWebSocket] = []

    async def connect(self, typed_websocket: TypedWebSocket):
        """Connect with category."""
        await typed_websocket.ws.accept()
        self.active_connections.append(typed_websocket)

    async def disconnect(self, typed_websocket: TypedWebSocket):
        """Disconnect."""
        await typed_websocket.ws.close()
        self.active_connections.remove(typed_websocket)
        logger.info(f"category:{typed_websocket.category}  client_id:{typed_websocket.client_id} Connection closed")

    async def broadcast(self, message: str, category: str = None):
        """Broadcast message to all connections or specific category.
        :exception ConnectionClosedOK, ConnectionClosedError
        """
        for typed_ws in self.active_connections:
            # note: the active connection may be closed ,we need to check it
            if (
                    category is None or typed_ws.category == category) and typed_ws.ws.client_state == WebSocketState.CONNECTED:
                await typed_ws.ws.send_text(message)


web_socket_manager = WebSocketManager()
