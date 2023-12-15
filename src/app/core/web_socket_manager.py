# coding=utf-8
from contextlib import asynccontextmanager
from typing import Type

from pydantic import BaseModel
from starlette.websockets import WebSocketState
from fastapi import WebSocket, Depends
from .config import logger

from ..common import TypedWebSocket


class WebSocketManager:
    """ WebSocket manager"""

    def __init__(self):
        self.active_connections: list[TypedWebSocket] = []

    @asynccontextmanager
    async def handle_connection(self, websocket: WebSocket, client_id: str, category: str):
        """Handle connection."""
        typed_ws = TypedWebSocket(
            ws=websocket,
            client_id=client_id,
            category=category)
        await self.connect(typed_ws)
        try:
            yield typed_ws
        finally:
            await self.disconnect(typed_ws)

    async def connect(self, typed_websocket: TypedWebSocket):
        """Connect with category."""
        await typed_websocket.ws.accept()
        self.active_connections.append(typed_websocket)

    async def disconnect(self, typed_websocket: TypedWebSocket):
        """Disconnect."""
        await typed_websocket.ws.close()
        self.active_connections.remove(typed_websocket)
        logger.info(
            f"category:{typed_websocket.category}  client_id:{typed_websocket.client_id} Connection closed")

    async def broadcast(self, message: str, category: str = None):
        """Broadcast message to all connections or specific category.
        :exception ConnectionClosedOK, ConnectionClosedError
        """
        for typed_ws in self.active_connections:
            # note: the active connection may be closed ,we need to check it
            if (category is None or typed_ws.category ==
                category) and typed_ws.ws.client_state == WebSocketState.CONNECTED:
                await typed_ws.ws.send_text(message)


class WebSocketConnection:
    """auto handel data send and receive"""

    def __init__(self, typed_websocket: TypedWebSocket):
        self.typed_websocket = typed_websocket

    async def send_data(self, data: BaseModel | str):
        """Send data as JSON or str
        :exception TypeError
        """
        if isinstance(data, BaseModel):
            await self.typed_websocket.ws.send_json(data.model_dump_json())
        elif isinstance(data, str):
            await self.typed_websocket.ws.send_text(data)
        else:
            logger.warn(f"send_data: data:{data} failed")
            raise TypeError("data must be BaseModel or str")

    async def receive_data(self, data_model: Type[BaseModel] | None = None) -> BaseModel | str:
        """Receive and decode data.
        :exception TypeError
        """
        if issubclass(data_model, BaseModel):
            received_data = await self.typed_websocket.ws.receive_text()
            return data_model.model_validate_json(received_data)
        elif data_model is None:
            received_data = await self.typed_websocket.ws.receive_text()
            return received_data
        else:
            logger.warn(
                f"receive_data: data_model:{data_model} received_data: None failed")
            raise TypeError(
                "data_model must be a subclass of BaseModel or None")


def websocket_endpoint(category: str):
    """Decorator for websocket endpoints."""
    from ..api.deps import validate_user

    def decorator(func):
        async def wrapper(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
            async with web_socket_manager.handle_connection(websocket, client_id, category) as typed_ws:
                connection = WebSocketConnection(typed_ws)
                return await func(connection, session)

        return wrapper

    return decorator


web_socket_manager = WebSocketManager()
