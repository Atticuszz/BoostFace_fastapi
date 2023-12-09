import asyncio
import datetime

from fastapi import APIRouter, WebSocket, Depends
from websockets.exceptions import ConnectionClosedOK

from fastapi_app.dependence import validate_user
from fastapi_app.model import identifyResult

identify_router = APIRouter(prefix="/identify", tags=["identify"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """connect"""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    async def disconnect(self, client_id: str):
        """disconnect"""
        await self.active_connections[client_id].close()
        del self.active_connections[client_id]

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


manager = ConnectionManager()


@identify_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    :param session:
    :return:
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            # test identifyResult
            time_now = datetime.datetime.now()
            result = identifyResult(
                id=session.user.id,
                name=session.user.user_metadata.get("name"),
                time=time_now.strftime("%Y-%m-%d %H:%M:%S"),
            )
            await websocket.send_json(result.model_dump())
            await asyncio.sleep(1)  # 示例延时
            # test identifyResult




    except ConnectionClosedOK:
        await manager.disconnect(client_id)
        await manager.broadcast(f"Client #{client_id} left the chat")
