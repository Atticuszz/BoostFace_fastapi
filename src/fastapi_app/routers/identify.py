import asyncio
import datetime

from fastapi import APIRouter, WebSocket, Depends
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from fastapi_app.dependence import validate_user
from fastapi_app.internal import identifyResult
from fastapi_app.internal import logger, log_queue
from fastapi_app.internal.client import web_socket_manager

identify_router = APIRouter(prefix="/identify", tags=["identify"])


@identify_router.websocket("/ws/{client_id}")
async def identify_ws(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    :param session:
    """
    await web_socket_manager.connect(websocket, client_id, category="identify")
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
        await web_socket_manager.disconnect(client_id)
        await web_socket_manager.broadcast(f"Client #{client_id} left the chat")


@identify_router.websocket("/cloud_logging/ws/{client_id}")
async def cloud_logging_ws(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    """
    await web_socket_manager.connect(websocket, client_id, category="cloud_logging")
    try:
        while True:
            # test cloud_logging
            logger.info(f"Client #{client_id} joined the chat")

            message: str = await log_queue.get()
            await asyncio.sleep(2)
            await web_socket_manager.broadcast(message, category="cloud_logging")
            # test cloud_logging

    except (ConnectionClosedOK, ConnectionClosedError) as e:
        await web_socket_manager.disconnect(client_id)
        await web_socket_manager.broadcast(f"Client #{client_id} left the chat, {e}")
