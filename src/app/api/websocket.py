# coding=utf-8
import asyncio
import datetime

from fastapi import APIRouter, Depends, WebSocket

from .deps import validate_user
from ..common import ConnectionClosedOK, ConnectionClosedError, TypedWebSocket
from ..core import web_socket_manager
from ..core.config import logger, log_queue
from ..schemas import IdentifyResult, SystemStats
from ..utils import cloud_system_stats

identify_router = APIRouter(prefix="/identify", tags=["identify"])


@identify_router.websocket("/ws/{client_id}")
async def identify_ws(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    :param session:
    """
    typed_ws = TypedWebSocket(
        ws=websocket,
        client_id=client_id,
        category="identify")
    await web_socket_manager.connect(typed_ws)
    while True:
        # test identifyResult
        try:
            time_now = datetime.datetime.now()
            result = IdentifyResult(
                id=session.user.id,
                name=session.user.user_metadata.get("name"),
                time=time_now.strftime("%Y-%m-%d %H:%M:%S"),
            )
            await websocket.send_json(result.model_dump())
            await asyncio.sleep(1)  # 示例延时
            # test identifyResult

        except ConnectionClosedOK:
            await web_socket_manager.disconnect(typed_ws)
            break


@identify_router.websocket("/cloud_logging/ws/{client_id}")
async def cloud_logging_ws(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    """
    typed_ws = TypedWebSocket(
        ws=websocket,
        client_id=client_id,
        category="cloud_logging")
    await web_socket_manager.connect(typed_ws)
    while True:
        # test cloud_logging
        try:
            logger.info(f"Client #{client_id} joined the chat")

            message: str = await log_queue.get()
            await asyncio.sleep(0.1)
            await web_socket_manager.broadcast(message, category="cloud_logging")
            # test cloud_logging

        except (ConnectionClosedOK, ConnectionClosedError):
            await web_socket_manager.disconnect(typed_ws)
            break


@identify_router.websocket("/cloud_system_monitor/ws/{client_id}")
async def cloud_system_monitor(websocket: WebSocket, client_id: str, session=Depends(validate_user)):
    """
    :param client_id:
    :param websocket:
    """
    category = "cloud_system_monitor"
    typed_ws = TypedWebSocket(
        ws=websocket,
        client_id=client_id,
        category=category)
    await web_socket_manager.connect(typed_ws)
    while True:

        try:

            message: str = SystemStats(cloud_system_stats).model_dump_json()
            await asyncio.sleep(1)
            await web_socket_manager.broadcast(message, category=category)

        except (ConnectionClosedOK, ConnectionClosedError):
            await web_socket_manager.disconnect(typed_ws)
            break
