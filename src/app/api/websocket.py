# coding=utf-8
import asyncio
import datetime
import uuid

from fastapi import APIRouter, Depends, WebSocket
from gotrue import Session
from starlette.websockets import WebSocketDisconnect

from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from .deps import validate_user
from ..core import web_socket_manager, websocket_endpoint, WebSocketConnection
from ..core.config import logger, log_queue
from ..schemas import IdentifyResult, SystemStats, Face2SearchSchema, Face2Search
from ..utils.system_stats import cloud_system_stats

identify_router = APIRouter(prefix="/identify", tags=["identify"])





@identify_router.websocket("/identify/ws/{client_id}")
@websocket_endpoint(category="identify")
async def identify_ws(connection: WebSocketConnection, session: Session):
    while True:
        # test identifyResult
        try:
            rec_data = await connection.receive_data(Face2SearchSchema)
            search_data = Face2Search.from_schema(rec_data)
            logger.info(f"get the search data:{search_data}")

            time_now = datetime.datetime.now()
            result = IdentifyResult(
                id=str(uuid.uuid4()),
                name=session.user.user_metadata.get("name"),
                time=time_now.strftime("%Y-%m-%d %H:%M:%S"),
                uid=search_data.uid,
                score=0.99
            )
            await connection.send_data(result)
            await asyncio.sleep(1)  # 示例延时
        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError, WebSocketDisconnect) as e:
            logger.info(
                f"occurred error {e} Client #{session.user.id} left the chat")
            break


@identify_router.websocket("/cloud_logging/ws/{client_id}")
@websocket_endpoint(category="cloud_log")
async def cloud_logging_ws(connection: WebSocketConnection, session: Session):
    """ cloud_logging websocket"""
    while True:
        # test cloud_logging
        try:
            # logger.info(f"Client #{session.user.id} joined the chat")

            message: str = await log_queue.get()
            await asyncio.sleep(0.1)
            await web_socket_manager.broadcast(message, category="cloud_logging")
            # test cloud_logging

        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError, WebSocketDisconnect) as e:
            logger.info(
                f"occurred error {e} Client #{session.user.id} left the chat")
            break


@identify_router.websocket("/cloud_system_monitor/ws/{client_id}")
@websocket_endpoint(category="cloud_system_monitor")
async def cloud_system_monitor(connection: WebSocketConnection, session: Session):
    """ cloud_system_monitor websocket"""
    while True:
        try:
            message: str = SystemStats(cloud_system_stats).model_dump_json()
            await asyncio.sleep(1)
            await web_socket_manager.broadcast(message, category='cloud_system_monitor')

        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError, WebSocketDisconnect) as e:
            logger.info(
                f"occurred error {e} Client #{session.user.id} left the chat")
            break




@identify_router.websocket("/test/ws/{client_id}")
@websocket_endpoint(category="test")
async def test_connect(connection: WebSocketConnection, session: Session):
    """ cloud_system_monitor websocket"""
    while True:
        try:
            data = await connection.receive_data()
            logger.debug(f"test websocket receive data:{data}")
            await connection.send_data(data)
            logger.debug(f"test websocket send data:{data}")
        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError, WebSocketDisconnect) as e:
            logger.info(
                f"occurred error {e} Client {session.user.id} left the chat")
            break
