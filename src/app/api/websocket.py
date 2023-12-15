# coding=utf-8
import asyncio
import datetime

from fastapi import APIRouter
from gotrue import Session
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from ..core import web_socket_manager, websocket_endpoint, WebSocketConnection
from ..core.config import logger, log_queue
from ..schemas import IdentifyResult, SystemStats, Face2SearchSchema, Face2Search
from ..utils.system_stats import cloud_system_stats

identify_router = APIRouter(prefix="/identify", tags=["identify"])


# TODO: websocket decorator


@identify_router.websocket("/identify/ws/{client_id}")
@websocket_endpoint(category="identify")
async def identify_ws(connection: WebSocketConnection, session: Session):
    while True:
        # TODO: handle face images
        # test identifyResult
        try:
            rec_data = await connection.receive_data(Face2SearchSchema)
            search_data = Face2Search.from_schema(rec_data)
            logger.info(f"get the search data:{search_data}")

            time_now = datetime.datetime.now()
            result = IdentifyResult(
                id=session.user.id,
                name=session.user.user_metadata.get("name"),
                time=time_now.strftime("%Y-%m-%d %H:%M:%S"),
            )
            await connection.send_data(result)
            await asyncio.sleep(1)  # 示例延时
        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError) as e:
            logger.info(
                f"occurred error {e} Client #{session.user.id} left the chat")
            break


@identify_router.websocket("/cloud_logging/ws/{client_id}")
@websocket_endpoint(category="cloud_logging")
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

        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError) as e:
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

        except (ConnectionClosedOK, ConnectionClosedError, RuntimeError) as e:
            logger.info(
                f"occurred error {e} Client #{session.user.id} left the chat")
            break
