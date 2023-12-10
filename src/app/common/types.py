# coding=utf-8
from typing import NamedTuple

from fastapi import WebSocket


class TypedWebSocket(NamedTuple):
    client_id: str
    category: str
    ws: WebSocket
