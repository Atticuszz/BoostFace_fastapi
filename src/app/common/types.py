# coding=utf-8
from multiprocessing import Queue
from typing import NamedTuple

from fastapi import WebSocket

from app.services.db.base_model import MatchedResult
from app.services.inference.common import TaskType, Face


class TypedWebSocket(NamedTuple):
    client_id: str
    category: str
    ws: WebSocket


task_queue = Queue(maxsize=100)  # Queue[tuple[TaskType, Face]
result_queue = Queue(maxsize=100)  # Queue[MatchedResult]
registered_queue = Queue(maxsize=100)  # Queue[str]
