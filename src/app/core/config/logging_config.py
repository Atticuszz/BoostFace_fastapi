# coding=utf-8
"""
config stream handler and websocket handler for root logger
"""
import asyncio
import logging

log_queue = asyncio.Queue()

log_format = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')


class WebSocketHandler(logging.Handler):
    """websocket log handler"""

    def __init__(self):
        super().__init__()
        self.setFormatter(log_format)

    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put_nowait(log_entry)


ws_handler = WebSocketHandler()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(ws_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger()
