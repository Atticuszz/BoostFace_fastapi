# coding=utf-8
import asyncio
import logging

log_queue = asyncio.Queue()


class WebSocketHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        log_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        self.setFormatter(log_format)

    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put_nowait(log_entry)


handler = WebSocketHandler()
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # 设置日志级别
root_logger.addHandler(handler)  # 添加WebSocket处理器

# 现在记录一些日志
logger = logging.getLogger(__name__)
