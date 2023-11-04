import concurrent
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer

import numpy as np
from line_profiler_pycharm import profile

from src.boostface.app.common import IdentifyManager
from src.boostface.app.identifier import IdentifyWorker
from test import generate_light_image


@profile
def process_image(images, identifier_manager):
    elapsed = []
    for image in images:
        start = default_timer()
        uuid = identifier_manager.add_task(
            (image, image.faces[0][0], image.faces[0][1], image.faces[0][2]))
        result = identifier_manager.get_result(uuid)
        elapsed.append(default_timer() - start)
        print("task:", uuid, "result:", result)
    return elapsed


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        print("main process start")

        identifier_task_queue = manager.Queue(maxsize=100)
        identifier_result_dict = manager.dict()
        identifier_manager = IdentifyManager(
            identifier_task_queue, identifier_result_dict)
        print("created identifier_manager")

        worker = IdentifyWorker(identifier_task_queue, identifier_result_dict)
        try:
            worker.start()
            print("created sub process")

            # 创建虚拟数据
            fake_img = [generate_light_image(size=(640, 640)) for _ in range(20)]
            # 使用线程池处理图像
            elapsed_time = []
            # 模拟5个线程同时喂给处理进程
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_image, fake_img, identifier_manager) for _ in range(1)]
                for future in concurrent.futures.as_completed(futures):
                    elapsed = future.result()
                    elapsed_time.append(np.mean(elapsed))

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            # 打印耗时
            print("mean:", np.mean(elapsed_time))
            worker.stop()
