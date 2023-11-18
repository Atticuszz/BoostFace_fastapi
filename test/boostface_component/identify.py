# coding=utf-8
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import cv2
import numpy as np

from src.boostface.component.camera import CameraTask
from src.boostface.component.common import ClosableQueue, IdentifyManager
from src.boostface.component.detector import DetectorTask
from src.boostface.component.identifier import IdentifierTask, IdentifyWorker
from src.boostface.utils.decorator import thread_error_catcher
from src.boostface.utils.draw import draw_bbox, draw_text


@thread_error_catcher
def read_video(res_queue: ClosableQueue):
    print("read_video start")
    for img in res_queue:
        if img is None:
            Warning("read_video None")
            continue
        imgshow = img.nd_arr
        # TODO: 只会显示一个人脸
        if isinstance(imgshow, np.ndarray):
            for face in img.faces:
                draw_bbox(imgshow, face.bbox, (0, 255, 0))
                draw_text(imgshow, face.bbox, face.match_info.uid)
            cv2.imshow("video", imgshow)
        else:
            print("read_video None")
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        print("main process start")

        identifier_task_queue = manager.Queue(maxsize=100)
        identifier_result_dict = manager.dict()
        identifier_manager = IdentifyManager(
            identifier_task_queue, identifier_result_dict)
        print("created identifier_manager")

        worker = IdentifyWorker(identifier_task_queue, identifier_result_dict)
        mean_elapsed_times = []
        try:
            worker.start()
            print("created sub process")
            with ThreadPoolExecutor() as executor:
                src = ClosableQueue("camera read", maxsize=200)
                detected = ClosableQueue("detected", maxsize=200)
                identified = ClosableQueue("identified", maxsize=200)
                identify_task = IdentifierTask(detected, identified, identifier_manager)
                executor.submit(identify_task.run)
                # sleep(10)
                if_done = Event()

                my_camera = CameraTask(src, if_done)
                detect_task = DetectorTask(src, detected)

                executor.submit(my_camera.run)
                executor.submit(detect_task.run)

                executor.submit(read_video, identified)

                executor.shutdown(wait=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            worker.stop()
