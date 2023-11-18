import collections
import functools
import queue
from threading import Event
from timeit import default_timer as current_time

import cv2
from line_profiler_pycharm import profile
from turbojpeg import TurboJPEG

from .camera import Camera
from .detector import Detector
from .drawer import Drawer
from .identifier import Identifier

COST_TIME = {}
threads_done = Event()
streaming_event = Event()
image2web_deque = collections.deque(maxlen=3)
__all__ = ["MultiThreadFaceAnalysis", "COST_TIME", "threads_done", "streaming_event"]


def cost_time_recording(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = current_time()
        res = func(*args, **kwargs)
        end = current_time()
        COST_TIME.setdefault(func.__name__, []).append(end - start)
        return res

    return wrapper


class MultiThreadFaceAnalysis:
    def __init__(
            self,
            test_folder: str = "test_01",
            camera_params: dict = None,
            identifier_params: dict = None,
    ):
        self.show_times = 0
        camera_params = {
                            "app": "",
                            "approach": "",
                            "test_folder": test_folder,
                            "resolution": (1920, 1080),
                            "fps": 30,
                        } | camera_params
        identifier_params = {
                                "server_refresh": False,
                                "npz_refresh": False,
                                "flush_threshold": 1000,
                            } | identifier_params
        self.camera = Camera(**camera_params)  # http://192.168.0.102:4747/video
        self._detect = Detector()
        self._identifier = Identifier(
            self._detect, test_folder=test_folder, **identifier_params
        )
        self._screen = Drawer()

    def video_read(self, jobs: queue.Queue):
        print("video_read start")
        self.camera.read_video(jobs)
        return "video_read done"

    @profile
    def image2detect(self, jobs: queue.Queue, results: queue.Queue):
        print("detect_thread start")
        while not threads_done.is_set():
            try:
                # print('image2detect.qsize() = ', jobs.qsize())
                detect_job = jobs.get(timeout=10)

            except queue.Empty:
                print("detect_thread,queue.Empty")
                break
            else:
                image_2_show = self._detect(detect_job)
                results.put(image_2_show)
        return "image2detect done"

    @profile
    def detect2identify(self, jobs: queue.Queue, results: queue.Queue):
        print("detect2identify start")
        while not threads_done.is_set():
            try:
                # print(f'detect2identify.qsize() = {jobs.qsize()}')
                to_update = jobs.get(timeout=10)

                ide_res = self._identifier.identified_results(to_update)
                results.put(ide_res)

            except queue.Empty:
                print("detect2identify is empty")
                break
        return "detect2identify done"

    @profile
    def image2web(self, jobs: queue.Queue):
        print("image2web start")
        self.show_times = 0
        jpeg = TurboJPEG()
        while not threads_done.is_set():
            try:
                to_update = jobs.get(timeout=10)
                to_web = self._screen.show(to_update)
                # 没有请求之前不 捕获图像
                if streaming_event.is_set():
                    jpeg_bytes = jpeg.encode(to_web)
                    image2web_deque.append(jpeg_bytes)
                else:
                    cv2.imshow("screen", to_web)
                self.show_times += 1
            except queue.Empty:
                print("detect2identify is empty")
                break
        return "image2web done"

    @profile
    def image_show(self, jobs: queue.Queue):
        # from new_detector_fps import draw_bbox
        self.show_times = 0
        print("image_continued_show start")
        # redis_queue = redis.Redis(host="localhost", port=6379)
        while not threads_done.is_set():
            start = current_time()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                threads_done.set()
                break
            try:
                # print(f'finally_show_queue.qsize() = {jobs.qsize()}')
                to_update = jobs.get(timeout=15)
                image2show_nd_arr = self._screen.show(to_update)
                cv2.imshow('screen', image2show_nd_arr)
                # draw_bbox(to_update)
                self.show_times += 1
                end_time = current_time()
                sleep_time = (
                    1 / 30 - (end_time - start) if (end_time - start) < 1 / 30 else 0
                )
                if sleep_time == 0:
                    print(f"Warning: sleep_time = {sleep_time}")
                # sleep(sleep_time)

            except queue.Empty:
                print("finally_show_queue is empty")
                break

    def test_stop(self):
        self._identifier.stop_milvus()
        print("test_of_face_analysis ends!")
