# coding=utf-8
"""
加速摄像头启动
cmd 运行setx OPENCV_VIDEOIO_PRIORITY_MSMF 0后重启，可以加快摄像头打开的速度
"""
from enum import Enum
from threading import Event
from time import sleep
from typing import NamedTuple

import cv2
from line_profiler_pycharm import profile

from .common import ClosableQueue, Image2Detect
from ..types import Image
from ..utils.decorator import thread_error_catcher


class CameraUrl(Enum):
    """
    url configs for camera
    """
    laptop: int = 0
    usb: int = 1
    ip: str = "http://"
    video: str = r"C:\Users\18317\OneDrive\python\BoostFace\src\boostface\db\data\test_01\video\Nola_Lyirs.mp4"


class CameraConfig(NamedTuple):
    """
    config for Camera
    """
    fps: int = 30
    resolution: tuple[int, ...] = (1920, 1080)
    url: CameraUrl = CameraUrl.video


class Camera:
    """
    read image from camera by opencv.VideoCapture.read() from the given url
    """

    def __init__(self, config: CameraConfig = CameraConfig()):
        """
        cmd 运行setx OPENCV_VIDEOIO_PRIORITY_MSMF 0后重启，可以加快摄像头打开的速度
        :param config: CameraOptions()
        """
        self.config = config
        self.videoCapture = None
        self._open()
        if config.url != CameraUrl.video:
            self._prepare()
        print(self)

    def read(self) -> Image:
        """
        read a Image from url by opencv.VideoCapture.read()
        :return: Image
        """
        ret, frame = self.videoCapture.read()
        if ret is None or frame is None:
            raise ValueError(
                f"in {self.__name__}.read()  self.videoCapture.read() get None")
        return frame

    def __repr__(self):
        """
        print camera info
        :return:
        """
        self.real_resolution = int(
            self.videoCapture.get(
                cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.videoCapture.get(
                cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧数
        self.real_fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))
        repr_string = (f"The video  codec  is {self.cap_codec_format}\n"
                       f"camera params = {self.config}")
        return repr_string

    def _prepare(self):
        """
        for usb or ip camera, set fps and resolution, not necessary for mp4
        :return:
        """
        #  设置帧数
        self.videoCapture.set(cv2.CAP_PROP_FPS, self.config.fps)
        # 设置分辨率
        self.videoCapture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.resolution[0])
        self.videoCapture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.resolution[1])

        # 设置视频编解码格式 note: 务必在set分辨率之后设置，否则不知道为什么又会回到默认的YUY2
        self.videoCapture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
        )

    def _open(self):
        """
        connect to camera  by url, raise ValueError if failed
        :return:
        """
        self.videoCapture = cv2.VideoCapture(self.config.url.value)
        if not self.videoCapture.isOpened():
            raise ConnectionError(
                f"Could not open video source from url: {self.config.url.value}")

    @property
    def cap_codec_format(self):
        """
        get current video codec format
        :return:
        """
        # 获取当前的视频编解码器
        fourcc = self.videoCapture.get(cv2.CAP_PROP_FOURCC)
        # 因为FOURCC编码是一个32位的值，我们需要将它转换为字符来理解它
        # 将整数编码值转换为FOURCC编码的字符串表示形式
        codec_format = "".join(
            [chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        return codec_format


class CameraTask(Camera):
    """
    generate Image2Detect into results queue from Camera instance may be in a thread
    """

    def __init__(self, results: ClosableQueue, camera_read_done: Event):
        super().__init__()
        self._results = results
        self.signal: Event = camera_read_done

    @profile
    @thread_error_catcher
    def run_camera(self):
        """
        read image from camera and put into results queue
        :return:
        """
        print("camera_read start")
        try:
            while not self.signal.is_set():
                frame: Image = self.read()
                if self.config.url == CameraUrl.video:
                    sleep(1 / self.config.fps)
                # print("camera_read get img")
                self._results.put(Image2Detect(image=frame, faces=[]))
        except Exception:
            print("camera_read error{}".format(Exception))
        finally:
            self.videoCapture.release()
