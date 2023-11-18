import multiprocessing
import queue
import uuid
from queue import Queue
from time import sleep
from timeit import default_timer
from typing import NamedTuple

import numpy as np
from line_profiler_pycharm import profile
from numpy.linalg import norm as l2norm

# from easydict import EasyDict
__all__ = ['Face', 'RawTarget', 'Target', 'ClosableQueue', 'LightImage']

from .sort_plus import KalmanBoxTracker
from ..types import Image, Bbox, Kps, Color, Embedding, MatchInfo


class LightImage(NamedTuple):
    nd_arr: np.ndarray
    # faces = [face, face, ...]
    faces: list[list] = []
    # face=[bbox:[4,2], kps:[5,2], det_score,colors,match_info]
    screen_scale: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __str__(self):
        return f"LightImage(nd_arr={self.nd_arr}, faces={self.faces})"


class Face(dict):
    def __init__(self, d=None, **kwargs):
        """

        :param d:
        :param kwargs:
        """
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            # 把k作为self的属性，并且v作为该属性的值，等效于self.k=v
            setattr(self, k, v)
        # Class attributes
        # for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def face_location(self):
        return self.bbox if self.bbox else None

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


class Face2Search(NamedTuple):
    """
    face to search, it is a  image filled with face
    """
    face_img: Image
    bbox: Bbox
    kps: Kps
    det_score: float


class FaceNew:
    def __init__(
            self,
            bbox: Bbox,
            kps: Kps,
            det_score: float,
            scense_scale: tuple[int, int, int, int],
            color: Color = (
                    50,
                    205,
                    255),
            match_info: MatchInfo = MatchInfo(0.0, '')):
        """
        init a face
        :param bbox:shape [4,2]
        :param kps: shape [5,2]
        :param det_score:
        :param color:
        :param scense_scale: (x1,y1,x2,y2) of scense image
        :param match_info: MatchInfo(uid,score)
        """
        self.bbox: Bbox = bbox
        self.kps: Kps = kps
        self.det_score: float = det_score
        self.scense_scale: tuple[int, int, int, int] = scense_scale
        # 默认是橙色
        self.bbox_color: Color = color
        self.embedding: Embedding = np.zeros(512)
        self.id: int = 0  # target id
        self.match_info: MatchInfo = match_info

    @property
    def embedding_norm(self) -> float:
        if self.embedding is None:
            raise ZeroDivisionError('embedding is None')
        return l2norm(self.embedding)

    @property
    def normed_embedding(self) -> np.ndarray[512]:
        if self.embedding is None:
            raise ZeroDivisionError('embedding is None')
        return self.embedding / self.embedding_norm

    def update_match_info(self, match_info: MatchInfo):
        self.match_info = match_info

    def face_image(self, scense: Image) -> Face2Search:
        """
        get face image from scense
        :param scense:
        :return:
        """
        # 确保 bbox 中的值是整数
        x1, y1, x2, y2 = map(int, [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]])

        # 避免超出图像边界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(scense.shape[1], x2)  # scense.shape[1] 是图像的宽度
        y2 = min(scense.shape[0], y2)  # scense.shape[0] 是图像的高度

        # 裁剪人脸图像
        face_img = scense[y1:y2, x1:x2]
        bbox = np.array([0, 0, face_img.shape[1], face_img.shape[0]])

        # 调整关键点位置
        kps = self.kps - np.array([x1, y1])

        return Face2Search(face_img, bbox, kps, self.det_score)


class Image2Detect:
    """
    image to detect
    :param image: image
    :param faces: [face, face, ...]
    """

    def __init__(self, image: Image, faces: list[FaceNew]):
        self.nd_arr: Image = image
        self.faces: list[FaceNew] = faces

    @property
    def scale(self) -> tuple[int, int, int, int]:
        """
        :return: (x1, y1, x2, y2)
        :return:
        """
        return 0, 0, self.nd_arr.shape[1], self.nd_arr.shape[0]

    def __str__(self):
        return f"LightImage(nd_arr={self.nd_arr}, faces={self.faces})"


class RawTarget(NamedTuple):
    id: int
    bbox: np.ndarray[4]
    kps: np.ndarray[5, 2]
    score: float = 0.0


class Target:
    def __init__(self,
                 face: FaceNew,
                 score: float = 0.0):

        self._hit_streak = 0  # frames of keeping existing in screen
        self._time_since_update = 0  # frames of keeping missing in screen
        self.face: FaceNew = face
        self.frames_since_reced: int = 0
        self._tracker: KalmanBoxTracker = KalmanBoxTracker(face.bbox)
        self.normed_embedding: np.ndarray[512] = np.zeros(512)

    def update_pos(self, bbox: np.ndarray, kps: np.ndarray, score: float):
        self.face.bbox = bbox
        self.face.kps = kps
        self.face.det_score = score

    def update_tracker(self, detect_tar: RawTarget):
        self._time_since_update = 0
        self._tracker.update(detect_tar.bbox)

    def old_enough(self, max_age: int) -> bool:
        return self._time_since_update > max_age

    def in_screen(self, min_hits: int) -> bool:
        return self._time_since_update < 1 and self._hit_streak >= min_hits

    @property
    def get_predicted_tar(self) -> FaceNew:
        # get predicted bounding box from Kalman Filter
        if self._tracker is None:
            raise ValueError('tracker is None')
        bbox = self._tracker.predict()[0]
        # if failed to update before predicted bbox, reset the hit_streak
        # coming after the update_tracker is meaning that the target is  in
        # screen continuously
        if self._time_since_update == 0:
            self._hit_streak += 1
        else:
            self._hit_streak = 0
        self._time_since_update += 1
        perdicted_face: FaceNew = FaceNew(
            bbox=bbox,
            kps=self.face.kps,
            det_score=self.face.det_score,
            scense_scale=self.face.scense_scale)
        return perdicted_face

    @property
    def screen_height(self):
        return self.face.scense_scale[3] - self.face.scense_scale[1]

    @property
    def screen_width(self):
        return self.face.scense_scale[2] - self.face.scense_scale[0]

    @property
    def bbox_width(self):
        return self.face.bbox[2] - self.face.bbox[0]

    @property
    def bbox_height(self):
        return self.face.bbox[3] - self.face.bbox[1]

    @property
    def name(self):
        return f'target[{self.face.id}]'

    @property
    def time_satified(self) -> bool:
        if not self.if_matched:
            return False
        elif self.frames_since_reced < 100:
            self.frames_since_reced += 1
            return False
        else:
            print(self.frames_since_reced)
            self.frames_since_reced = 0
            return True

    @property
    def scale_satified(self) -> bool:
        target_area = self.bbox_width * self.bbox_height
        screen_area = self.screen_height * self.screen_width
        return (target_area / screen_area) > 0.03

    @property
    def if_matched(self) -> bool:
        return self.face.match_info.uid is not None

    @property
    def rec_satified(self) -> bool:
        if self.scale_satified and not self.if_matched and self.in_screen(3):
            return True
        elif self.if_matched and self.scale_satified and self.time_satified and self.in_screen(3):
            return True
        else:
            return False

    @property
    def colors(self):
        red = (0, 0, 255)
        yellow = (50, 205, 255)
        green = (152, 251, 152)
        if self.if_matched:
            # 有匹配对象
            if self.face.match_info.score > 0.4:
                bbox_color = green
                name_color = green
            else:
                # 有匹配对象，但是匹配分数不够，定义为匹配失败的红色
                bbox_color = red
                name_color = red
        else:  # 还没有匹配到对象
            bbox_color = yellow
            name_color = yellow
        return bbox_color, name_color


class ClosableQueue(Queue):
    def __init__(self, task_name: str, maxsize: int = 100):
        super().__init__(maxsize=maxsize)
        self.task_name = task_name

    def __iter__(self):
        try:
            while True:
                # print("task_name:", self.task_name,self.qsize())
                item = self.get(timeout=10000)
                yield item
        except queue.Empty:
            raise StopIteration
        finally:
            print(
                f"{self.task_name} queue wait for 5 sec got none,so close it")


# TODO: test IdentifyManager class
class IdentifyManager:
    """
    be shared for add ,get
    """

    def __init__(self, task_queue: multiprocessing.Queue, result_dict: dict):
        self.task_queue = task_queue
        self.result_dict = result_dict
        self._cost_time = {}
        print("IdentifyManager init done")

    @profile
    def add_task(self,
                 task: Face2Search,
                 timeout: int = 5):
        try:
            start_time = default_timer()
            task_id: uuid = uuid.uuid4()
            self.task_queue.put((task_id, task), timeout=timeout)
            self._cost_time.setdefault(
                'add_task', []).append(
                default_timer() - start_time)
            return task_id
        except queue.Full:
            raise queue.Full("The task queue is full. Try again later.")

    @profile
    def get_result(self, task_id: str, timeout: int = 100) -> MatchInfo:
        """
        get result from working process
        :param task_id:
        :param timeout:
        :return:  result of task_id ,it represent the uuid of matched face
        """
        # 尝试获取结果，如果结果尚未就绪，等待后重试
        start_time = default_timer()
        while True:
            get_start_time = default_timer()
            res = self.result_dict.get(task_id, None)
            if res is not None:
                self._cost_time.setdefault(
                    'get_result', []).append(
                    default_timer() - get_start_time)
                return res
            elif default_timer() - start_time > timeout:
                raise queue.Empty(
                    f"Timeout while waiting for the result of task {task_id}")
            sleep(0.001)  # 避免过于频繁的查询

    @property
    def cost_time(self):
        cost_time = {
            key: sum(costs) / len(costs) for key,
            costs in self._cost_time.items() if len(costs) > 0}
        return cost_time


camera_2_detect_queue = ClosableQueue("camera_2_detect", maxsize=200)
detect_2_rec_queue = ClosableQueue("detect_2_rec", maxsize=200)
rec_2_draw_queue = ClosableQueue("rec_2_draw", maxsize=400)
