import multiprocessing
import queue
import uuid
from queue import Queue
from timeit import default_timer
from typing import NamedTuple

import numpy as np
from line_profiler_pycharm import profile
from numpy.linalg import norm as l2norm
from time import sleep

from boostface.types import Image, Bbox, Kps, Color, Embedding, MatchInfo
from .sort_plus import KalmanBoxTracker


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
    face to search, it is a image filled with face for process transfer
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
            scene_scale: tuple[int, int, int, int],
            face_id: int = 0,
            color: Color = (50, 205, 255),
            match_info: MatchInfo = MatchInfo(0.0, ''),
    ):
        """
        init a face
        :param bbox:shape [4,2]
        :param kps: shape [5,2]
        :param det_score:
        :param color:
        :param scene_scale: (x1,y1,x2,y2) of scense image
        :param match_info: MatchInfo(uid,score)
        """
        self.bbox: Bbox = bbox
        self.kps: Kps = kps
        self.det_score: float = det_score
        self.scense_scale: tuple[int, int, int, int] = scene_scale
        # 默认是橙色
        self.bbox_color: Color = color
        self.embedding: Embedding = np.zeros(512)
        self.id: int = face_id  # target id
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
        x1, y1, x2, y2 = map(
            int, [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]])

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
        """
        return 0, 0, self.nd_arr.shape[1], self.nd_arr.shape[0]


class Target:
    """
    :param face: FaceNew
    :ivar _hit_streak: frames of keeping existing in screen
    :ivar _frames_since_update: frames of keeping missing in screen
    :ivar face: FaceNew
    :ivar _frames_since_identified: frames since reced
    :ivar _tracker: KalmanBoxTracker
    :ivar normed_embedding: Embedding
    """

    def __init__(self, face: FaceNew):

        self._hit_streak = 0  # frames of keeping existing in screen
        self._frames_since_update = 0  # frames of keeping missing in screen
        self._frames_since_identified = 0
        self.face: FaceNew = face
        self._tracker: KalmanBoxTracker = KalmanBoxTracker(face.bbox)
        self.normed_embedding: Embedding = np.zeros(512)

    @property
    def rec_satified(self) -> bool:
        if self._scale_satisfied and not self._if_matched and self.in_screen:
            return True
        elif self._if_matched and self._scale_satisfied and self._time_satisfied and self.in_screen:
            return True
        else:
            return False

    @property
    def colors(self):
        """
        state color
        :return:
        """
        red = (0, 0, 255)
        yellow = (50, 205, 255)
        green = (152, 251, 152)
        if self._if_matched:
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

    def update_pos(self, bbox: Bbox, kps: Kps, score: float):
        self.face.bbox = bbox
        self.face.kps = kps
        self.face.det_score = score

    def update_tracker(self, bbox: Bbox):
        """
        update tracker with bbox, and update state of continuation
        """
        self._frames_since_update = 0
        self._hit_streak += 1
        self._tracker.update(bbox)

    def unmatched(self):
        """
        update state of continuation
        :return:
        """
        self._frames_since_update += 1
        self._hit_streak = 0

    def old_enough(self, max_age: int) -> bool:
        """
        if the target is too old ,should be del
        """
        return self._frames_since_update > max_age

    @property
    def in_screen(self) -> bool:
        """
        if the target is in screen should be satisfied min_hits,forbid the shiver
        """
        min_hits = 3  # almost 0.1s if fps=30
        return self._hit_streak >= min_hits

    @property
    def get_predicted_tar(self) -> FaceNew:
        """
        get predicted FaceNew by tracker
        :return:
        """
        # get predicted bounding box from Kalman Filter
        bbox = self._tracker.predict()[0]
        predicted_face: FaceNew = FaceNew(
            bbox,
            self.face.kps,
            self.face.det_score,
            face_id=self.face.id,
            scene_scale=self.face.scense_scale)
        return predicted_face

    @property
    def name(self) -> str:
        return f'target[{self.face.id}]'

    @property
    def _time_satisfied(self) -> bool:
        """
        Checks if the time(frames) elapsed since the target was last identified exceeds a predefined threshold.
        """
        frames_threshold = 100  # almost 3 sec if fps=30
        if not self._if_matched:
            return False
        elif self._frames_since_identified < frames_threshold:
            self._frames_since_identified += 1
            return False
        else:
            self._frames_since_identified = 0
            return True

    @property
    def _scale_satisfied(self) -> bool:
        """
        if the scale of target is satisfied
        """
        scale_threshold = 0.03
        target_area = (self.face.bbox[2] - self.face.bbox[0]) * \
                      (self.face.bbox[3] - self.face.bbox[1])
        screen_area = (self.face.scense_scale[3] - self.face.scense_scale[1]) * (
                self.face.scense_scale[2] - self.face.scense_scale[0])
        return (target_area / screen_area) > scale_threshold

    @property
    def _if_matched(self) -> bool:
        return self.face.match_info.uid is not None


class ClosableQueue(Queue):
    """
    A Queue that can be closed.
    :param task_name: name of the task
    :param maxsize: maxsize of the queue
    :param wait_time: wait time for get,if over this time,close the queue for iteration
    """

    def __init__(self, task_name: str, maxsize: int = 100, wait_time: int = 5):
        super().__init__(maxsize=maxsize)
        self._task_name = task_name
        self._closed = False
        self._wait_time = wait_time

    def __iter__(self):
        """
        rewrite the iter method to auto cancel the iter when queue is empty for self._wait_time
        :return:
        """
        while not self._closed:
            try:
                # 设置更合理的超时时间，例如5秒
                item = self.get(timeout=5)
                yield item
            except queue.Empty:
                print(
                    f"{self._task_name} queue wait for 5 sec got none, so close it")
                self.closed = True
                break


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
