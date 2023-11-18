import collections
import heapq
import multiprocessing
import queue
import uuid
from multiprocessing import Process
from pathlib import Path

import numpy as np
from line_profiler_pycharm import profile

from .common import Face, Target, IdentifyManager, Image2Detect, FaceNew, Face2Search, ClosableQueue
from .drawer import streaming_event
from .sort_plus import associate_detections_to_trackers
from ..db.operations import Matcher
from ..model_zoo import get_model, ArcFaceONNX
from ..types import Image
from ..utils.decorator import thread_error_catcher

matched_and_in_screen_deque = collections.deque(maxlen=1)


class Extractor:
    """
    extract face embedding from given target bbox and kps, and det_score by running model in onnx
    """

    def __init__(self):
        root: Path = Path(__file__).parents[1].joinpath(
            'model_zoo\\models\\insightface\\irn50_glint360k_r50.onnx')
        self.rec_model: ArcFaceONNX = get_model(root,
                                                providers=(
                                                    'CUDAExecutionProvider',
                                                    'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def __call__(self,
                 img2extract: Image,
                 bbox: np.ndarray[4, 2],
                 kps: np.ndarray[5, 2],
                 det_score: float) -> np.ndarray[512]:
        """
        get embedding of face from given target bbox and kps, and det_score
        :param img2extract: target at which image
        :param bbox: target bbox
        :param kps: target kps
        :param det_score: target det_score
        :return: face embedding
        """
        face = Face(bbox=bbox,
                    kps=kps,
                    det_score=det_score)
        self.rec_model.get(img2extract, face)
        return face.normed_embedding



def identify_works(
        task_queue: multiprocessing.Queue,
        result_dict: dict,
        stop_event: multiprocessing.Event):
    """
    long term works in single process
    1. get target from worker_queue
    2. extract embedding
    3. search in milvus by embedding
    4. put result in result_queue
    :param stop_event:
    :param result_dict:
    :param task_queue:
    :return:
    """
    matcher = Matcher()
    extractor = Extractor()
    try:
        while not stop_event.is_set():
            try:
                item = task_queue.get(timeout=1)
                task_id: uuid = item[0]
                task: Face2Search = item[1]
                # start = default_timer()
                emmbedding = extractor(
                    task.face_img, task.bbox, task.kps, task.det_score)
                # print(f'extractor cost time: {default_timer() - start}')
                # start = default_timer()
                result_dict[task_id] = matcher(emmbedding)
            except queue.Empty:
                continue  # 这不是一个错误条件，只是队列暂时为空
            except Exception as e:
                print(
                    f"An error occurred while processing the{task_id} task: {e}")
                # # 可以选择是否将异常信息放入结果字典
                # result_dict[task_id] = str(e)
    finally:
        matcher.shut_down()


# TODO: test IdentifyWorker class


class IdentifyWorker(Process):
    """
    solo worker for extractor and then search by milvus
    """

    def __init__(
            self,
            task_queue: multiprocessing.Queue,
            result_dict: dict,
    ):
        self._stop_event = multiprocessing.Event()
        super().__init__(
            target=identify_works,
            daemon=True,
            args=(
                task_queue,
                result_dict,
                self._stop_event))

    def start(self):
        super().start()
        print("IdentifyWorker start")

    def stop(self):
        self._stop_event.set()
        super().join()
        print("IdentifyWorker stop")


class Identifier:
    def __init__(
            self,
            manager: IdentifyManager,
            max_age=10,
            min_hits=1,
            iou_threshold=0.3,
    ):
        """
        :param max_age: 超过这个帧数没被更新就删除
        :param min_hits: 超过这个帧数 才会被 识别
        :param iou_threshold: 卡尔曼滤波器的阈值
        :param manager: 识别器的manager
        """
        self._targets: dict[int, Target] = {}
        self.max_age = max_age  # 超过该帧数没被更新就删除
        self.min_hits = min_hits  # 至少被检测到的次数才算
        self.iou_threshold = iou_threshold
        self._recycled_ids = []
        self._frame_cnt = 1

        self._identifier_manager: IdentifyManager = manager

    @profile
    @thread_error_catcher
    def identified_results(self, image2identify: Image2Detect) -> Image2Detect:
        """
        :param image2identify:
        :return: get image2identify match info
        """
        self._update(image2identify)
        self._search(image2identify)
        if self._frame_cnt < 100000:
            self._frame_cnt += 1
        else:
            self._frame_cnt = 0
        # [tar.face.match_info for tar in self._targets.values()]

        return Image2Detect(image2identify.nd_arr,
                            [tar.face for tar in self._targets.values() if tar.in_screen(self.min_hits)])

    def _send2web(self, new_targets: list[dict]):

        if streaming_event.is_set():
            matched_and_in_screen_deque.append(new_targets)

    @thread_error_catcher
    def _search(self, image2identify: Image2Detect):
        """
        search in a process and then update face.match_info
        :param image2identify:
        """
        for tar in self._targets.values():
            if tar.rec_satified:
                uuid = self._identifier_manager.add_task(tar.face.face_image(image2identify.nd_arr))
                tar.face.update_match_info(self._identifier_manager.get_result(uuid))

    @profile
    @thread_error_catcher
    def _update(self, image2update: Image2Detect):
        """
        according to the "memory" in Kalman tracker update former targets info by Hungarian algorithm
        :param image2update:
        :return:
        """
        # 更新目标

        detected_tars: list[FaceNew] = image2update.faces
        # 第一次的时候，直接添加
        if self._targets:
            # 提取预测的位置
            predicted_tars: list[FaceNew] = []
            to_del: list[int] = []
            for i, tar in enumerate(self._targets.values()):
                raw_tar: FaceNew = tar.get_predicted_tar
                predicted_tars.append(raw_tar)
                pos = raw_tar.bbox
                if np.any(np.isnan(pos)):
                    to_del.append(raw_tar.id)
            #  根据预测的位置清空即将消失的目标
            for k in to_del:
                assert k in self._targets, f'k = {k} not in self._targets'
                heapq.heappush(self._recycled_ids, k)
                del self._targets[k]

            # 根据预测的位置和检测的  **targets**  进行匹配
            matched, unmatched_det_tars, unmatched_pred_tars = associate_detections_to_trackers(
                detected_tars, predicted_tars, self.iou_threshold)
            # update pos and tracker for matched targets
            for pred_tar, detected_tar in matched:
                self._targets[pred_tar.id].update_pos(
                    detected_tar.bbox, detected_tar.kps, detected_tar.det_score)
                self._targets[pred_tar.id].update_tracker(detected_tar)
        else:
            unmatched_det_tars: list[FaceNew] = detected_tars
        # add new targets
        for detected_tar in unmatched_det_tars:
            new_id = self._generate_id()
            assert new_id not in self._targets, f'{new_id} is already in self._targets'
            detected_tar.id = new_id
            self._targets[new_id] = Target(face=detected_tar)
            # print(f'new target id = {new_id}')
        self._clear_old_targets()

    def _clear_old_targets(self):
        # clear dead targets
        keys = []
        for tar in self._targets.values():
            # remove dead targets
            if tar.old_enough(self.max_age):
                keys.append(tar.face.id)
        for k in keys:
            try:
                del self._targets[k]
            except KeyError:
                print(f'KeyError: tar.id = {k}')
            else:
                heapq.heappush(self._recycled_ids, k)

    def _generate_id(self) -> int:

        try:
            return heapq.heappop(self._recycled_ids)

        except IndexError:
            return len(self._targets)


class IdentifierTask(Identifier):
    def __init__(
            self,
            jobs: ClosableQueue,
            results: ClosableQueue,
            manager: IdentifyManager):
        """
        Args:
            jobs:
            results:
            identifier_params: { manager:IdentifyManager,
                                 max_age: int = 120,
                                 min_hits: int = 3,
                                 iou_threshold: float = 0.3,}
        """
        super().__init__(manager)
        self.jobs = jobs
        self.results = results

    @profile
    @thread_error_catcher
    def run(self):
        for img in self.jobs:
            identified = self.identified_results(img)
            self.results.put(identified)

        return "IdentifierTask Done"
