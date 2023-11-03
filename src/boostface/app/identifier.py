"""


"""

import collections
import heapq
import multiprocessing
import queue
from multiprocessing import Process
from pathlib import Path

import numpy as np
from line_profiler_pycharm import profile

# from memory_profiler import profile
from database.milvus_standalone.common import MatchInfo
from .common import Face, RawTarget, Target, ClosableQueue, IdentifyManager
from .common import LightImage
from .detector import Detector, detect_task
from .drawer import streaming_event
from .matcher import Matcher
from .sort_plus import associate_detections_to_trackers

matched_and_in_screen_deque = collections.deque(maxlen=1)


class Extractor:
    """
    extract face embedding from given target bbox and kps, and det_score by running model in onnx
    """

    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path(__file__).parents[1].joinpath(
            'model_zoo\\models\\insightface\\irn50_glint360k_r50.onnx')
        self.rec_model = get_model(
            root,
            providers=(
                'CUDAExecutionProvider',
                'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def __call__(self,
                 img2extract: LightImage,
                 bbox: np.ndarray[4,
                 2],
                 kps: np.ndarray[5,
                 2],
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
        self.rec_model.get(img2extract.nd_arr, face)
        return face.normed_embedding


def identify_works(
        task_queue: multiprocessing.queues.Queue,
        result_dict: dict,
        stop_event: multiprocessing.Event):
    """
    long term works in single process
    1. get target from worker_queue
    2. extract embedding
    3. search in milvus by embedding
    4. put result in result_queue
    :param worker_queue:
    :param result_queue:
    :return:
    """
    extractor = Extractor()
    matcher = Matcher()
    while not stop_event.is_set():
        try:
            task_id, task = task_queue.get(timeout=1)
            emmbedding = extractor(task)
            res = matcher(emmbedding)
            result_dict[task_id] = res
        except queue.Empty:
            continue  # 这不是一个错误条件，只是队列暂时为空
        except Exception as e:
            print(f"An error occurred while processing the task: {e}")
            # 可以选择是否将异常信息放入结果字典
            # self.result_dict[task_id] = str(e)


# TODO: test IdentifyWorker class
class IdentifyWorker(Process):
    """
    solo worker for extractor and then search by milvus
    """

    def __init__(
            self,
            task_manager: IdentifyManager):
        self._manager = task_manager
        self._stop_event = multiprocessing.Event()
        super().__init__(
            target=identify_works,
            daemon=True,
            args=(
                self._manager.task_queue,
                self._manager.result_dict,
                self._stop_event))

    def run_work(self):
        super().start()

    def stop(self):
        self._stop_event.set()


class Identifier:
    def __init__(
            self,
            detector: Detector,
            flush_threshold: int,
            max_age=120,
            min_hits=3,
            iou_threshold=0.3,
            server_refresh=False,
            npz_refresh=False,
            test_folder='test_01',
    ):
        from database.milvus_standalone.milvus_for_realtime import MilvusRealTime
        """
        :param detector:
        :param max_age: 超过这个帧数没被更新就删除
        :param min_hits: 超过这个帧数 才会被 识别
        :param iou_threshold: 卡尔曼滤波器的阈值
        :param server_refresh: milvus server 是否刷新
        :param npz_refresh: 是否用新的 npz 文件刷新 milvus server
        """
        self._targets: dict[int, Target] = {}
        self.max_age = max_age  # 超过该帧数没被更新就删除
        self.min_hits = min_hits  # 至少被检测到的次数才算
        self.iou_threshold = iou_threshold
        self._recycled_ids = []
        self._extractor = Extractor()
        self._detector = detector
        self._milvus = MilvusRealTime(
            test_folder=test_folder,
            refresh=server_refresh,
            flush_threshold=flush_threshold)
        if server_refresh:
            self._milvus.load_registered_faces(
                extractor=self._extractor,
                detector=self._detector,
                refresh=npz_refresh)
        else:
            self._milvus.load2RAM()
        self._frame_cnt = 1

    @profile
    def identified_results(self, image2identify: LightImage) -> LightImage:
        self._update(image2identify)
        self._extract(image2identify)
        self._search()
        image2identify.faces.clear()
        matched_and_in_screen = []
        for i, target in enumerate(self._targets.values()):
            if not target.in_screen(self.min_hits):
                continue
            # 没有匹配到
            if target.match_info.face_id == -1:
                match_info = MatchInfo(face_id=-1, name=target.name, score=0.0)
                target.match_info = match_info
            else:
                matched_and_in_screen.append(
                    {"ID": target.id, "Name": target.name})
            image2identify.faces.append(
                [target.bbox, target.kps, target.score, target.colors, target.match_info])

        self._send2web(matched_and_in_screen)

        if self._frame_cnt < 100000:
            self._frame_cnt += 1
        else:
            self._frame_cnt = 0
        return image2identify

    def stop_milvus(self):
        self._milvus.stop_milvus()

    def _send2web(self, new_targets: list[dict]):

        if streaming_event.is_set():
            matched_and_in_screen_deque.append(new_targets)

    @profile
    def _update(self, image2update: LightImage):
        # 更新目标
        detected_tars = [
            RawTarget(
                id=-1,
                bbox=face[0],
                kps=face[1],
                score=face[2]) for face in image2update.faces]
        # 第一次的时候，直接添加
        if self._targets:
            # 提取预测的位置
            predicted_tars = []
            to_del = []
            for i, tar in enumerate(self._targets.values()):
                raw_tar = tar.get_raw_target
                predicted_tars.append(raw_tar)
                pos = raw_tar.bbox
                if np.any(np.isnan(pos)):
                    to_del.append(tar.id)
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
                    detected_tar.bbox, detected_tar.kps, detected_tar.score)
                self._targets[pred_tar.id].update_tracker(detected_tar)
        else:
            unmatched_det_tars = detected_tars
        # add new targets
        for detected_tar in unmatched_det_tars:
            new_id = self._generate_id()
            assert new_id not in self._targets, f'new_id is already in self._targets'
            self._targets[new_id] = Target(
                id=new_id,
                bbox=detected_tar.bbox,
                screen_scale=image2update.screen_scale,
                kps=detected_tar.kps)
        self._clear_old_targets()

    def _clear_old_targets(self):
        # clear dead targets
        keys = []
        for tar in self._targets.values():
            # remove dead targets
            if tar.old_enough(self.max_age):
                keys.append(tar.id)
        for k in keys:
            try:
                del self._targets[k]
            except KeyError:
                print(f'KeyError: tar.id = {k}')
            else:
                heapq.heappush(self._recycled_ids, k)

    def _search(self):
        if not self._milvus:
            raise ValueError('milvus is not initialized')
        self._milvus.face_match([tar for tar in self._targets.values()
                                 if tar.normed_embedding.all()], 0.0)

    def _extract(self, image2extract: LightImage):
        for tar in self._targets.values():
            if tar.rec_satified:
                tar.normed_embedding = self._extractor(
                    image2extract, tar.bbox, tar.kps, tar.score)
        return image2extract

    def _generate_id(self):
        try:
            return heapq.heappop(self._recycled_ids)
        except IndexError:
            return len(self._targets)


class IdentifierTask(Identifier):
    def __init__(
            self,
            jobs: ClosableQueue,
            results: ClosableQueue,
            identifier_params: dict):
        """

        Args:
            jobs:
            results:
            identifier_params: { detector:Detector,
                                 flush_threshold: int,
                                 max_age: int = 120,
                                 min_hits: int = 3,
                                 iou_threshold: float = 0.3,
                                 server_refresh: bool = False,
                                 npz_refresh: bool = False,
                                 test_folder: str = 'test_01'}
        """
        super().__init__(**identifier_params)
        self.jobs = jobs
        self.results = results

    @profile
    def run(self):
        for img in self.jobs:
            identified = self.identified_results(img)
            self.results.put(identified)
        # self.stop_milvus()

        return "IdentifierTask Done"


identifier_params = {
    "flush_threshold": 1000,
    "server_refresh": True,
    "npz_refresh": True,
    "detector": detect_task.detector,
    "test_folder": "test_02"
}
# identifier_task = IdentifierTask(
#     jobs=detect_2_rec_queue, results=rec_2_draw_queue,
#     identifier_params=identifier_params
# )
