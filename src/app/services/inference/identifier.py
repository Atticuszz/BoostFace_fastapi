import collections
from multiprocessing import Process, Event, Queue
from pathlib import Path

from boostface.types import Embedding
from app.services.inference.common import Face, TaskType
from .model_zoo import ArcFaceONNX
from .model_zoo import get_model
from ..db.base_model import MatchedResult
from ..db.operations import Matcher, Registrar
from ...core.config import logger

matched_and_in_screen_deque = collections.deque(maxlen=1)


class Extractor:
    """
    extract face embedding from given target bbox and kps, and det_score by running model in onnx
    """

    def __init__(self):
        root = Path(__file__).parent / \
            'model_zoo' / 'models' / 'irn50_glint360k_r50.onnx'
        self.rec_model: ArcFaceONNX = get_model(root, providers=(
            'CUDAExecutionProvider', 'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def run_onnx(self, face: Face) -> Embedding:
        """
        get embedding of face from given target kps, and det_score
        :return: face embedding
        """
        self.rec_model.get(face.img, face)
        return face.normed_embedding


class IdentifyWorker(Process):
    """
    solo worker for extractor and then search by milvus
    """

    def __init__(
            self,
            task_queue: Queue,
            result_queue: Queue,
            registered_queue: Queue):
        super().__init__(daemon=True)
        self.registered_queue = registered_queue
        self._is_running = Event()
        self._task_queue: Queue[tuple[TaskType, Face]] = task_queue
        self._result_queue: Queue[MatchedResult] = result_queue
        self._extractor = Extractor()
        self._matcher = Matcher()
        self._registrar = Registrar()

    def start(self) -> None:
        self._is_running.set()
        super().start()
        logger.debug("IdentifyWorker start")

    def run(self):
        """long time works in a single process"""
        logger.debug("IdentifyWorker running")
        while self._is_running.is_set():
            task_type, face = self._task_queue.get()
            if task_type == TaskType.IDENTIFY:
                self._identify(face)
            elif task_type == TaskType.REGISTER:
                self._register(face)
            else:
                raise TypeError("task_type must be TaskType")

    def stop(self):
        self._is_running.clear()
        self._matcher.stop_client()
        super().join()
        logger.debug("IdentifyWorker stop")

    def _identify(self, face: Face):
        normed_embedding = self._extractor.run_onnx(face)
        match_info = self._matcher.search(normed_embedding)
        match_info.face_id = face.face_id
        assert match_info is not None, "match_info must not be None"
        self._result_queue.put(match_info)

    def _register(self, face: Face):
        normed_embedding = self._extractor.run_onnx(face)
        self._registrar.sign_up(
            normed_embedding,
            face.sign_up_info.id,
            face.sign_up_info.name)
        self._result_queue.put(face.face_id)
