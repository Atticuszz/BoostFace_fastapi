
from pathlib import Path

from line_profiler_pycharm import profile

from .common import ClosableQueue, Image2Detect, FaceNew

__all__ = ['Detector', 'DetectorTask']

from ..utils.decorator import thread_error_catcher


class Detector:
    """
    scrfd det_2.5g.onnx with onnxruntime
    """

    def __init__(self):
        from ..model_zoo import get_model
        root = Path(__file__).parents[1].joinpath('model_zoo\\models\\insightface\\det_2.5g.onnx')
        self.detector_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        prepare_params = {'ctx_id': 0,
                          'det_thresh': 0.5,
                          'input_size': (320, 320)}
        self.detector_model.prepare(**prepare_params)

    @profile
    def run_onnx(self, img2detect: Image2Detect) -> Image2Detect:
        """
        run onnx model
        :param img2detect:
        :return: img2detect with faces
        """
        detect_params = {'max_num': 0, 'metric': 'default'}
        bboxes, kpss = self.detector_model.detect(img2detect.nd_arr, **detect_params)
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            face: FaceNew = FaceNew(bbox, kps, det_score,
                                    (0, 0, img2detect.nd_arr.shape[1], img2detect.nd_arr.shape[0]))
            img2detect.faces.append(face)
        return img2detect


class DetectorTask:
    """
    run detector on images in jobs and put results in results queue in a thread
    """

    def __init__(self, jobs: ClosableQueue, results: ClosableQueue):
        self.detector = Detector()
        self._jobs = jobs
        self._results = results

    @profile
    @thread_error_catcher
    def run_detection(self) -> str:
        """
        run_detector
        :return:
        """
        print("detector start")
        for img in self._jobs:
            # print("detector start{}".format(img.nd_arr.shape))
            results: Image2Detect = self.detector.run_onnx(img)
            self._results.put(results)
        return "DetectorTask Done"
