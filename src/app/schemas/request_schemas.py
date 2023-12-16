# coding=utf-8
import base64
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, Field

Kps = NDArray[np.float64]  # shape: (5, 2)
Bbox = NDArray[np.float64]  # shape: (4, 2)
Embedding = NDArray[np.float64]  # shape: (512, )
Image = NDArray[np.uint8]  # shape: (height, width, 3)


class UserLogin(BaseModel):
    email: str
    password: str


class Face2SearchSchema(BaseModel):
    """Face2Search schema"""
    face_img: str = Field(..., description="Base64 encoded image data")
    bbox: list[float] = Field(...,
                              description="Bounding box coordinates")
    kps: list[list[float]] = Field(..., description="Keypoints")
    det_score: float = Field(..., description="Detection score")
    uid: str = Field(..., description="Face ID")


# 定义 Face2Search
@dataclass
class Face2Search:
    face_img: Image
    bbox: Bbox
    kps: Kps
    det_score: float
    uid: str

    @classmethod
    def from_schema(cls, schema: BaseModel) -> "Face2Search":
        # 将 base64 编码的图像转换为 Image 类型 (NumPy ndarray)
        image_data = base64.b64decode(schema.face_img)
        image = np.frombuffer(image_data, dtype=np.uint8)  # 假设解码后为正确的图像数据格式

        # 将列表转换为 NumPy ndarrays
        bbox = np.array(schema.bbox, dtype=np.float64)
        kps = np.array(schema.kps, dtype=np.float64)

        return cls(
            face_img=image,
            bbox=bbox,
            kps=kps,
            det_score=schema.det_score,
            uid=schema.uid
        )
