# coding=utf-8
from pydantic import BaseModel, Field


class UserLogin(BaseModel):
    email: str
    password: str


class Face2SearchSchema(BaseModel):
    face_img: str = Field(..., description="Base64 encoded image data")
    bbox: list[list[float]] = Field(...,
                                    description="Bounding box coordinates")
    kps: list[list[float]] = Field(..., description="Keypoints")
    det_score: float = Field(..., description="Detection score")
    # TODO: rebuild to face2search
