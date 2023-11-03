from __future__ import division, annotations

from enum import Enum, auto
from typing import NamedTuple

__all__ = ['MatchInfo']


class MatchInfo(NamedTuple):
    score: float
    face_id: int
    name: str


class IndexType(Enum):
    IVF_FLAT = auto()
    FLAT = auto()
    IVF_SQ8 = auto()
    IVF_SQ8H = auto()
    IVF_PQ = auto()
    HNSW = auto()
    ANNOY = auto()


class MetricType(Enum):
    # 度量类型 L2->欧式距离,不支持余弦相似度，IP->内积
    L2 = auto()
    IP = auto()
