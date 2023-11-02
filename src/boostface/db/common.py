from __future__ import division, annotations

from typing import NamedTuple

__all__ = ['MatchInfo']


class MatchInfo(NamedTuple):
    score: float
    face_id: int
    name: str
