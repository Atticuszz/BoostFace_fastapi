# coding=utf-8
from numpy import ndarray

from src.boostface.db.milvus_client import MilvusClient


# TODO: build Matcher
class Matcher(MilvusClient):
    """
    继承milvus_client ,作为匹配器
    """

    def __int__(self):
        pass

    def __call__(self, emmbedding: ndarray):
        pass
