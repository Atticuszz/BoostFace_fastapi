# coding=utf-8
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from pymilvus.orm import utility

from src.boostface.db.milvus_client import MilvusClient
from test import generate_normalized_embeddings

__all__ = ["register", "matcher"]
client = MilvusClient()


# TODO: build Matcher
class Matcher:
    """
    继承milvus_client ,作为匹配器
    """

    def __init__(self, client: MilvusClient, threshold=0.5):
        self._client = client
        self._threshold = threshold
        print("Loading collection to RAM")
        self._client.collection.load(timeout=10)
        utility.wait_for_loading_complete(self._client.collection.name, timeout=10)

    def __call__(self, embedding: ndarray[512]) -> str | None:
        """
        :param emmbedding: must be normed
        :return:
        """
        assert np.isclose(norm(embedding), 1), "embedding must be normed"
        results: list[list[dict]] = self._client.search([embedding])
        ret = None
        for i, result in enumerate(results):
            result = result[0]  # top_k=1
            if result['score'] > self._threshold:
                ret = str(result['id'])
        return ret

    def shut_down(self):
        self._client.shut_down()


class Register:
    def __init__(self, client: MilvusClient):
        self._client = client

    def insert(self, embedding: ndarray[512], id: str):
        assert np.isclose(norm(embedding), 1), "embedding must be normed"
        self._client.insert([np.array([id]), np.array(['name']), np.array([embedding])])

    # 批量插入
    def insert_batch(self, faces: list[list[ndarray[512], str, str]]):
        ids = []
        embeddings = []
        names = []
        for embedding, id, name in faces:
            assert np.isclose(norm(embedding), 1), "embedding must be normed"
            ids.append(id)
            embeddings.append(embedding)
            names.append(name)
        self._client.insert([np.array(ids), np.array(names), np.array(embeddings)])


# register = Register(client)
matcher = Matcher(client)
if __name__ == '__main__':
    num_faces = 10000
    embeddings = generate_normalized_embeddings(num_faces)
    ids = [str(i) for i in range(num_faces)]  # 生成 ID 列表
    names = [f"Person_{i}" for i in range(num_faces)]  # 生成名称列表
    print("data generated")
    # 创建 Register 实例
    register = Register(client)
    print("register created")
    # 批量插入数据
    print("inserting data...")
    faces_data = list(zip(embeddings, ids, names))
    register.insert_batch(faces_data)
