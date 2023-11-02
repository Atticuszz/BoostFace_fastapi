""" example.py based from pymilvus
"""
import pprint
from pathlib import Path

import numpy as np
from numpy import ndarray
from milvus import MilvusServer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

__all__ = ["Milvus"]

from sympy import ShapeError


# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search
# 想象成pandas的DataFrame，列名是字段名，每一行是一个实体，实体由不同的字段组成
#################################################################################
# 2. field view
# +-+------------+------------+------------------+------------------------------+
# | | "pk"      | "random"   |    "embeddings"   |
# +-+------------+------------+------------------+------------------------------+
# |1|  VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
# |2|  VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
# |3|| VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
##############################################################################
# Data type of the data to insert must **match the schema of the collection**
# otherwise Milvus will raise exception.
class Milvus:
    def __init__(
            self, flush_threshold: int, name: str = "Milvus_server", **kwargs: dict
    ):
        """
        :param name: 设置默认的Milvus服务器的存储位置。如果不设置，则默认会存储到 %APPDATA%/milvus-io/milvus-server 路径下
        :param kwargs:refresh 决定是否启动前先删除原有的数据
        """
        self._flush_threshold = flush_threshold
        self._new_added = 0
        self._kwargs = {"refresh": False}
        self._kwargs.update(kwargs)
        self._server_start(name)
        self._base_config_set(**kwargs)
        self.collection = self._create_collection()
        self.list_collections()
        print(f"\nMlilvus init done.")

    def _base_config_set(self, **kwargs: dict):

        # 定义常量名，包括集合名和字段名
        self._collection_name = kwargs.get("collection_name", "Faces")
        # fields params
        self._id_field_param = {
            "name": "id",
            "dtype": DataType.INT64,
            "description": "face_ids",
            "is_primary": True,
        }
        self._name_field_param = {
            "name": "name",
            "dtype": DataType.VARCHAR,
            "description": "names of faces",
            "max_length": 50,
        }
        self._embedding_field_param = {
            "name": "normed_embedding",
            "dtype": DataType.FLOAT_VECTOR,
            "description": "face_normed_embedding vector",
            "dim": 512,
        }
        # 存储索引的最大文件大小，单位为MB
        self._index_file_size = kwargs.get("index_file_size", 32)
        # 集合 存储数据的 分片数量
        self._shards_num = kwargs.get(
            "shards_num", 6
        )  # **test_needed to measure the performance**
        # 定义索引的参数
        self._index_type = (
            "IVF_FLAT",
            "FLAT",
            "IVF_SQ8",
            "IVF_SQ8H",
            "IVF_PQ",
            "HNSW",
            "ANNOY",
        )
        self._metric_type = ("L2", "IP")  # 度量类型 L2->欧式距离,不支持余弦相似度，IP->内积
        # 定义 聚类的数量
        self._nlist = kwargs.get(
            "nlist", 1024
        )  # **test_needed to measure the performance**
        # 定义了搜索时候的 聚类数量
        self._nprobe = kwargs.get(
            "nprobe", 256 // 13
        )  # **test_needed to measure the performance**

        # 搜索参数预备
        self._prepared_search_param = {
            "metric_type": self._metric_type[1],
            "params": {"nlist": self._nlist, "nprobe": self._nprobe},
        }
        self._index_param = {
            "index_type": self._index_type[0],
            **self._prepared_search_param,
        }
        # 指定搜素时返回的最大结果数
        self._top_k = kwargs.get("top_k", 1)
        # 指定集合搜素字段
        # expr: 指定筛选条件
        # partition_names: 指定分区名
        # output_fields: 指定额外的返回的字段
        # _async，_callback异步编程相关
        self._collection_search_param = {
            "param": self._prepared_search_param,
            "anns_field": self._embedding_field_param["name"],
            "limit": self._top_k,
            "output_fields": [
                self._id_field_param["name"],
                self._name_field_param["name"],
            ],
        }  # search doesn't support vector field as output_fields

    @property
    def milvus_params(self) -> dict:
        params = {
            "collection_name": self._collection_name,
            "collection_description": self.collection.description,
            "collection_schema": self.collection.schema,
            "collection_num_entities": self.collection.num_entities,
            "collection_primary_field": self.collection.primary_field,
            "collection_fields": self.collection.schema.fields,
            "index_file_size": self._index_file_size,
            "search_param": self._collection_search_param,
            "index_param": self._index_param,
        }
        return params

    # 2023-7-31 base_dir* new: 指定文件路径
    def _server_start(self, name: str = "test_milvus"):
        # 路径仍然是无效的，需要在内部修改代码
        base_dir: Path = Path(__file__).absolute().parent.joinpath(name)
        self._milvus_server = MilvusServer(wait_for_started=False, debug=True)
        self._milvus_server.set_base_dir(base_dir.as_posix())
        if self._kwargs[
            "refresh"
        ]:  # 2023-7-31 new: 取消清除之前的数据，不需要每次都build index,collections
            self._milvus_server.cleanup()
        self._milvus_server.start()
        print(f"Milvus server is running on {self._milvus_server.server_address}")
        print(f"\nCreate connection...")
        connections.connect(
            host="127.0.0.1", port=self._milvus_server.listen_port, timeout=120
        )
        print(f"\nList connections:")
        print(connections.list_connections())

    # 创建一个的集合
    def _create_collection(self) -> Collection:
        if (
                utility.has_collection(self._collection_name)
                and not self._kwargs["refresh"]
        ):
            print(f"\nFound collection: {self._collection_name}")
            # 2023-7-31 new: 如果存在直接返回 collection
            return Collection(self._collection_name)
        elif utility.has_collection(self._collection_name) and self._kwargs["refresh"]:
            print(f"\nFound collection: {self._collection_name}, deleting...")
            utility.drop_collection(self._collection_name)
            print(f"Collection {self._collection_name} deleted.")

        print(f"\nCollection {self._collection_name} is creating...")
        id_field = FieldSchema(**self._id_field_param)
        name_field = FieldSchema(**self._name_field_param)
        embedding_field = FieldSchema(**self._embedding_field_param)
        fields = [id_field, name_field, embedding_field]
        # 2023-7-31 new: 允许了动态字段，即可以在插入数据时动态添加字段
        schema = CollectionSchema(
            fields=fields,
            description="collection faces_info_collection",
            enable_dynamic_field=True,
        )
        # ttl指定了数据的过期时间，单位为秒，0表示永不过期
        collection = Collection(
            name=self._collection_name,
            schema=schema,
            shards_num=self._shards_num,
            properties={"collection.ttl.seconds": 0},
        )
        print("collection created:", self._collection_name)
        return collection

    @staticmethod
    def _check_data(data: list[ndarray, ndarray, ndarray]) -> list:
        ids, names, normed_embeddings = data
        # 不可以有缺失值
        if (
                (ids == "").any()
                or (names == "").any()
                or (normed_embeddings == np.NAN).any()
        ):
            raise ValueError('data cannot be ""or NAN')
        # 条目数必须相同
        if not (ids.shape[0]) == names.shape[0] == normed_embeddings.shape[0]:
            raise ValueError("data is not same length")
        # id必须是int64
        if ids.dtype != np.int64:
            ids = ids.astype(np.int64)
        # id 必须唯一
        if np.unique(ids).shape[0] != ids.shape[0]:
            raise ShapeError("ids must be unique")
        # name必须是str
        if names.dtype != str:  # np.str: deprecated
            names = names.astype(str)
        # normed_embeddings必须是float32
        if normed_embeddings.dtype != np.float32:
            normed_embeddings = normed_embeddings.astype(np.float32)
        # normed_embeddings必须是512维
        if normed_embeddings.shape[1] != 512:
            raise ShapeError("normed_embeddings must be 512 dim")
        # normed_embeddings必须是 单位向量
        norms_after_normalization = np.linalg.norm(normed_embeddings, axis=1)
        is_normalized = np.allclose(norms_after_normalization, 1)
        if not is_normalized:
            raise ValueError("normed_embeddings must be normalized")
        # name长度不能超过50
        if not all([len(name) <= 50 for name in names]):
            raise ValueError("name length must be less than 50")

        # 提取成列表
        entries = [
            [_id for _id in ids],
            [name for name in names],
            [embedding for embedding in normed_embeddings],
        ]
        return entries

    def insert(self, entities: list[ndarray, ndarray, ndarray]):
        """

        :param entities: [[id:int64],[name:str,len<50],[normed_embedding:float32,shape(512,)]]
        :return:
        """
        # print 当前collection的数据量
        print(
            f"\nbefore_inserting,Collection:[{self._collection_name}] has {self.collection.num_entities} entities."
        )

        print("\nEntities check...")
        entities = self._check_data(entities)
        print("\nInsert data...")
        self.collection.insert(entities)

        print(f"Done inserting new {len(entities[0])}data.")
        if not self.collection.has_index():  # 如果没有index，手动创建
            # Call the flush API to make inserted data immediately available for search
            self.collection.flush()  # 新插入的数据在segment中达到一定阈值会自动构建index，持久化
            print("\nCreate index...")
            self._create_index()
            # 将collection 加载到到内存中
            print("\nLoad collection to memory...")
            self.collection.load()
            utility.wait_for_loading_complete(self._collection_name, timeout=10)
        else:
            # 由于没有主动调用flush, 只有达到一定阈值才会持久化 新插入的数据
            # 达到阈值后，会自动构建index，持久化，持久化后的新数据，才能正常的被加载到内存中，可以查找
            # 异步的方式加载数据到内存中，避免卡顿
            # 从而实现动态 一边查询，一边插入
            self._new_added += 1
            if self._new_added >= self._flush_threshold:
                print("\nFlush...")
                self.collection.flush()
                self._new_added = 0
                self.collection.load(_async=True)

        # print 当前collection的数据量
        print(
            f"after_inserting,Collection:[{self._collection_name}] has {self.collection.num_entities} entities."
        )

    # 向集合中插入实体
    def insert_from_files(self, file_paths: list):  ### failed
        print("\nInsert data...")
        # 3. insert entities
        task_id = utility.do_bulk_insert(
            collection_name=self._collection_name,
            partition_name=self.collection.partitions[0].name,
            files=file_paths,
        )
        task = utility.get_bulk_insert_state(task_id=task_id)
        print("Task state:", task.state_name)
        print("Imported files:", task.files)
        print("Collection name:", task.collection_name)
        print("Start time:", task.create_time_str)
        print("Entities ID array generated by this task:", task.ids)
        while task.state_name != "Completed":
            task = utility.get_bulk_insert_state(task_id=task_id)
            print("Task state:", task.state_name)
            print("Imported row count:", task.row_count)
            if task.state == utility.BulkInsertState.ImportFailed:
                print("Failed reason:", task.failed_reason)
                raise Exception(task.failed_reason)
        self.collection.flush()
        print(self.get_entity_num)
        print("Done inserting data.")
        self._create_index()
        utility.wait_for_index_building_complete(self._collection_name)

    # 列出Milvus中的所有集合
    @staticmethod
    def list_collections():
        print("\nlist collections:")
        print(utility.list_collections())

    # 获取集合中的实体数量
    @property
    def get_entity_num(self):
        return self.collection.num_entities

    # 创建索引
    def _create_index(self):
        self.collection.create_index(
            field_name=self._embedding_field_param["name"],
            index_params=self._index_param,
        )
        # 检查索引是否创建完成
        utility.wait_for_index_building_complete(self._collection_name, timeout=60)
        print("\nCreated index:\n{}".format(self.collection.index().params))

    # 搜索集合
    # noinspection PyTypeChecker
    def search(self, search_vectors: list[np.ndarray]) -> list[list[dict]]:
        # search_vectors可以是多个向量
        # print(f"\nSearching ...")
        results = self.collection.search(
            data=search_vectors, **self._collection_search_param
        )
        # print("collecting results ...")
        ret_results = [[] for _ in range(len(results))]
        for i, hits in enumerate(results):
            for hit in hits:
                ret_results[i].append(
                    {
                        "score": hit.score,
                        "id": hit.entity.get(self._id_field_param["name"]),
                        "name": hit.entity.get(self._name_field_param["name"]),
                    }
                )
        # pprint.pprint(f"Search results : {ret_results}")
        return ret_results

    # 删除集合中的所有实体,并且关闭服务器
    # question: 可以不删除吗？下次直接读取上一次的内容？
    def shut_down(self):
        # 将仍未 持久化的数据持久化
        print(f"\nFlushing to seal the segment ...")
        self.collection.flush()
        # 释放内存
        self.collection.release()
        print(f"\nReleased collection : {self._collection_name} successfully !")
        # self.collection.drop_index()
        # print(f"Drop index: {self._collection_name} successfully !")
        # self.collection.drop()
        # print(f"Drop collection: {self._collection_name} successfully !")
        self._milvus_server.stop()
        print(f"Stop Milvus server successfully !")

    def has_collection(self):
        return utility.has_collection(self._collection_name)

    def __bool__(self):
        return self.get_entity_num > 0


def main():
    # 创建连接
    # Faces_server = Milvus()
    pass


if __name__ == "__main__":
    main()
