# coding=utf-8
"""
    basic api to connect to Milvus server, create collection, insert entities, create index, search from docker
"""
from pathlib import Path

import numpy as np
from numpy import ndarray
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

__all__ = ["MilvusClient"]

from src.boostface.utils.checker import insert_data_check
from src.boostface.utils.load_config import load_config


# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search
# 想象成pandas的DataFrame，列名是字段名，每一行是一个实体，实体由不同的字段组成
##########################################################################
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


class MilvusClient:
    """
    MilvusClient to connect to Milvus server, create collection, insert entities, create index, search from docker
    """

    def __init__(
            self,
            flush_threshold: int = 1000,
            refresh: bool = False, ):
        """

        """
        config_path = Path(
            __file__).parents[3] / 'config' / 'boostface' / 'db' / 'milvus_client.yaml'
        self._config: dict = load_config(config_path)
        assert self._config, "milvus config is empty"
        self._flush_threshold = flush_threshold
        self._new_added = 0
        self._kwargs = {"refresh": refresh, 'flush_threshold': flush_threshold}
        self._connect_to_milvus()
        self._base_config_set()
        self.collection = self._create_collection()
        print("\nlist collections:")
        print(utility.list_collections())
        print(f"\nMlilvus init done.")

    def _base_config_set(self):

        # 定义常量名，包括集合名和字段名
        self._collection_name = self._config.get('collection_name', 'Faces')

        # fields params
        self._id_field_param = self._config['id_field'] | {
            'dtype': DataType[self._config['id_field']['dtype']]}
        self._name_field_param = self._config['name_field'] | {
            'dtype': DataType[self._config['name_field']['dtype']]}
        self._embedding_field_param = self._config['embedding_field'] | {
            'dtype': DataType[self._config['embedding_field']['dtype']]}

        # 存储索引的最大文件大小，单位为MB
        self._index_file_size = self._config['index_file_size']
        # 集合 存储数据的 分片数量
        self._shards_num = self._config['shards_num']
        # **test_needed to measure the performance**
        self._metric_type = ("L2", "IP")
        # 定义 聚类的数量
        self._nlist = self._config['nlist']
        # **test_needed to measure the performance**
        # 定义了搜索时候的 聚类数量
        self._nprobe = self._config['nprobe']
        # **test_needed to measure the performance**

        # 搜索参数预备
        self._prepared_search_param = {
            "metric_type": self._config['metric_type'],
            "params": {"nlist": self._nlist, "nprobe": self._nprobe},
        }
        self._index_param = {
            "index_type": self._config['index_type'],
            **self._prepared_search_param,
        }
        # 指定搜素时返回的最大结果数
        self._top_k = self._config['top_k']
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

    def _connect_to_milvus(self):
        print(f"\nCreate connection...")
        connections.connect(
            host=self._config.get(
                'host',
                '127.0.0.1'),
            port=self._config.get('port', 19530),
            timeout=120)
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
        entities = insert_data_check(entities)
        print("\nInsert data...")
        self.collection.insert(entities)

        print(f"Done inserting new {len(entities[0])}data.")
        if not self.collection.has_index():  # 如果没有index，手动创建
            # Call the flush API to make inserted data immediately available
            # for search
            self.collection.flush()  # 新插入的数据在segment中达到一定阈值会自动构建index，持久化
            print("\nCreate index...")
            self._create_index()
            # 将collection 加载到到内存中
            print("\nLoad collection to memory...")
            self.collection.load()
            utility.wait_for_loading_complete(
                self._collection_name, timeout=10)
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
    def insert_from_files(self, file_paths: list):  # failed
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
        utility.wait_for_index_building_complete(
            self._collection_name, timeout=60)
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
        print(
            f"\nReleased collection : {self._collection_name} successfully !")
        # self.collection.drop_index()
        # print(f"Drop index: {self._collection_name} successfully !")
        # self.collection.drop()
        # print(f"Drop collection: {self._collection_name} successfully !")
        print(f"Stop MilvusClient successfully !")

    def has_collection(self):
        return utility.has_collection(self._collection_name)

    def __bool__(self):
        return self.get_entity_num > 0


def main():
    pass


if __name__ == "__main__":
    main()
