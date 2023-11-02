""" example.py based from pymilvus
"""
import pprint
import random
from timeit import default_timer
from milvus import default_server

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search


# 设置默认的Milvus服务器的存储位置。如果不设置，则默认会存储到 %APPDATA%/milvus-io/milvus-server 路径下
default_server.set_base_dir('test_milvus')

# 清理之前的数据
default_server.cleanup()

# 启动Milvus服务器
default_server.start()

# 定义服务器的IP地址和监听的端口号
_HOST = '127.0.0.1'
_PORT = default_server.listen_port

# 定义常量名，包括集合名和字段名
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# 定义向量的维度和索引文件的大小
# **注意**：向量的维度必须和数据集中的向量维度一致,列的个数是向量的维度
_DIM = 512  # arconnx face.embedding的维度
_INDEX_FILE_SIZE = 32  # 存储索引的最大文件大小

# 定义索引的参数
# 指定相似度 度量类型 L2->欧式距离,不支持余弦相似度，IP->内积
_METRIC_TYPE = 'IP'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3


# 连接到Milvus服务器
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# 创建一个名为'demo'的集合
def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection


# 检查是否存在指定的集合
def has_collection(name):
    return utility.has_collection(name)


# 在Milvus中删除一个集合
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


# 列出Milvus中的所有集合
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


# 向集合中插入实体
def insert(collection: Collection, num: int, dim: int):
    import numpy as np
    data = [
        [i for i in range(num)],
        [[random.random() for _ in range(dim)] for _ in range(num)],
    ]
    print(np.array(data[1]).shape)
    collection.insert(data)
    return data[1]


# 获取集合中的实体数量
def get_entity_num(collection: Collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


# 创建索引
def create_index(collection: Collection, filed_name: str):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


# 删除索引
def drop_index(collection: Collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


# 加载集合
def load_collection(collection: Collection):
    collection.load()


# 释放集合
def release_collection(collection: Collection):
    collection.release()


# 搜索集合
def search(collection: Collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"}
    start = default_timer()
    results = collection.search(**search_param)
    end = default_timer()
    print(f"searching cost time: {end - start} sec")
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))


# 设置集合的属性
def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 1800})


# 主函数，调用以上函数完成数据的插入、搜索等操作
def main():
    # 创建连接
    create_connection()
    # 如果集合已存在，则删除集合
    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)
    # 创建集合
    collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
    # 设置集合属性
    set_properties(collection)

    # 显示所有集合
    list_collections()

    # 插入10000个维度为128的向量
    vectors = insert(collection, 10000, _DIM)

    collection.flush()
    # 获取实体数量
    get_entity_num(collection)
    # 创建索引
    create_index(collection, _VECTOR_FIELD_NAME)
    # 加载集合
    load_collection(collection)

    # 搜索
    search(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, vectors[:3])

    # 释放集合
    release_collection(collection)
    # 删除集合索引
    drop_index(collection)
    # 删除集合
    drop_collection(_COLLECTION_NAME)
    # 断开连接
    default_server.stop()


if __name__ == '__main__':
    main()
