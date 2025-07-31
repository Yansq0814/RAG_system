from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
import json
import os

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='./BAAI/bge-m3', # Specify the model name
    device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

'''
注意：
BGE-M3 稠密部分	把句子整体转成一个 [hidden_size] 向量
BGE-M3 稀疏部分	把句子映射成一个 [vocab_size] 的稀疏向量

包含稠密向量和稀疏向量
Dense Vector	全部有值，浮点数，固定长度	语义相似性
Sparse Vector	大部分是0，少量非零	关键词匹配、token overlap

Embeddings: {'dense': [array([-0.01111861,  0.01530762, -0.04442898, ..., -0.00964683,
       -0.05404138, -0.03924716], dtype=float32), array([-0.00922055,  0.00473605, -0.01914384, ...,  0.00338532,
       -0.01915456,  0.01020869], dtype=float32), array([-0.04059177,  0.01056544, -0.0491685 , ...,  0.02139822,
       -0.04805486,  0.02999171], dtype=float32)], 'sparse': <Compressed Sparse Row sparse array of dtype 'float64'
	with 14 stored elements and shape (3, 250002)>}
Dense document dim: 1024 (1024,)
Sparse document dim: 250002 (250002,)
'''

def load_txt(file_path):
    '''
    将文本文件加载为Document对象
    :param txt_path: 文本文件路径
    :return: Document对象列表，列表中只有一个元素，代表整个文本文档
    '''
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    '''
    page_content="文本内容"
    metadata={'source': '文件路径'}
    '''
    return docs

def load_txts(folder_path):
    '''
    将文件夹下的所有文本文件加载为Document对象列表
    :param txt_path: 文本文件路径
    :return: Document对象列表，列表中只有一个元素，代表整个文本文档
    '''
    loader = DirectoryLoader(folder_path, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    return docs

def load_pdf(file_path):
    '''
    将PDF文件加载为Document对象
    :param pdf_path: PDF文件路径
    :return: Documen对象列表，一页一个document对象
    '''
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    '''
    page_content="文字内容"
    metadata={'source': '文件路径', 'page': 所在页码}
    '''
    return docs

def load_pdfs(folder_path):
    '''
    将文件夹下的所有PDF文件加载为Document对象列表，这里用的是非结构化数据加载器，并不是pypdf加载器
    :param pdf_path: PDF文件路径
    :return: Documen对象列表，一页一个document对象
    '''
    loader = DirectoryLoader(folder_path)
    docs = loader.load()
    return docs

def split_document(docs, chunk_size, chunk_overlap):
    '''
    将传入的document对象列表，进行切分
    :param docs: documents对象列表
    :param chunk_size: 切割长度
    :param chunk_overlap:覆盖长度
    :return:
    '''
    # 创建文本块分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # 加载文档 切分文档
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def document_vectorition(docs):
    '''
    批量将文档使用bgem3模型进行向量化表示
    :param docs: 文档字符串列表
    :return: bed-me模型的输出结果
    '''
    docs_embeddings = bge_m3_ef.encode_documents(docs)
    return docs_embeddings

def query_vectiorition(queries):
    '''
    将用户的query使用bgem3模型进行向量化表示
    :param queries: query对象字符串
    :return: bed-me模型的输出结果
    '''
    query_embeddings = bge_m3_ef.encode_queries(queries)
    return query_embeddings

def batch_vector(docs):
    '''
    批量将文档使用bgem3模型进行向量化表示
    :param docs: document对象列表
    :return:
    '''

    # 取出document对象中的原始文本部分
    docs_str_list = [doc.page_content for doc in docs]

    # 对文档进行向量化
    docs_embeded = document_vectorition(docs_str_list)

    # 取出稠密向量部分
    dense_vector = docs_embeded["dense"]
    # 取出稀疏向量部分
    # sparse_vector = docs_embeded["sparse"]

    return dense_vector

def creat_milvusdatabase():
    '''
    创建milvus向量数据库，连接、创建数据库、创建filed、创建collection、设置索引、设置分区
    :return: 无
    '''
    # 连接MIivus服务
    client = MilvusClient(uri="http://127.0.0.1:19530")

    # 创建或者使用数据库
    databases = client.list_databases()
    if "project_data" not in databases:
        client.create_database(db_name="project_data")
    else:
        client.using_database(db_name="project_data")

    # 1.创建schema(表格框架)
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    # 添加字段filed
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="row_text", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)

    # 2. 创建集合
    client.create_collection(collection_name="data", schema=schema)
    print('collection创建成功')

    # 3.设置索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name='vector', metric_type="COSINE", index_type='', index_name="vector_index")
    client.create_index(collection_name='data', index_params=index_params)

    # 创建分区
    client.create_partition(collection_name='data', partition_name='part_text')
    client.create_partition(collection_name='data', partition_name='part_md')
    client.create_partition(collection_name='data', partition_name='part_pdf')

def create_text_entitys(datasets_path, format):
    '''
    插入text类型的数据文件加载、切分、向量话、创建entitys，并保存起来
    :param datasets_path: 数据文件夹路径
    :param format: md和text两种格式，用于动态创建需要存储的文件名
    :return: 无
    '''

    # 加载数据
    docs = load_txts(datasets_path)
    print('加载的文档长度', len(docs))

    # 切分数据文档
    chunk_size = 1000
    chunk_overlap = 200
    docs = split_document(docs, chunk_size, chunk_overlap)
    print('切分后的文档长度：', len(docs))

    # 文档向量化
    docs_vector = batch_vector(docs)
    print(f'向量化之后的长度:{len(docs_vector)}')
    print("Dense query dim:", docs_vector[0].shape) # 1024

    # 构建entitys
    entitys = []
    for i in range(len(docs_vector)):
        # entity = {"id": i, "vector": docs_vector[i].tolist(), 'row_text': docs[i].page_content, 'source':  docs[i].metadata['source']}
        entity = {"vector": docs_vector[i].tolist(), 'row_text': docs[i].page_content, 'source':  docs[i].metadata['source']}
        entitys.append(entity)

    # 保存到本地
    with open(f'./save_data/entitys_{format}.json', 'w', encoding='utf-8') as f:
        json.dump(entitys, f, ensure_ascii=False)

def create_pdf_entitys(datasets_path):
    '''
    插入pdf类型的数据文件加载、切分、向量话、创建entitys，并保存起来
    :param datasets_path:文件夹路径
    :return:
    '''

    # 存储实体数据的列表
    entitys = []
    # 加载数据
    files_list = os.listdir(datasets_path)
    print(files_list)
    for file in files_list:
        print(f'--------------------{file}文件处理开始-----------------------------')
        file_path = os.path.join(datasets_path, file)

        docs = load_pdf(file_path)
        print('加载的文档长度', len(docs))

        # 切分数据文档
        chunk_size = 1000
        chunk_overlap = 200
        docs = split_document(docs, chunk_size, chunk_overlap)
        print('切分后的文档长度：', len(docs))

        # 文档向量化
        docs_vector = batch_vector(docs)
        print(f'向量化之后的长度:{len(docs_vector)}')
        print("Dense query dim:", docs_vector[0].shape) # 1024

        # 构建entitys
        for i in range(len(docs_vector)):
            # entity = {"id": i, "vector": docs_vector[i].tolist(), 'row_text': docs[i].page_content, 'source':  docs[i].metadata['source'],
            #           'page':  docs[i].metadata['page']}
            entity = {"vector": docs_vector[i].tolist(), 'row_text': docs[i].page_content, 'source':  docs[i].metadata['source'],
                      'page':  docs[i].metadata['page']}
            entitys.append(entity)

        print(f'--------------------{file}文件处理结束-----------------------------')

    # 保存到本地
    with open(f'./save_data/entitys_pdf.json', 'w', encoding='utf-8') as f:
        json.dump(entitys, f, ensure_ascii=False)

def insert_data(file_path, format):
    '''
    加载已处理过的数据，分批次插入到向量数据库中
    :param file_path:
    :return:
    '''
    # 连接MIivus服务
    client = MilvusClient(uri="http://127.0.0.1:19530")
    # 使用指定数据库
    client.using_database(db_name="project_data")

    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        entitys = json.load(f)
        print(len(entitys)) # 15922
    # 按照批次进行插入，一次5000条，防止插入数据失败
    batch = len(entitys) / 5000
    for i in range(int(batch)):
        print(f'--------------------第{i+1}批文件开始处理-----------------------------')
        start = i * 5000
        end = start + 5000
        entitys_batch = entitys[start:end]
        res = client.insert(collection_name='data', data=entitys_batch, partition_name=f'part_{format}')
        print(f'--------------------第{i+1}批文件处理结束-----------------------------')
    if end < len(entitys):
        entitys_batch = entitys[end:]
        print(f'end:{end},处理最后一批数据')
        res = client.insert(collection_name='data', data=entitys_batch, partition_name=f'part_{format}')

    # todo: 更新插入数据 如果集合中已存在该实体的主键，则现有实体将被覆盖。如果集合中不存在主键，则将插入一个新实体。
    # res = client.upsert(collection_name='demo_v2', data=data, partition_name='partitionA' )
    # todo:删除实体（数据）
    # 按照过滤器删除；如果不指定分区，默认情况下会在整个集合中进行删除
    # res = client.delete(collection_name='demo_v2', filter='id in [12, 5, 6]')
    # todo:按照id进行删除；指定分区删除数据
    # res = client.delete(collection_name='demo_v2', ids=[1, 2, 3, 4], partition_name='partitionA')


if __name__ == '__main__':

    datasets_path = '数据集路径'

    # creat_milvusdatabase()
    # create_text_entitys(datasets_path,  'text')
    create_pdf_entitys(datasets_path)
    insert_data('./save_data/entitys_pdf.json', 'pdf')