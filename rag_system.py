from pymilvus.model.reranker import BGERerankFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from pymilvus import MilvusClient
import jieba
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama
import json

model_path = '模型的路径'


bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='./BAAI/bge-m3', # Specify the model name
    device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

bge_rf = BGERerankFunction(
    model_name="./BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
    device="cuda:0" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)


def query_retrieval(query, top_k=5):
    '''
    向量检索
    :param query: 查询字符串
    :param top_k:
    :return: 检索出来的文档字符串列表
    '''

    # 加载milvus向量数据库
    client = MilvusClient(uri="http://127.0.0.1:19530")
    client.using_database(db_name="project_data")
    query = [query]
    # 将query向量化
    query_embeddings = bge_m3_ef.encode_queries(query)

    dense_vector_query = query_embeddings['dense']

    # 检索
    res = client.search(collection_name='data',
        data=dense_vector_query,
        limit=top_k,
        search_params={"metric_type": "COSINE"}, # search_params是在查询时执行距离计算方式，如果定义索引的时候，已经制定了方式可以不写
        output_fields=['vector', 'row_text', 'source']
    )
    retrival_result = []
    for i in res[0]:
        retrival_result.append(i['row_text'])

    return retrival_result

def bm25_search(query, top_k=5):
    '''
    bm25检索算法
    :param query: 查询字符串
    :param top_k:
    :return: 检索出来的文档列表
    '''

    # 方式一：分词+过滤停用词 提取关键词
    # 分词
    cut_res = list(jieba.cut(query))
    # 过滤停用词
    with open("./cn_stopwords.txt", "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f.readlines()]
    keywords = [word for word in cut_res if word not in stop_words and word.strip()]

    # 方式二：词性标注
    # import jieba.posseg as pseg
    # words = pseg.cut(query)
    # keywords = [w.word for w in words if w.flag.startswith("n")]  # 只取名词
    # print(keywords)

    # 检索
    query_text = ' '.join(keywords)

    # 打开名为 "bm25_index" 的 Whoosh 索引目录
    ix = open_dir("bm25_index_200")
    # 创建一个 查询解析器，用于将用户输入的自然语言查询 query_text 转换为 Whoosh 能理解的查询对象。
    parser = QueryParser("content", schema=ix.schema)

    # 创建一个搜索器 searcher，用于在打开的索引中执行检索操作。
    with ix.searcher() as searcher:
        # 将用户输入的查询文本解析成 Whoosh 查询对象 query，用于执行检索。
        query = parser.parse(query_text)
        # 检索：使用解析后的 query 在索引中检索；返回前 top_k 个最相关的文档。
        results = searcher.search(query, limit=top_k)

        # 打印 top-k 结果
        ids = [r["id"] for r in results]
        print("Top results IDs from BM25:", ids)

        # 查询 原始数据
        with open("./save_data/total_data200.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            docs = [data[int(id)] for id in ids]

        return docs

def rerank_query(query, documents, top_k=3):
    '''
    调用重排序模型
    :param query:
    :param documents:
    :param top_k:
    :return: 重排序以后的文档列表
    '''

    results = bge_rf(
        query=query,
        documents=documents,
        top_k=top_k,
    )
    return results

def mixed_retrieval(query, top_k1=5, top_k2=5, rerank_top_k=3):
    '''
    混合检索后重排序
    :param query1: ['str']
    :param query2:
    :param top_k1:
    :param top_k2:
    :param rerank_top_k:
    :return:
    '''
    # 1.向量检索
    docs1 = query_retrieval(query, top_k1)

    # 2.bm25检索
    docs2 = bm25_search(query, top_k2)

    # 3.重排序
    docs = docs1 + docs2
    rerank_results = rerank_query(query, docs, rerank_top_k)

    results = []
    for item in rerank_results:
        results.append(item.text)

    return results


# transformers的方式加载模型
def load_model_transformer(prompt):
    '''
    使用Transformer加载本地LLM
    :param prompt:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    print(f'text:{text}')
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f'model_inputs:{model_inputs}')

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
    return thinking_content, content

# ollama加载模型
def load_model_ollama(prompt, model='qwen3:4b'):
    '''
    使用Ollama的方式调用LLM
    :param prompt:
    :return:
    '''

    # 聊天式
    # response = ollama.chat(model='qwen3:1.7b', messages=[{'role': 'user', 'content': prompt}])
    # return response['message']['content']

    # ⽣成式 ollama run qwen3:4b
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']

def rag(query, top_k1=5, top_k2=5, rerank_top=3, model_type='qwen3:4b'):
    '''
    rag集成
    :param query:查询query
    :param top_k1:向量检索top_k
    :param top_k2: bm25检索top_k
    :param rerank_top: 重排序top_k
    :return:
    '''

    prompt = """
    你是一个知识严谨、表达清晰的AI助手,请根据以下提供的参考资料回答用户的问题。
       - 回答必须仅基于资料内容，不要依赖你的常识或已有知识；
       - 如果资料中找不到相关信息，请明确说明“资料中未提及”；
       - 回答应简洁准确，如有引用请尽可能贴近原文表达；
       - 不要杜撰，不要引入资料中未包含的信息。

    用户问题：{}

    参考资料：
    {}

    请基于以上资料作答：
        """

    # R: 混合检索
    rerank_docs = mixed_retrieval(query, top_k1, top_k2, rerank_top)

    # A:拼接Prompt
    prompt = prompt.format(query, '\n\n'.join(rerank_docs))
    # G：使用Ollama调用本地LLM生成答案
    response = ollama.generate(model=model_type, prompt=f'/no_think{prompt}')

    # 聊天式
    # response = ollama.chat(model=model_type, messages=[{'role': 'user', 'content': prompt}])
    # return response['message']['content']
    return rerank_docs, response['response']

















