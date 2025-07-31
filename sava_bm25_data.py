from whoosh.index import create_in, open_dir
import jieba
from whoosh.analysis import Tokenizer, Token, Tokenizer, RegexTokenizer
from whoosh.fields import Schema, TEXT, ID
from save_vector_database import *

# 加载数据为列表
def process_total_bm25data():
    '''
    加载原始数据集，然后分割，然后将document的文字部分取出来存储到本地。
    :return:
    '''
    datasets_path_md = '.\datasets\data\output_files_v4'
    datasets_path_txt = '.\datasets\data\some_txt'
    datasets_path_pdf = '.\datasets\data\pdf_upload_ok'

    # 加载md数据
    md_docs = load_txts(datasets_path_md)
    # 加载docs数据
    txt_docs = load_txts(datasets_path_txt)

    # 加载pdf数据
    pdf_docs = []
    files_list = os.listdir(datasets_path_pdf)
    for file in files_list:
        file_path = os.path.join(datasets_path_pdf, file)
        docs = load_pdf(file_path)
        pdf_docs.extend(docs)

    # 总数据
    total_docs = md_docs + txt_docs + pdf_docs

    # 切割数据
    split_docs = split_document(total_docs, chunk_size=250, chunk_overlap=50)

    # 取出document对象中的原始文本部分
    docs_str_list = [doc.page_content for doc in split_docs]

    print(f'切分后的总文档数量：{len(docs_str_list)}')
    temp = docs_str_list[0]
    print(f'数据类型：{type(temp)}')
    print(temp)

    # 保存到本地
    json.dump(docs_str_list, open("./save_data/total_data200.json", "w", encoding="utf-8"), ensure_ascii=False)


class JiebaTokenizer(Tokenizer):
    def __call__(self, value, **kwargs):
        for word in jieba.cut(value):
            t = Token()
            t.text = word
            t.pos = 0
            yield t
jieba_analyzer = JiebaTokenizer()

def write_index():
    # str_list = []
    # for i in ['md', 'pdf', 'text']:
    #     print(f"正在处理{i}文件...")
    #     with open(f"./save_data/entitys_{i}.json", "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #         for item in tqdm(data):
    #             str_list.append(item['row_text'])

    # 加载数据
    with open("./save_data/total_data200.json", "r", encoding="utf-8") as f:
        str_list = json.load(f)

    # 创建 Whoosh 索引目录
    if not os.path.exists("bm25_index_200"):
        os.mkdir("bm25_index_200")

    # 定义一个索引结构：
    # - id 字段是唯一标识，每个文档都应该有不同的 ID（存储型字段）
    # - content 是文档的文本内容，会被分词并用于搜索（支持 BM25）
    # schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True, analyzer=jieba_analyzer))

    # 在 "bm25_index" 目录中基于 schema 创建索引
    ix = create_in("bm25_index_200", schema)

    # 写入索引
    writer = ix.writer()

    for i in range(len(str_list)):
        writer.add_document(id=str(i), content=str_list[i])
    writer.commit()




