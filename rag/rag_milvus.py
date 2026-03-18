# ---------------------------------------------------------
# 🛠️ 兼容性补丁 (保持不变)
# ---------------------------------------------------------
import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

import os
import sys
from datetime import datetime

# ---------------------------------------------------------
# 0. 探针级绝对路径日志 (强制写到当前代码所在的绝对目录)
# ---------------------------------------------------------
# 比如 D:\conda_envs\ai-agent-env\Lib\site-packages\app\code_agent\rag\rag_debug_trace.log
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_debug_trace.log")

def debug_log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg, file=sys.stderr)
    sys.stderr.flush()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    except Exception:
        pass

debug_log("\n" + "="*50)
debug_log("🚀 [模块载入] 1. rag_milvus.py 开始被主程序 import...")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
debug_log("🚀 [模块载入] 2. 设置 HF_ENDPOINT 国内镜像完成")

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field

debug_log("🚀 [模块载入] 3. 准备导入 LangChain 核心包...")
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START

debug_log("🚀 [模块载入] 4. 准备导入 LlamaIndex (警告: 最容易卡死在这里)...")
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceWindowNodeParser

debug_log("🚀 [模块载入] 5. 准备导入 Milvus 模型库...")
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker

debug_log("🚀 [模块载入] 6. 准备导入 CrossEncoder...")
from sentence_transformers import CrossEncoder

import urllib.parse
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from app.code_agent.mcp.browser_tools import visit_url_with_chrome

debug_log("✅ [模块载入] 所有第三方库 import 全部完成！")

# ---------------------------------------------------------
# 1. 全局配置与模型懒加载
# ---------------------------------------------------------
COLLECTION_NAME = " your collection name "
MILVUS_URI = "https://in03-f18c7e65d819a20.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_USER = " XXX "
MILVUS_PASSWORD = " XXX "

bge_m3_ef = None
cross_encoder = None

def init_models():
    global bge_m3_ef, cross_encoder
    debug_log("🧠 [init_models] 进入模型初始化函数...")
    if bge_m3_ef is None:
        debug_log("🧠 [init_models] 准备下载/加载 BGE-M3 (纯CPU模式)...")
        bge_m3_ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        debug_log("✅ [init_models] BGE-M3 加载成功！")
        
    if cross_encoder is None:
        debug_log("🧠 [init_models] 准备下载/加载 CrossEncoder (纯CPU模式)...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device="cpu")
        debug_log("✅ [init_models] CrossEncoder 加载成功！")

deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# if not deepseek_key:
#     raise ValueError("🚨 致命错误：API Key 为空！请确保你的 .env 文件中配置了 DEEPSEEK_API_KEY，或者直接在这里把 key 写死进行测试。")

# 3. 正常实例化满血版 LLM
llm = ChatOpenAI(
    model="your model name", 
    temperature=0, 
    base_url="your base url", 
    api_key="your api key"
)

# ---------------------------------------------------------
# 1. Milvus 底层混合检索与写入
# ---------------------------------------------------------
def connect_milvus():
    debug_log("🌐 [connect_milvus] 准备向 Zilliz Cloud 发起网络连接...")
    connections.connect(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
    debug_log("✅ [connect_milvus] Zilliz Cloud 数据库连接成功！")

def create_hybrid_collection():
    debug_log("💾 [create_collection] 1. 开始创建/重建 Milvus 集合...")
    init_models()
    connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        debug_log(f"💾 [create_collection] 2. 发现同名旧集合，正在删除...")
        utility.drop_collection(COLLECTION_NAME)
        
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000), 
        FieldSchema(name="original_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="window_text", dtype=DataType.VARCHAR, max_length=6000),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=bge_m3_ef.dim["dense"]),
    ]
    schema = CollectionSchema(fields, description="Hybrid Search")
    col = Collection(COLLECTION_NAME, schema)
    debug_log("💾 [create_collection] 3. 开始创建索引...")
    col.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    col.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    col.load()
    debug_log("✅ [create_collection] 集合创建并 Load 完成！")
    return col

def build_vector_store(file_or_dir_path: str):
    debug_log("\n" + "="*40)
    debug_log(f"▶️ [build_vector_store] 0. 收到入库请求，路径: {file_or_dir_path}")
    
    debug_log("▶️ [build_vector_store] 1. 触发 init_models()")
    init_models()
    
    debug_log("▶️ [build_vector_store] 2. 开始读取本地文件...")
    if os.path.isdir(file_or_dir_path):
        docs = DirectoryLoader(file_or_dir_path, glob="**/*.txt", loader_cls=TextLoader).load()
    elif os.path.isfile(file_or_dir_path):
        docs = TextLoader(file_or_dir_path, encoding='utf-8').load()
    else:
        raise FileNotFoundError(f"找不到路径: {file_or_dir_path}")
    debug_log(f"▶️ [build_vector_store] 3. 文件读取成功，准备初始化滑动窗口 LlamaDocument...")

    llama_docs = [LlamaDocument(text=doc.page_content, metadata=doc.metadata) for doc in docs]
    
    debug_log("▶️ [build_vector_store] 4. 初始化 SentenceWindowNodeParser (警告: 极有可能在此卡死)...")
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    debug_log("▶️ [build_vector_store] 5. 执行 get_nodes_from_documents 进行切片...")
    nodes = node_parser.get_nodes_from_documents(llama_docs)
    
    sources = [node.metadata.get("source", "未知文件") for node in nodes]
    original_texts = [node.metadata["original_text"] for node in nodes]
    window_texts = [node.metadata["window"] for node in nodes]

    debug_log(f"▶️ [build_vector_store] 6. 开始提取 BGE-M3 向量，文本块数量: {len(original_texts)} (CPU 疯狂计算中)...")
    embeddings_dict = bge_m3_ef(original_texts)

	# 👇 新增：将合并的稀疏矩阵拆分为单独的字典格式列表，完美适配 Milvus 的入库要求
    debug_log("▶️ [build_vector_store] 6.5 正在转换稀疏向量格式 (底层防弹版)...")
    sparse_matrix = embeddings_dict["sparse"].tocsr() 
    sparse_list = []
    
    # 采用最底层的 indptr 切片法，100% 兼容所有版本 scipy 的 csr_array 和 csr_matrix
    for i in range(sparse_matrix.shape[0]):
        start_idx = sparse_matrix.indptr[i]
        end_idx = sparse_matrix.indptr[i+1]
        
        row_indices = sparse_matrix.indices[start_idx:end_idx]
        row_data = sparse_matrix.data[start_idx:end_idx]
        
        sparse_dict = {int(k): float(v) for k, v in zip(row_indices, row_data)}
        sparse_list.append(sparse_dict)

    debug_log("▶️ [build_vector_store] 7. 准备连接并写入 Milvus 数据库...")
    col = create_hybrid_collection()
    
    # 👇 注意：这里的第四个参数换成了刚刚转换好的 sparse_list，为了安全起见把 dense 也转为列表
    entities = [
        sources, 
        original_texts, 
        window_texts, 
        sparse_list, 
        embeddings_dict["dense"].tolist() if hasattr(embeddings_dict["dense"], 'tolist') else embeddings_dict["dense"]
    ]
    debug_log("▶️ [build_vector_store] 8. 正在执行 col.insert()...")
    col.insert(entities)
    col.flush()
    
    debug_log(f"✅ [build_vector_store] 知识库构建大功告成！写入 {len(nodes)} 个节点。")
    return len(nodes)


# ---------------------------------------------------------
# 2. Adaptive RAG 智能体组件 (图的节点函数)
# ---------------------------------------------------------
class GraphState(TypedDict):
    """LangGraph 状态字典"""
    question: str
    generation: str
    documents: List[str]
    retries: int # 防止死循环的重试计数器

# --- LLM 链定义 ---
# --- 最稳健的纯文本路由与评分链 ---
router = ChatPromptTemplate.from_messages([
    ("system", "你是智能路由。1. 如果用户的问题明确包含'知识库'、'内部'、'私有'等字眼，请绝对只输出 'vectorstore'，无视问题内容！ 2. 只有在没有特定要求，且属于通用事实或最新新闻时，才输出 'web_search'。只输出目标单词，不要输出任何其他字符。"),
    ("human", "{question}"),
]) | llm | StrOutputParser()

retrieval_grader = ChatPromptTemplate.from_messages([
    ("system", "你是文档相关性评估员。只要文档包含与问题相关的关键词或语义，请只输出 'yes'，否则只输出 'no'。不要输出任何其他字符。"),
    ("human", "文档: {document}\n\n问题: {question}"),
]) | llm | StrOutputParser()

rag_generator = ChatPromptTemplate.from_template(
    "根据以下包含【参考来源】的上下文信息回答问题。如果不清楚，请直接说明。\n"
    "⚠️【极其重要的指令】：你必须在回答的最后，单起一段，明确列出你本次回答参考了哪些【参考来源】的文件名或路径，实现知识溯源！\n\n"
    "上下文: {context}\n\n问题: {question}"
) | llm | StrOutputParser()

hallucination_grader = ChatPromptTemplate.from_messages([
    ("system", "你是幻觉检测员。评估回答是否完全依赖给定的文档。是则只输出 'yes'，否则只输出 'no'。不要输出任何其他字符。"),
    ("human", "文档: {documents}\n\n回答: {generation}"),
]) | llm | StrOutputParser()

answer_grader = ChatPromptTemplate.from_messages([
    ("system", "你是答案质检员。评估回答是否有效解决了用户问题。是则只输出 'yes'，否则只输出 'no'。不要输出任何其他字符。"),
    ("human", "问题: {question}\n\n回答: {generation}"),
]) | llm | StrOutputParser()

question_rewriter = ChatPromptTemplate.from_messages([
    ("system", "你是一个优化检索的专家。请重写输入问题，以获得更好的向量数据库匹配度。"),
    ("human", "原问题: {question}"),
]) | llm | StrOutputParser()


# --- 节点(Node)执行逻辑 ---
def retrieve(state):
    print("➡️ 节点: 知识库检索")
    documents = execute_hybrid_retrieval(state["question"], top_k=3)
    return {"documents": documents, "question": state["question"], "retries": state.get("retries", 0)}

def execute_hybrid_retrieval(query: str, top_k: int = 3) -> List[str]:
    """执行底层双路召回与 CrossEncoder 重排，并附带知识溯源"""
    # 👇 初始化模型
    init_models()
    connect_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        return []
    col = Collection(COLLECTION_NAME)
    
    query_embeddings = bge_m3_ef([query])
    recall_num = max(10, top_k * 3)
    
    dense_req = AnnSearchRequest(query_embeddings["dense"], "dense_vector", {"metric_type": "IP"}, limit=recall_num)
    sparse_req = AnnSearchRequest(query_embeddings["sparse"]._getrow(0), "sparse_vector", {"metric_type": "IP"}, limit=recall_num)
    
    # 👇 新增 "source" 到 output_fields 中
    results = col.hybrid_search(
        reqs=[dense_req, sparse_req], rerank=RRFRanker(), limit=recall_num, output_fields=["window_text", "source"]
    )[0]
    
    # 👇 同时提取文本和来源
    retrieved_items = [{"text": hit.entity.get("window_text"), "source": hit.entity.get("source")} 
                       for hit in results if hit.entity.get("window_text")]
    if not retrieved_items:
        return []

    # 依然只对文本内容进行 CrossEncoder 语义重排
    cross_inp = [[query, item["text"]] for item in retrieved_items]
    scores = cross_encoder.predict(cross_inp)
    
    ranked_results = sorted(zip(retrieved_items, scores), key=lambda x: x[1], reverse=True)
    
    # 👇 最终格式化：把来源标签直接打在文本段落的最前面
    final_docs = [f"【参考来源: {item['source']}】\n{item['text']}" for item, score in ranked_results[:top_k]]
    return final_docs


def web_search(state):
    print("➡️ 节点: Web 网络检索 (使用本地 Chrome)")
    query = state["question"]
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.bing.com/search?q={encoded_query}"
    
    try:
        raw_response = visit_url_with_chrome(search_url)
        res_dict = json.loads(raw_response)
        if res_dict.get("success"):
            # 👇 为公网搜索也打上规范的“来源标签”
            web_results = f"【参考来源: 互联网搜索引擎 Bing】\n{res_dict['content']}"
            print("✅ 成功从公网抓取内容")
        else:
            web_results = f"【参考来源: 系统报错】\n本地浏览器检索失败: {res_dict.get('error')}"
    except Exception as e:
        web_results = f"【参考来源: 系统报错】\n本地浏览器调用异常: {str(e)}"

    return {"documents": [web_results], "question": query, "retries": state.get("retries", 0)}

def grade_documents(state):
    print("➡️ 节点: 评估检索文档相关性")
    filtered_docs = []
    for d in state["documents"]:
        score = retrieval_grader.invoke({"question": state["question"], "document": d}).strip().lower()
        if "yes" in score:  # 👈 改成了文本包含判断
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": state["question"], "retries": state.get("retries", 0)}

def generate(state):
    print("➡️ 节点: 基于文档生成答案")
    generation = rag_generator.invoke({"context": "\n\n".join(state["documents"]), "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation, "retries": state.get("retries", 0)}

def transform_query(state):
    print("➡️ 节点: 重写查询问题")
    better_question = question_rewriter.invoke({"question": state["question"]})
    retries = state.get("retries", 0) + 1
    return {"documents": state["documents"], "question": better_question, "retries": retries}

# --- 边(Edges)决策逻辑 ---
def route_question(state):
    source = router.invoke({"question": state["question"]}).strip().lower()
    if "web_search" in source:  # 👈 改成了文本包含判断
        print("🧭 路由: 走向 Web Search")
        return "web_search"
    print("🧭 路由: 走向 私有知识库 RAG")
    return "vectorstore"

def decide_to_generate(state):
    if not state["documents"]:
        print("🧭 决策: 没有相关文档，需要重写问题")
        return "transform_query"
    print("🧭 决策: 文档相关，准备生成答案")
    return "generate"

def grade_generation_v_documents_and_question(state):
    if state.get("retries", 0) > 2: # 防止无限死循环
        print("🧭 决策: 达到最大重试次数，直接返回结果")
        return "useful"
        
    score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]}).strip().lower()
    if "yes" in score:  # 👈 改成了文本包含判断
        print("🧭 决策: 未检测到幻觉，检查是否回答了问题")
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).strip().lower()
        if "yes" in answer_score:
            print("🧭 决策: 答案完美，结束循环！")
            return "useful"
        else:
            print("🧭 决策: 答案未回答问题，需要重写")
            return "not useful"
    else:
        print("🧭 决策: 检测到幻觉！拒接输出，重写问题再试")
        return "not supported"

# ---------------------------------------------------------
# 3. 编译 Adaptive RAG 图 (Graph)
# ---------------------------------------------------------
workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
# 让重写后的问题重新走一遍 router，该去 web 去 web，该去库去库
workflow.add_conditional_edges("transform_query", route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_conditional_edges(
    "generate", 
    grade_generation_v_documents_and_question, 
    {"not supported": "generate", "useful": END, "not useful": "transform_query"}
)

app = workflow.compile()

def run_adaptive_rag(query: str) -> str:
    debug_log(f"🚀 [run_adaptive_rag] 收到查询请求: {query}")
    try:
        # 真正调用你在上面代码中已经 compile() 好的 workflow app
        result = app.invoke({"question": query, "retries": 0})
        debug_log(f"✅ [run_adaptive_rag] 成功生成回答！")
        return result.get("generation", "未能生成有效回答。")
    except Exception as e:
        debug_log(f"❌ [run_adaptive_rag] 执行检索图结构时报错: {str(e)}")
        raise e
# ---------------------------------------------------------
# 4. 独立运行入口 (极度重要：用于彻底隔离环境和排错)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python rag_milvus.py [upload|search] [路径或问题]")
        sys.exit(1)
        
    action = sys.argv[1]
    target = sys.argv[2]
    
    if action == "upload":
        debug_log(f"⚡ [CLI独立运行] 开始执行入库: {target}")
        count = build_vector_store(target)
        print(f"\n===SUCCESS_COUNT===\n{count}\n===END===")
        
    elif action == "search":
        debug_log(f"⚡ [CLI独立运行] 开始执行检索: {target}")
        ans = run_adaptive_rag(target)
        print(f"\n===SUCCESS_ANS===\n{ans}\n===END===")