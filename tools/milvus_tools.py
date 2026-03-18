import asyncio
from langchain_core.tools import StructuredTool

async def get_stdio_milvus_tools():
    """
    LangGraph Supervisor 模式：直接将底层的 RAG 状态图包装为原生工具。
    彻底抛弃 MCP 协议与独立子进程。
    """
    
    async def rag_upload_file_async(path: str) -> str:
        print(f"\n[🧠 Supervisor 调度] 正在将入库任务委派给底层的 RAG 引擎，请观察终端日志...")
        try:
            # 🚀 绝对关键：必须在这里（纯同步线程内部）局部导入！
            def run_upload():
                from app.code_agent.rag.rag_milvus import build_vector_store
                return build_vector_store(path)
            
            count = await asyncio.to_thread(run_upload)
            return f"成功导入！文件已被切分为 {count} 个向量块。"
        except Exception as e:
            import traceback
            return f"导入失败: {str(e)}\n{traceback.format_exc()}"

    async def rag_search_async(query: str) -> str:
        print(f"\n[🧠 Supervisor 调度] 正在将复杂检索委派给内层 Adaptive RAG 智能体...")
        try:
            # 🚀 绝对关键：必须在这里（纯同步线程内部）局部导入！
            def run_graph(q):
                from app.code_agent.rag.rag_milvus import app as rag_graph
                result = rag_graph.invoke({"question": q, "retries": 0})
                return result.get("generation", "未能生成有效回答。")
            
            ans = await asyncio.to_thread(run_graph, query)
            return ans
        except Exception as e:
            import traceback
            return f"检索失败: {str(e)}\n{traceback.format_exc()}"

    # 封装为 LangChain 原生结构化工具，供外层 Supervisor (Qwen) 调用
    upload_tool = StructuredTool.from_function(
        coroutine=rag_upload_file_async,
        name="rag_upload_file",
        description="将指定路径下的文本文件(.txt)或整个目录导入并向量化到Milvus私有知识库中"
    )
    
    search_tool = StructuredTool.from_function(
        coroutine=rag_search_async,
        name="rag_search",
        description="高级知识库专家工具。将问题交给内部 RAG 智能体处理，自动完成检索、重排与生成。"
    )
    
    return [upload_tool, search_tool]