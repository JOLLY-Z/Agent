import sys
import json
from typing import Annotated, Optional
from pydantic import BaseModel, Field

# =====================================================================
# 🛡️ 终极护盾：在主线程最顶层预加载，彻底消灭管道堵塞与线程死锁！
# =====================================================================
_original_stdout = sys.stdout
sys.stdout = sys.stderr  # 强行拦截所有污染输出，保护 MCP 协议

try:
    from app.code_agent.rag.rag_milvus import build_vector_store, run_adaptive_rag, init_models
    
    # 🚀 关键魔法：在 MCP 启动前，提前在主线程把大模型加载进内存！
    # 这样不仅绝对不会引发死锁，而且后续每次被 Agent 调用时都是“瞬间响应”
    init_models()
except Exception as e:
    import traceback
    print(f"初始化模型严重失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
finally:
    sys.stdout = _original_stdout

from mcp.server.fastmcp import FastMCP
mcp = FastMCP()

class RAGResponse(BaseModel):
    success: bool
    query: str
    data: Optional[list[str]] = None
    error: Optional[str] = None

@mcp.tool(name="rag_upload_file", description="将指定路径下的文本文件(.txt)或整个目录导入并向量化到Milvus私有知识库中")
def rag_upload_file(
    path: Annotated[str, Field(description="需要导入知识库的文件或目录的绝对路径或相对路径")]
) -> str:
    try:
        # 因为模型早就预热好了，这里只需专心执行入库即可
        chunks_count = build_vector_store(path)
        response = RAGResponse(success=True, query=path, data=[f"成功导入！文件已被切分为 {chunks_count} 个向量块。"])
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        import traceback
        error_response = RAGResponse(success=False, query=path, error=f"{str(e)}\n{traceback.format_exc()}")
        return json.dumps(error_response.model_dump(), ensure_ascii=False)

@mcp.tool(name="rag_search", description="高级知识库专家工具。在私有知识库中搜索相关背景信息，并自带逻辑路由等能力。")
def rag_search(
    query: Annotated[str, Field(description="需要向知识专家提问的复杂自然语言问题")]
) -> str:
    try:
        ans = run_adaptive_rag(query)
        response = RAGResponse(success=True, query=query, data=[ans])
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        import traceback
        error_response = RAGResponse(success=False, query=query, error=f"{str(e)}\n{traceback.format_exc()}")
        return json.dumps(error_response.model_dump(), ensure_ascii=False)

if __name__ == '__main__':
    mcp.run(transport="stdio")