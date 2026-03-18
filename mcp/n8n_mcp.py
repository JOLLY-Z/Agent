import json
import requests
from typing import Optional
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("N8nWorkflows")

# 请替换为你 n8n 实际的 Webhook URL (注意区分 Test URL 和 Production URL)
N8N_SCRAPE_WEBHOOK = "http://localhost:5678/webhook/scrape-ai-news"
N8N_REPORT_WEBHOOK = "http://localhost:5678/webhook/generate-ai-report"

class N8nResponse(BaseModel):
    success: bool
    workflow: str
    message: str
    error: Optional[str] = None

@mcp.tool(name="trigger_ai_news_scraper", description="触发网页爬虫工作流，去网站抓取最新的AI资讯并保存到Notion数据库。当用户要求生成最新AI日报时，必须【先】调用此工具更新基础数据。")
def trigger_ai_news_scraper() -> str:
    """
    触发 n8n 爬虫工作流获取最新资讯。
    """
    try:
        # 设置较大的 timeout，因为爬虫和写入 Notion 可能需要一些时间
        response = requests.post(N8N_SCRAPE_WEBHOOK, timeout=120)
        response.raise_for_status()
        
        result = N8nResponse(
            success=True,
            workflow="网页爬虫",
            message="成功抓取最新AI资讯并已同步至 Notion1 数据库。"
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)
        
    except Exception as e:
        error_result = N8nResponse(
            success=False,
            workflow="网页爬虫",
            message="抓取资讯失败",
            error=str(e)
        )
        return json.dumps(error_result.model_dump(), ensure_ascii=False)

@mcp.tool(name="trigger_ai_daily_report", description="触发AI日报生成工作流。该工作流会提取Notion中最新的AI资讯，利用AI筛选总结生成日报，并自动生成音频和网页截图。前提：需确保资讯数据已更新。")
def trigger_ai_daily_report() -> str:
    """
    触发 n8n 日报生成与多媒体处理工作流。
    """
    try:
        # 总结、TTS音频生成和截图处理极其耗时，这里将 timeout 设置得长一些
        response = requests.post(N8N_REPORT_WEBHOOK, timeout=300)
        response.raise_for_status()
        
        result = N8nResponse(
            success=True,
            workflow="日报生成",
            message="成功生成最新AI日报，并已完成音频TTS与截图生成，结果已保存至 Notion2。"
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)
        
    except Exception as e:
        error_result = N8nResponse(
            success=False,
            workflow="日报生成",
            message="生成AI日报流程出现异常",
            error=str(e)
        )
        return json.dumps(error_result.model_dump(), ensure_ascii=False)

if __name__ == '__main__':
    # 启动 MCP 标准输入输出服务
    mcp.run(transport="stdio")