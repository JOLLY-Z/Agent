import json
import time
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

mcp = FastMCP()

class BrowserResponse(BaseModel):
    success: bool
    url: str
    content: str
    error: Optional[str] = None

def get_chrome_instance():
    options = webdriver.ChromeOptions()
    # 如果你想看着浏览器自动操作，就保持下面这行被注释掉；如果嫌烦想让它后台静默运行，就取消注释
    # options.add_argument('--headless') 
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage') # 解决资源限制导致的崩溃
    
    # 填入你本地正确的 chromedriver 绝对路径
    chromedriver_path = r"D:\Jolly\chromedriver-win64\chromedriver.exe"
    service = Service(executable_path=chromedriver_path)
    
    driver = webdriver.Chrome(service=service, options=options)
    # 设置隐式等待，防止网页加载慢导致找不到元素
    driver.implicitly_wait(5) 
    return driver

@mcp.tool(name="visit_url_with_chrome", description="使用本地 Chrome 浏览器打开指定的 URL，并获取网页上的可见纯文本内容")
def visit_url_with_chrome(
    url: Annotated[str, Field(
        description="要访问的完整网页链接。如果是执行搜索任务，请直接构造搜索引擎的URL，例如：https://www.bing.com/search?q=惠州天气 或 https://www.sogou.com/web?query=惠州天气", 
        examples=["https://www.bing.com/search?q=惠州天气", "https://news.ycombinator.com/"]
    )]
) -> str:
    """
    Agent 通用上网工具。
    """
    driver = None
    try:
        driver = get_chrome_instance()
        driver.get(url)
        
        # 强制等待 2 秒，确保动态渲染的内容（如天气卡片、JS加载的新闻）能加载出来
        time.sleep(2) 
        
        # 直接抓取整个网页的可见文本内容
        body_text = driver.find_element(By.TAG_NAME, "body").text
        
        # 截取前 3000 个字符，防止网页太大撑爆大模型的上下文
        truncated_text = body_text[:3000] if len(body_text) > 3000 else body_text
        
        response = BrowserResponse(
            success=True,
            url=url,
            content=truncated_text
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
        
    except Exception as e:
        error_response = BrowserResponse(
            success=False,
            url=url,
            content="",
            error=f"浏览器访问失败: {str(e)}"
        )
        return json.dumps(error_response.model_dump(), ensure_ascii=False)
        
    finally:
        # 无论成功失败，确保浏览器被关闭，释放内存
        if driver:
            driver.quit()

if __name__ == '__main__':
    # 启动 MCP 标准输入输出服务
    mcp.run(transport="stdio")