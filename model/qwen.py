import os

from langchain_openai import ChatOpenAI

# llm_qwen = ChatOpenAI(
#     model="deepseek-coder",
#     base_url="https://api.deepseek.com/v1",
#     api_key=os.environ.get("DeepSeek_Key"),
#     streaming=True,
# )
llm_qwen = ChatOpenAI(
    model="deepseek-coder",  # DeepSeek 模型名称，可选：deepseek-chat/deepseek-coder
    base_url="https://api.deepseek.com/v1",  # DeepSeek 原生 API 地址
    api_key=os.environ.get("DeepSeek_Key"),  # 替换为自己的 API Key
    streaming=True,
)
