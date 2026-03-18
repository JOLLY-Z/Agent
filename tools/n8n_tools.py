# 假设你在 n8n_tools.py 同级目录写了一个加载器，或者直接利用你的 mcp.py
from app.code_agent.utils.mcp import create_mcp_stdio_client

async def get_stdio_n8n_tools():
    params = {
        "command": "python",
        "args": [
            "D:/conda_envs/ai-agent-env/Lib/site-packages/app/code_agent/mcp/n8n_mcp.py",
        ]
    }

    client, tools = await create_mcp_stdio_client("n8n_tools", params)

    return tools
