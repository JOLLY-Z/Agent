import asyncio
import time

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from app.code_agent.model.qwen import llm_qwen
from app.code_agent.tools.browser_tools import get_stdio_browser_tools
from app.code_agent.tools.mysql_tools import get_stdio_mysql_tools
from app.code_agent.tools.milvus_tools import get_stdio_milvus_tools
from app.code_agent.tools.shell_tools import get_stdio_shell_tools
from app.code_agent.tools.n8n_tools import get_stdio_n8n_tools

from langchain_core.tools import StructuredTool

from langchain_core.messages import ToolMessage  # 确保顶部引入了 ToolMessage

def format_debug_output(step_name: str, content: str, is_tool_call = False) -> None:
    if is_tool_call:
        print(f'🔄 【工具调用】 {step_name}')
        print("-" * 40)
        print(content.strip())
        print("-" * 40)
    else:
        print(f"💭 【{step_name}】")
        print("-" * 40)
        print(content.strip())
        print("-" * 40)


async def run_agent():
    # memory = FileSaver()
    memory = MemorySaver()
    shell_tools = await get_stdio_shell_tools()
    milvus_tools = await get_stdio_milvus_tools()
    n8n_tools = await get_stdio_n8n_tools()
    browser_tools = await get_stdio_browser_tools()
    mysql_tools = await get_stdio_mysql_tools()

    tools = n8n_tools + browser_tools + mysql_tools + milvus_tools + shell_tools
    
# ---------------------------------------------------------
# 🛠️ 兼容性修复 V4：重新实例化工具，彻底绕开 Pydantic 冻结限制
# ---------------------------------------------------------
    patched_tools = []
    for t in tools:
        if isinstance(t, StructuredTool):
            # 1. 提取原始的同步和异步核心执行函数
            original_func = t.func
            original_coro = t.coroutine

            # 2. 包装同步函数，将 List 转为 String
            def make_sync(func):
                if not func: return None
                def wrapper(*args, **kwargs):
                    res = func(*args, **kwargs)
                    if isinstance(res, list):
                        return "\n".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in res])
                    return str(res)
                return wrapper

            # 3. 包装异步函数，将 List 转为 String
            def make_async(coro):
                if not coro: return None
                async def wrapper(*args, **kwargs):
                    res = await coro(*args, **kwargs)
                    if isinstance(res, list):
                        return "\n".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in res])
                    return str(res)
                return wrapper

            # 4. 创建一个全新的 StructuredTool 实例，Pydantic 将视其为合法对象
            new_tool = StructuredTool(
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,  # 继承原有的参数定义，完美通过类型校验
                func=make_sync(original_func),
                coroutine=make_async(original_coro),
            )
            patched_tools.append(new_tool)
        else:
            # 如果不是 StructuredTool，直接保留原样
            patched_tools.append(t)

    # 替换掉原本的 tools 列表
    tools = patched_tools
# ---------------------------------------------------------

# 1. 重构系统提示（直接定义 SystemMessage，无需 PromptTemplate 中转）
    system_message = SystemMessage(content="""# 角色
你是一名优秀的工程师，你的名字叫做Bot。

# 核心纪律
当你调用 `rag_search` 等工具并得到包含【参考来源】或引用出处的结果时，你**必须**在最终回答的末尾，原封不动、一字不落地将这些来源附上！绝不允许省略、过滤或合并溯源信息！""")
    
    agent = create_agent(
        model=llm_qwen,
        tools=tools,
        checkpointer=memory,
        debug=True,
        #prompt=SystemMessage(content=prompt.format(name="Bot")),
    )
	

    config = RunnableConfig(configurable={"thread_id": 1}, recursion_limit=100)

    while True:
        user_input = input("用户: ").strip()# 去掉首尾空格

        if user_input.lower() == "exit":
            break

        print("\n🤖 助手正在思考...")
        print("=" * 60)
#         user_prompt = \
# f"""# 要求
# 执行任务之前先使用 query_rag 工具查询知识库，根据知识库中的知识执行任务
#
# # 用户问题
# {user_input}"""
        user_prompt = user_input

        iteration_count = 0
        

        start_time = time.time()
        last_tool_time = start_time

        messages = [
			system_message,  # 系统提示
			HumanMessage(content=user_prompt)  # 用户输入
		]
        print(f"DEBUG: [INIT] Initial messages length = {len(messages)}, type: {type(messages[0]) if messages else 'empty'}")
        print(f"DEBUG: [INIT] First message content: {messages[0].content[:50]}{'...' if len(messages[0].content) > 50 else ''}")


        # 在 /Users/sam/llm/.temp/project 目录下创建一个名为 vue2-test 的 vue2 项目
        # 在虚拟机 /home/sam.linux/nginx/uploads/ 目录下，创建文件夹 test4
        async for chunk in agent.astream(input={"messages": messages}, config=config):
            iteration_count += 1

            print(f"\n📊 第 {iteration_count} 步执行：")
            print("-" * 30)

            items = chunk.items()

            for node_name, node_output in items:
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                format_debug_output("AI思考", msg.content)
                            else:
                                for tool in msg.tool_calls:
                                    format_debug_output("工具调用", f"{tool['name']}: {tool['args']}")

                        elif isinstance(msg, ToolMessage):
                            
                            print("="*80)
                            print(msg)
                            print("="*80)

                            tool_name = getattr(msg, "name", "unknown")
                            tool_content = msg.content
                            
                            current_time = time.time()
                            tool_duration = current_time - last_tool_time
                            last_tool_time = current_time

                            tool_result = f"""🔧 工具：{tool_name}
📤 结果：
{tool_content}
✅ 状态：执行完成，可以开始下一个任务
️⏱️ 执行时间：{tool_duration:.2f}秒"""

                            format_debug_output("工具执行结果", tool_result, is_tool_call=True)

                        else:
                            format_debug_output("未实现", f"暂未实现的打印内容: {chunk}")

        print()


if __name__ == '__main__':
    asyncio.run(run_agent())