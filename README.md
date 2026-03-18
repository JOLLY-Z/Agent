# Agent
An AI intelligent agent system developed based on the Supervisor architecture. It uses the MCP protocol to standardize the access to underlying tools, has built-in adaptive RAG and MongoDB context memory functions, and realizes the interaction between natural language and databases, operating systems, and public network data.



## 配置

首先在modle里配置自己模型的key

然后是rag文件里rag.milvus里的milvus与模型的相关配置

要调用browser_tool还需要安装与自己goole版本对应的goole_driver,并在文件里配置.exe文件路径



## 相关python依赖

pip install -r requirements.txt



## N8N工作流

在导入json文件并配置自己的notion，运行网页爬虫工作流还要记得部署firecrawl

并在tool/n8n_tools.py中连接自己的webhook url

