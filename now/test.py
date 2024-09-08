import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
#配置openai的api key和模型  

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(model="gpt-3.5-turbo")

#加载索引
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False
if index_loaded == False:
    #设置数据源
    documents = SimpleDirectoryReader('data').load_data()
    #创建索引
    index = VectorStoreIndex.from_documents(documents)
    #将索引存储到磁盘
    index.storage_context.persist()
#创建查询引擎
engine = index.as_query_engine(similarity_top_k=3)
#创建查询引擎工具
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="commentsQuery",
            description="用于查询股票评论数据"
        ),
    )
]
#创建agent
agent = ReActAgent.from_tools(query_engine_tools,llm=llm,verbose=True)
template = """
忘记之前的所有指令。
你是一个金融文本分析专家，下面有某支股票的股吧评论数据列表，列表里每一个评论元素包括评论标题(post_title)，阅读量(read_num)，回复量(replay_num)等数据。
请对该股票下的所有评论数据进行分析，要逐一遍历列表当中每一个元素，综合评判后由评论数据得出未来第五个交易日相较于当前情况下的变动情况，输出股价上涨(up)、下跌(down)的概率分布，概率之和必须为1。
股吧评论数据列表:
{message}
工作步骤: 
- 第一步，请逐一对评论数据列表中每一个元素进行单独分析。对该条评论元素标题(post_title)的情感（积极、消极、中性）分布做出最终的总体评价，同时结合阅读量(read_num)、回复量(replay_num)对该条评论的质量进行评估。
对于标题：Attention 1: 请特别注意使用的虚拟语气。
Attention 2: 请特别注意使用的修辞（如讽刺、否定陈述等）。
Attention 3: 请关注说话者的情绪，而不是第三方。
Attention 4: 请关注股票代码/标签/话题，而不是其他实体。
Attention 5: 请特别注意时间表达、价格和其他未说明的事实。
- 第二步，综合每一条评论元素，得出最终对该股票未来第五个交易日相较于当前情况下的变动情况分布（上涨的概率）。

输出格式要求:
- 输出格式必须是以JSON格式的，key为up和down，值为分别对应上涨和下跌的概率, 以及原因(reason)，对应的是中文字符串文本描述这样预测的原因。
- 概率值必须是0-1之间的数字，并且上涨和下跌的概率之和必须为1 
- 请你直接对该数据进行分析处理并得出结果而不是给我python代码，结果只输出该JSON格式的对象，不要有其他内容。
- 我需要对浦发银行，也就是股票代码为600000的评论数据进行分析。
"""
agent.chat(template)