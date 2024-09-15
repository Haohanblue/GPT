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
llm = OpenAI(model="gpt-4o")

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
            description="查询数据资料"
        ),
    )
]
#创建agent
agent = ReActAgent.from_tools(query_engine_tools,llm=llm,verbose=True)
template = """以下是某位同学的参加义修队的报名表，请你先搜索义修队的特征，结合义修队对人的要求对其进行评价，
从技术能力、服务态度、加入意愿、性格特征、填写质量等五个维度是否和义修队相符打1-100分打分，并最终给出综合得分。
提交报名表的是18或19岁的大一或大二的中国学生，你要能够读出并判断哪些句子是谦虚，哪些是过于自大狂妄，哪些符合实际，哪些又是过于夸大其词。
并给出是否录取的建议，如果不是请说明理由，如果是，请给出他应该去义修队哪个部门的意见。
姓名	王悦宪	性别	女	年级	22级	专业	计算机类	


学号	42211012	电话	13731260271	QQ号	3288749273	
第一意向部门	办公室□       宣传部□       技术部√	是否服从调剂	是	
自我介绍及评价	哈喽义修队的学长学姐大家好！我叫王悦宪，来自河北。我是一个开始的时候社恐，认识之后直接变成社牛的一个人设！（直接就是一个搞笑女的人设哈哈哈哈）家里开过电脑店，耳濡目染对计算机相关的东西比较感兴趣！很希望加入义修队和学长学姐一起学习，一起成长，一起帮助大家！
对电脑义务维修工作的看法	在我看来这项工作非常重要，可以帮助到很多对电脑维修知识不太熟悉但是又遇到困难的同学。比如疫情期间有的同学电脑出了一些自己解决不了的问题又没办法出校维修或者可以出校的时候没时间出校维修，义修队就可以发挥到很大很大的作用！我认为这项工作具体内容包括但不限于拆机清理、重装系统、硬件检查、软件安装等方面。我本人在刚开的时候也遇到了一些以我的能力解决不了的问题，也是有咨询学长学姐，真的很感谢！

（希望不是废话哟）
为什么想加入义修队	因为刚开学的时候寻求过学长学姐的帮助，也想培养自己的能力，帮助其他同学，同时也想从中学到更多知识。我们家开过电脑店，那时候经常看家人做拆卸电脑、设置系统、硬件更换之类的工作，因为上学原因自己学会的不是特别多，只是简单的更换过硬盘装过系统，本人也是计算机类的，也对这方面也是比较感兴趣，想学到很多东西！
想要从义修队学到什么	熟练掌握系统安装及调试，学会分析故障原因和解决办法，了解计算机硬件的功能和性能及拆卸和调试。
用最简洁的方式让我们记住你	学长学姐好，我叫王悦宪，愉悦的悦，宪法的宪，咱就是说真～的很想加入义修队的一个大状态！想学习！想帮助同学！直接就是一个想要捣鼓电脑的一个大状态！康康我！


"""
agent.chat(template)