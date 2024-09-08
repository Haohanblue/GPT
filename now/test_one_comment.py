import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
import pandas as pd
from langchain.prompts import PromptTemplate
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
忘记之前的所有指令。你是一个金融文本分析专家，请对以下股吧评论进行情感分析，并输出积极（positive）、消极（negative）、中性（neutral）三种情感的概率分布。

股吧评论:
{message}

工作步骤: 
- 第一步，请逐一注意以下几点并做详细分析，同时初步评估股吧评论的情感分布（即，积极、消极、中性的概率）。
注意1：请特别注意任何虚拟语气（irrealis mood）的使用。
注意2：请特别注意任何修辞手法（讽刺、否定断言等）的使用。
注意3：请关注说话者的情感，这里要结合虚拟语气和修辞手法评估真实的情感，而不是字面意思。
注意4：请关注股票的代码/标签/评论的话题。
注意5：请特别注意时间表达、价格和其他未明说、隐含的意思。
- 第二步，基于第一步的详细分析，对情感（积极、消极、中性）分布做出最终的总体评价。

输出格式要求:
- Output the sentiments distribution in the second step strictly following JSON format with keys of positive, negative,  neutral and reason(as String).
- The sum of probabilities of sentiments must be equal to 1. 
- Answer里输出只有JSON格式的字符串，key为positive、negative、neutral，值为对应的概率，以及原因(reason)，对应的是中文字符串文本描述这样预测的原因。
- Answer里输出只有JSON格式的字符串，key为positive、negative、neutral，值为对应的概率，以及原因(reason)，对应的是中文字符串文本描述这样预测的原因。
"""
def read_data():
    df = pd.read_csv('sample.csv',usecols=['评论标题', '评论内容'])
    return df
data = read_data()
import time
total_rows = data.shape[0]
count = 0
for index, row in data.iterrows():
    # 读取Name列的内容
    title = row['评论标题']
    content = row['评论内容']
    message = {'title': title, 'content': content}
    # 进行处理
    print(f"message: ['title':{title}, 'content':{content}]")
    #time.sleep(1)
    try:
        prompt_template = PromptTemplate(template= template, input_variables=["message"])
        prompt = prompt_template.format(message = message)
        agent.chat(template)
        # 打印进度
        current_progress = (count + 1) / total_rows * 100
        count += 1
        print(f"Progress: {current_progress:.2f}%")
    except:
        continue

# agent.chat(template)