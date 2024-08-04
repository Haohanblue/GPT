import re
import json
from zhipuai import ZhipuAI
from langchain.prompts import PromptTemplate
import pandas as pd
import random
from tqdm import tqdm
from datetime import datetime
ZHIPUAI_API_KEY = "3b9892afa93ed95c01c62e34e244d1ea.hxyr2FYltUR09i83"
prompt_template_string = """忘记之前的所有指令。
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
- 输出格式必须是以JSON格式的，key为up和down，值为分别对应上涨和下跌的概率.
- 概率值必须是0-1之间的数字，并且上涨和下跌的概率之和必须为1 
- 请你直接对该数据进行分析处理并得出结果而不是给我python代码，结果只输出该JSON格式的对象，不要有其他内容。

"""
def zhipu_chat_complete(prompt, model = "glm-4-air"):
    """
    使用智谱的glm模型进行对话"""
    # 初始化智谱的API客户端
    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
    # 调用智谱的API进行对话
    response = client.chat.completions.create(
        model = model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1024
    )
    # 返回智谱的API的响应结果
    answer = response.choices[0].message.content
    return answer
# message = [
#     {"title": "现在浦发银行看起好像涨了很多，其实净资产打四折甩卖。只是以前超跌，现在只是恢复性", "reading_count": 150, "reply_count": 12},
#     {"title": "股市行情有点反弹迹象，但是要小心泡沫", "reading_count": 1, "reply_count": 0},
#     {"title": "今天大盘整体上涨，个股普遍红盘", "reading_count": 7, "reply_count": 0}
# ]
# # 将评论对象列表格式化为字符串
# message_str = json.dumps(message, ensure_ascii=False)

# prompt_template = PromptTemplate(template= prompt_template_string, input_variables=["message"])
# prompt = prompt_template.format(message = message_str)
# p = zhipu_chat_complete(prompt = prompt, model = "GLM-4-Air")
# print(p)

def read_data(is_all):
    is_all = is_all
    # 读取CSV文件并选择指定列
    df = pd.read_csv('./multi/labeldata.csv', usecols=['id', 'stock_code', 'replay_num', 'post_title', 'read_num'])
    # 将数据按股票代码分组
    grouped = df.groupby('stock_code')
    # 创建一个空字典来存储分组后的数据
    grouped_dict = {}
    # 遍历每个分组
    for stock_code, group in grouped:
        # 转换每个分组为字典对象，并排除分组列
        grouped_dict[stock_code] = group[['id', 'replay_num', 'post_title', 'read_num']].to_dict(orient='records')
    if not is_all:
        # 设置随机种子
        random.seed(80)
        # 指定要抽取的组数
        num_groups_to_sample = 5
        # 获取所有组的键（股票代码）
        all_groups = list(grouped_dict.keys())
        # 随机抽取指定数量的组
        sampled_group_keys = random.sample(all_groups, num_groups_to_sample)
        sampled_groups = {key: grouped_dict[key] for key in sampled_group_keys}
        return sampled_groups
    if is_all == True:
        sampled_group_keys = list(grouped_dict.keys())
        # 获取抽取的组对应的数据
        sampled_groups = {key: grouped_dict[key] for key in sampled_group_keys}
        return sampled_groups

def calculate_probability(sampled_groups):
    predicted = []
    for key, value in tqdm(sampled_groups.items(), desc="Processing groups"):
        try:
            message = value
            message_str = json.dumps(message, ensure_ascii=False)
            prompt_template = PromptTemplate(template= prompt_template_string, input_variables=["message"])
            prompt = prompt_template.format(message = message_str)
            p = zhipu_chat_complete(prompt = prompt, model = "GLM-4-Air")
            p = p.split("json")[1].split("`")[0]
            json_obj = json.loads(p)
            item = {}
            item["stock_code"] = key
            item["up"] = json_obj["up"]
            item["down"] = json_obj["down"]
            print(item)
            predicted.append(item)
        except:
            continue
    return predicted

## 是否为部分测试？
test_env = True
if test_env == True:
    data = calculate_probability(read_data(is_all=True))
    # Convert list to DataFrame
    df = pd.DataFrame(data)
    # Save to CSV

    # Get current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'./multi/output_{current_time}.csv'
    df.to_csv(output_path, index=False)
else:
    calculate_probability(read_data(is_all=False))










# def extract_sentiments_json(text):
#     # Regular expression to find JSON object
#     json_pattern = r'\{.*?\}'
#     # Find all JSON objects in the text
#     json_matches = re.findall(json_pattern, text, re.DOTALL)
#     # Parse the JSON strings
#     try:
#         json_data = json.loads(json_matches[0])
#         if ("positive" in json_data) and ("negative" in json_data) and ("neutral" in json_data):
#             return json_matches[0]
#         else:
#             return None
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return None
    
# def label_sentiment(message):
#     prompt_template = PromptTemplate(template= prompt_template_string, input_variables=["message"])
#     prompt = prompt_template.format(message = message)
#     p = zhipu_chat_complete(prompt = prompt, model = "GLM-4-Air")
#     sentiment_dict = json.loads(extract_sentiments_json(p))
#     # 获取情感倾向的最大值
#     max_sentiment = max(sentiment_dict.values())
#     # 判断情感倾向并返回相应的值
#     if sentiment_dict['positive'] == max_sentiment:
#         return 1
#     elif sentiment_dict['negative'] == max_sentiment:
#         return -1
#     else:
#         return 0

# # 读取CSV文件
# df = pd.read_csv('labeldata.csv')
# # 查看数据的前几行
# df.head()
# # 随机抽样50条数据，设置随机种子
# sampled_data = df.sample(n=200, random_state=22)
# predicted = []
# labels = []

# # 循环获取DataFrame中的数据
# import time
# total_rows = sampled_data.shape[0]
# count = 0
# for index, row in sampled_data.iterrows():
#     # 读取Name列的内容
#     message = row['评论内容']
#     # 进行处理
#     print(f"message: {message}")
#     #time.sleep(1)
#     try:
#         predicted_label = label_sentiment(message=message)
#         print(f"predicted_label: {predicted_label}")
#         predicted.append(predicted_label)
#         labels.append(row["label"])
#         print("acctual label: " + str(row["label"]))
        
#         # 打印进度
#         current_progress = (count + 1) / total_rows * 100
#         count += 1
#         print(f"Progress: {current_progress:.2f}%")
#     except:
#         continue
