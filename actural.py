import re
import json
from zhipuai import ZhipuAI
from langchain.prompts import PromptTemplate
import pandas as pd

ZHIPUAI_API_KEY = "3b9892afa93ed95c01c62e34e244d1ea.hxyr2FYltUR09i83"
prompt_template_string = prompt_template_string = """
忘记你之前的所有指令。你是一位具有股票推荐经验的金融专家。，请对以下股吧评论进行情感分析，并输出该评论对于未来股价的积极（positive）、消极（negative）、中性（neutral）三种情感的概率分布。

股吧评论:
{message}

工作内容: 该评论对于该股未来走势的态度是积极（positive）、消极（negative）、中性（neutral）？

输出格式:
- Output the sentiments distribution in the second step strictly following JSON format with keys of positive, negative, and neutral.
- The sum of probabilities of sentiments must be equal to 1. 
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

def extract_sentiments_json(text):
    # Regular expression to find JSON object
    json_pattern = r'\{.*?\}'
    # Find all JSON objects in the text
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    # Parse the JSON strings
    try:
        json_data = json.loads(json_matches[0])
        if ("positive" in json_data) and ("negative" in json_data) and ("neutral" in json_data):
            return json_matches[0]
        else:
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
def label_sentiment(message):
    prompt_template = PromptTemplate(template= prompt_template_string, input_variables=["message"])
    prompt = prompt_template.format(message = message)
    p = zhipu_chat_complete(prompt = prompt, model = "GLM-4-Air")
    sentiment_dict = json.loads(extract_sentiments_json(p))
    # 获取情感倾向的最大值
    max_sentiment = max(sentiment_dict.values())
    # 判断情感倾向并返回相应的值
    if sentiment_dict['positive'] == max_sentiment:
        return 1
    elif sentiment_dict['negative'] == max_sentiment:
        return -1
    else:
        return 0

# 读取CSV文件
df = pd.read_csv('labeldata.csv')
# 查看数据的前几行
df.head()
# 随机抽样50条数据，设置随机种子
sampled_data = df.sample(n=200, random_state=22)
predicted = []
labels = []

# 循环获取DataFrame中的数据
import time
total_rows = sampled_data.shape[0]
count = 0
for index, row in sampled_data.iterrows():
    # 读取Name列的内容
    message = row['评论内容']
    # 进行处理
    print(f"message: {message}")
    #time.sleep(1)
    try:
        predicted_label = label_sentiment(message=message)
        print(f"predicted_label: {predicted_label}")
        predicted.append(predicted_label)
        labels.append(row["label"])
        print("acctual label: " + str(row["label"]))
        
        # 打印进度
        current_progress = (count + 1) / total_rows * 100
        count += 1
        print(f"Progress: {current_progress:.2f}%")
    except:
        continue
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# 假设labels是真实标签，predicted_labels是预测结果
labels = labels  # 真实标签
predicted_labels = predicted  # 预测结果

#labels = new_labels  # 真实标签
#predicted_labels = new_predicted  # 预测结果

# 计算准确率
accuracy = accuracy_score(labels, predicted_labels)

# 计算精确度
precision = precision_score(labels, predicted_labels, average='weighted')

# 计算召回率
recall = recall_score(labels, predicted_labels, average='weighted')

# 计算F1分数
f1 = f1_score(labels, predicted_labels, average='weighted')

# 计算混淆矩阵
conf_matrix = confusion_matrix(labels, predicted_labels)

# 计算ROC AUC分数（如果模型输出概率）
# roc_auc = roc_auc_score(labels, predicted_labels, multi_class='ovr')

# 打印结果
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: {conf_matrix}")
# print(f"ROC AUC: {roc_auc}")