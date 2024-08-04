import re
import json
from langchain.prompts import PromptTemplate
import pandas as pd
import openai
# 设置OpenAI的API密钥
openai.api_key="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"

prompt_template_string = """
Consider the message on a listing firm, please extract the sentiments(positive, negative, neutral)distribution (i.e., probabilities of its sentiments) from the message.\ 
The sum of probabilities of sentiments must be equal to 1.

The message:
{message}

Workflow: 
- First step, please pay special attention to the following points one by one, and at the same time evaluate the sentiments distribution of the message.
Attention 1: Please pay special attention to any irrealis mood used.
Attention 2: Please pay special attention to any rhetorics (sarcasm, negative assertion, etc.) used.
Attention 3: Please focus on the speaker sentiment, not a third party.
Attention 4: Please focus on the stock ticker/tag/topic, not other entities.
Attention 5: Please pay special attention to the time expressions, prices, and other unsaid facts.
Attention 6: Please pay attention to identifying whether the judgment on the future trend of the stock price in the language is positive,negative or neutral, rather than the sentiment.
- Second step, Based on the initial evaluations in the first step, make the final overall evaluation of the sentiments (positive, negative, neutral) distribution.

Output requirements:
- Output the sentiments distribution in the second step strictly following JSON format with keys of positive, negative, and neutral.
- The sum of probabilities of sentiments must be equal to 1. 
"""
def gpt_chat_complete(prompt,model="gpt-4"):
    """
    使用OpenAI的GPT模型进行对话
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

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
    p = gpt_chat_complete(prompt = prompt, model = "gpt-4")
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
sampled_data = df.sample(n=200, random_state=42)
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
    except Exception as e:
        print(e)
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