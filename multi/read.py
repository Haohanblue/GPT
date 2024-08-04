import pandas as pd
import random

# 读取CSV文件并选择指定列
df = pd.read_csv('./multi/labeldata.csv', usecols=['id', 'stock_code', 'replay_num', 'post_title', 'read_num'])

# 将数据按股票代码分组，每组数据为字典对象
grouped = df.groupby('stock_code').apply(lambda x: x.to_dict(orient='records'))
grouped = grouped.to_dict()

# 设置随机种子
random.seed(42)
# 指定要抽取的组数
num_groups_to_sample = 2
# 获取所有组的键（股票代码）
all_groups = list(grouped.keys())
# 随机抽取指定数量的组
sampled_group_keys = random.sample(all_groups, num_groups_to_sample)
# 获取抽取的组对应的数据
sampled_groups = {key: grouped[key] for key in sampled_group_keys}
def calculate_probability(sampled_groups):
    predicted = []
    labels = []
    for key, value in sampled_groups.items():
        print(f"key:{key};message: {value}")
    return predicted, labels
calculate_probability(sampled_groups)