import pandas as pd
df = pd.read_csv('labeldata.csv',usecols=['评论标题', '评论内容', '标签'])
# 随机抽取二十条数据，并且分割为两个文件,一个是挑选出来的，一个是剩余的
df_sample = df.sample(n=20)
df_sample.to_csv('sample.csv',index=False)
df_rest = df.drop(df_sample.index)
df_rest.to_csv('rest.csv',index=False)
