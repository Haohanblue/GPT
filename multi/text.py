import pandas as pd
from textblob import TextBlob

# 示例数据
comments = [{"id": 510, "replay_num": 7, "post_title": "28本", "read_num": "372"}, {"id": 511, "replay_num": 3, "post_title": "三一重工周一行情分析：根据提供的数据，三一重工（600031.SH）在2024年", "read_num": "1359"}, {"id": 512, "replay_num": 4, "post_title": "昨天尾盘满仓进了，冲！周一涨停", "read_num": "393"}, {"id": 513, "replay_num": 0, "post_title": "三一重工本周融资净偿还5097.4万元，居工程机械板块第一", "read_num": "186"}, {"id": 514, "replay_num": 0, "post_title": "三一重工本周沪股通持股市值减少3690.72万元，居工程机械板块第五", "read_num": "145"}, {"id": 3485, "replay_num": 19, "post_title": "~ 供不应求，防癌疫苗15万亿大市场，24年业绩增长20倍；每隔8年还要重复需", "read_num": "3008"}, {"id": 3486, "replay_num": 0, "post_title": "楼主写的这么好，相信股票也好，点关注了", "read_num": "491"}, {"id": 3487, "replay_num": 18, "post_title": "和我上次分析的一样，呵呵。", "read_num": "1837"}, {"id": 3488, "replay_num": 1, "post_title": "护好盘，记一功", "read_num": "235"}, {"id": 3489, "replay_num": 18, "post_title": "三一股价比淮柴动力高就是股市最大的笑话！", "read_num": "1972"}, {"id": 3490, "replay_num": 3, "post_title": "那个大神能解释一下，为什么每个礼拜一都跌？唉", "read_num": "253"}, {"id": 3491, "replay_num": 0, "post_title": "一分钱玩一天？", "read_num": "202"}, {"id": 3492, "replay_num": 0, "post_title": "拉升起来了！稳住！加油！", "read_num": "247"}, {"id": 3493, "replay_num": 1, "post_title": "已出，不玩了", "read_num": "292"}, {"id": 3494, "replay_num": 3, "post_title": "持仓更新，迎接上涨", "read_num": "1551"}, {"id": 3495, "replay_num": 2, "post_title": "这80多万股东估计要套到天荒地老，肉烂在锅里，你们一起共存亡吧", "read_num": "525"}, {"id": 3496, "replay_num": 46, "post_title": "三一重工:三一重工股份有限公司关于股份回购实施结果暨股份变动的公告", "read_num": "8657"}, {"id": 3497, "replay_num": 7, "post_title": "三一重工北方长龙哪个好？", "read_num": "1436"}, {"id": 3498, "replay_num": 1, "post_title": "开会了，今天护盘，不会跌！！！！", "read_num": "235"}, {"id": 3499, "replay_num": 1, "post_title": "没人跟，看你呼呼的能拉多久", "read_num": "215"}, {"id": 3500, "replay_num": 0, "post_title": "知道你是干什么的", "read_num": "183"}, {"id": 3501, "replay_num": 0, "post_title": "今天上午肯定能红几分钟，下午跌停不能怪我[大笑]", "read_num": "260"}, {"id": 3502, "replay_num": 0, "post_title": "制造业标杆估值快被养猪的超过了哈哈", "read_num": "139"}, {"id": 3503, "replay_num": 0, "post_title": "下来了啊，又可以做T了", "read_num": "130"}, {"id": 3504, "replay_num": 8, "post_title": "15个月税费两千多元，交易有点频繁。", "read_num": "662"}, {"id": 3505, "replay_num": 2, "post_title": "下星期应该能到2块", "read_num": "174"}, {"id": 3506, "replay_num": 10, "post_title": "三一重工肯定是下一个白马股", "read_num": "1946"}, {"id": 3507, "replay_num": 3, "post_title": "利好", "read_num": "248"}]
# 读取评论数据
df = pd.DataFrame(comments)

# 情感分析函数
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# 为每个评论添加情感评分
df['sentiment'] = df['post_title'].apply(analyze_sentiment)

# 加权评分函数
def weighted_score(row):
    return row['sentiment'] * (int(row['read_num']) + int(row['replay_num']))

# 计算每个评论的加权评分
df['weighted_score'] = df.apply(weighted_score, axis=1)

# 计算综合评分
total_score = df['weighted_score'].sum()

# 将综合评分转换为概率（假设综合评分在[-1, 1]之间）
probability = (total_score + 1) / 2

# 输出结果
result = {
    "advance": probability
}

print(result)
