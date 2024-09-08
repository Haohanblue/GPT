import numpy as np

def find_min_cost(data):
    n = len(data)
    # 用一个数组来表示状态，dp[i]表示状态i的最小费用，初始化为正无穷大
    dp = [float('inf')] * (1 << n)
    # 初始状态，未购买任何会员的费用为0
    dp[0] = 0
    # 用一个数组来记录最优决策
    path = [-1] * (1 << n)
    
    # 遍历所有状态
    for mask in range(1 << n):
        # 遍历所有会员，尝试单独购买每个会员
        for i in range(n):
            if not (mask & (1 << i)):  # 如果会员i还没有购买
                new_mask = mask | (1 << i)
                cost = dp[mask] + data[i][i]
                if cost < dp[new_mask]:
                    dp[new_mask] = cost
                    path[new_mask] = (mask, i, i)  # 记录前一个状态和购买方式
            # 尝试联合购买套餐
            for j in range(n):
                if i != j and data[i][j] is not None and not (mask & (1 << i)) and not (mask & (1 << j)):
                    new_mask = mask | (1 << i) | (1 << j)
                    cost = dp[mask] + data[i][j]
                    if cost < dp[new_mask]:
                        dp[new_mask] = cost
                        path[new_mask] = (mask, i, j)  # 记录前一个状态和购买方式

    # 找到最终状态的最小费用和路径
    final_mask = (1 << n) - 1
    min_cost = dp[final_mask]
    best_combo = []
    purchase_matrix = np.zeros((n, n), dtype=int)

    # 反向推导最优路径
    mask = final_mask
    while mask:
        prev_mask, i, j = path[mask]
        if i == j:
            best_combo.append((i, "单独购买"))
            purchase_matrix[i][i] = 1
        else:
            best_combo.append((i, j, "联合购买"))
            purchase_matrix[i][j] = 1
        mask = prev_mask
    
    return min_cost, best_combo, purchase_matrix

# 数据矩阵
data = [[138,248,178,148,None],
        [248,158,238,None,168],
        [168,238,164,168,168],
        [None,None,None,118,None],
        [None,198,198,None,99]]

# 找到最小费用和最优组合
min_cost, best_combo, purchase_matrix = find_min_cost(data)

# 会员名称
membership_names = ["QQ SVIP", "绿钻", "Bilibili", "WPS", "JD PLUS"]

print(f"最划算的购买方案费用是: {min_cost}")
print("购买的会员方案:")
for combo in best_combo:
    if combo[1] == "单独购买":
        print(f"- {membership_names[combo[0]]}: 单独购买")
    else:
        print(f"- {membership_names[combo[0]]} 和 {membership_names[combo[1]]}: 联合购买")
print("\n最终购买情况的0-1矩阵:")
print(purchase_matrix)