print("Hello World")
# 解决鸡兔同笼问题
# 鸡兔同笼问题是一个古老的数学问题，它是一个关于二元一次方程组的问题。问题描述如下：
# 有若干只鸡和兔子在一个笼子里，从上面数共有35个头，从下面数共有94只脚。问鸡和兔子各有多少只？
# 问题分析
# 设鸡的数量为x，兔子的数量为y，根据题意可以得到以下两个方程：
# x + y = 35
# 2x + 4y = 94
# 通过解这两个方程，可以得到鸡和兔子的数量。
# 解题思路
# 1. 定义一个函数solve_chicken_rabbit，该函数接收两个参数，分别为头的数量和脚的数量。
# 2. 在函数内部，根据题意，可以得到以下两个方程：
# x + y = head
# 2x + 4y = foot
# 3. 通过解这两个方程，可以得到鸡和兔子的数量。
# 4. 最后，调用solve_chicken_rabbit函数，并输出结果。
def solve_chicken_rabbit(head, foot):
    for x in range(head + 1):
        y = head - x
        if 2 * x + 4 * y == foot:
            return x, y
    return None, None
# 调用solve_chicken_rabbit函数，并输出结果
head = 35
foot = 94
x, y = solve_chicken_rabbit(head, foot)
print(f"鸡的数量为：{x}，兔子的数量为：{y}")