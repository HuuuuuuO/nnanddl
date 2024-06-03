data = [(0, 0), (4, 0), (5, 0), (2, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (8, 8), (1, 8), (7, 9)]

# 步骤1: 筛选并统计出第二个值一样的每一项
second_value_counts = {}  # 用于存储第二个值相同的项的数量
for item in data:
    second_value = item[1]
    if second_value in second_value_counts:
        second_value_counts[second_value].append(item)
    else:
        second_value_counts[second_value] = [item]

# 步骤2: 求出每一项中第一个值等于第二个值的比例
result = {}
for second_value, items in second_value_counts.items():
    first_equal_second_count = sum(1 for item in items if item[0] == second_value)
    total_items = len(items)
    ratio = first_equal_second_count / total_items if total_items!= 0 else 0
    result[second_value] = ratio

print(result)
