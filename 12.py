# import random
# training_data=[1,2,3,4,5,6,7,8,9]
# random.shuffle(training_data)

# n=len(training_data)
# mini_batch_size=3
# mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
# print(mini_batches)


import random
training_data=[1,2,3,4,5,6,7,8,9]

epochs = 6
mini_batch_size = 3
n = len(training_data)  # 训练数据集的大小
custom_data = [91,92,93,94,95,96,97,98,99]  # 自定义数据列表

for epoch in range(epochs):
    random.shuffle(training_data)
    random.shuffle(custom_data)
    
    # 创建mini-batches
    mini_batches = [
        training_data[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)
    ]
    print(mini_batches)
    
     # 随机选择一个mini-batch进行替换
    if epoch % 2 == 0:  # 每隔两个epoch替换一次（可根据需要调整）
        replace_index = random.randint(0, len(mini_batches) - 1)
        mini_batches[replace_index] += custom_data[:mini_batch_size]  # 将自定义数据添加到选定的mini-batch中
        
        # 如果自定义数据量大于mini_batch_size，则从自定义数据中移除已添加的部分
        custom_data = custom_data[mini_batch_size:]
        
        # 更新mini_batches以反映更改
        mini_batches = [item for sublist in mini_batches for item in sublist][:n]
        
    print(mini_batches)

    # 这里可以继续处理mini_batches，例如训练模型
