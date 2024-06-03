import os
import numpy as np
import pickle
import gzip
from PIL import Image, ImageEnhance
import json
import pylab
import matplotlib.pyplot as plt
import random

def convert_images_to_mnist_format(directory):
    images_data = []
    array_data = []
    label_data = [] 
    array_data_out = []
    label_data_out = [] 
    
    def get_label(path):
        path_parts = path.split(os.sep)
        return path_parts[-1] if len(path_parts) > 1 else "unknown"  
    
    for root, dirs, files in os.walk(directory):  
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(os.path.join(root, filename)).convert('L')
                # contrast = ImageEnhance.Contrast(img)
                # img = contrast.enhance(7)  
                # enhanced_image.show()
                resized_img = img.resize((28, 28))
                arr = np.array(resized_img)
                arr_1 = arr.flatten()

                # 绘制分布图并标记出现次数最多的区间
                a,b,threshold_1,threshold_2=find_most_frequent_interval(arr_1)

                # 找到大面积像素点
                # 计算每个元素的出现次数
                counts = np.bincount(arr_1)

                # 找出出现次数最多的元素
                max_count = max(counts)
                most_frequent_elements = np.where(counts == max_count)[0]

                # 找出出现次数最多的元素
                # print("元素:",most_frequent_elements,"出现次数:",max_count)


                # 找到最小值和最大值
                min_val = int(min(arr_1))
                max_val = int(max(arr_1))
                middle_value = (min_val + max_val) *0.55
                # print('===========',min_val,max_val,(middle_value))

                # arr[(arr > threshold_1) & (arr < threshold_2)] = 255
                # arr[arr > threshold_1] = 255
              
                normalized_arr = 1- arr.astype(np.float32) / 255
                
                # 图片处理后保存时所用数据
                image_from_array = Image.fromarray((normalized_arr * 255).astype(np.uint8))
                images_data.append(image_from_array)
                

                # 图片数据，归一化数据拉平成一维数组
                flattened_arr = normalized_arr.flatten()
                array_data.append(flattened_arr)



                # 标签数据
                label = get_label(root) 
                label_data.append(label)
                    

    label_data_out=np.array(label_data, dtype=np.int64)
    array_data_out=np.array(array_data, dtype=np.float32)
    
    return images_data, array_data,label_data,array_data_out, label_data_out

def save_as_pkl_gz(data_list, file_path):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data_list, f)

def save_images(images_data, label_data, directory):
    os.makedirs(directory, exist_ok=True)
    
    for i, (image, label) in enumerate(zip(images_data, label_data)):
        sub_dir = os.path.join(directory, label)
        os.makedirs(sub_dir, exist_ok=True)
        
        image.save(f'{sub_dir}/image_{label}_{i}.png')
        # show_image(f'{sub_dir}/image_{label}_{i}.png')

def load_data():

    f = gzip.open('mydata.pkl.gz', 'rb')
    test_data = pickle.load(f, encoding="latin1")
    print(test_data)
    f.close()
    return test_data

def show_image(image_path):
    img_x = Image.open(image_path)
    pylab.imshow(img_x)
    pylab.gray()
    pylab.show()
import numpy as np
import matplotlib.pyplot as plt

def find_most_frequent_interval(arr, intervals=1):
    """
    找出出现次数最多的区间。
    
    参数:
    arr - 输入数组
    intervals - 区间的数量
    
    返回:
    most_frequent_interval - 出现次数最多的区间
    frequency - 最高频率
    """
    # 确定区间大小
    interval_size = 256 // intervals
    # 使用numpy的digitize函数将数组中的数值分配到不同的区间
    bins = np.arange(0, 256, interval_size)
    labels = np.digitize(arr, bins) - 1  # 减1是因为digitize函数的输出是基于bins的索引，而我们想要的是实际的区间编号
    
    # 统计每个区间内的元素数量
    counts = np.bincount(labels, minlength=intervals)
    
    # 找出出现次数最多的区间及其频率
    max_index = np.argmax(counts)
    most_frequent_interval = (max_index * interval_size, (max_index + 1) * interval_size)
    frequency = counts[max_index]
    
    return most_frequent_interval, frequency,max_index * interval_size,(max_index + 1) * interval_size

def plot_distribution_and_intervals(arr, intervals=10):
    """
    绘制数组分布图，并标记出现次数最多的区间。
    
    参数:
    arr - 输入数组
    intervals - 区间的数量
    """
    # 计算分布并找出出现次数最多的区间
    most_frequent_interval, frequency = find_most_frequent_interval(arr, intervals)
    
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(arr, bins=intervals, edgecolor='black', alpha=0.7, label='Distribution')
    
    # 标记出现次数最多的区间
    plt.axvline(x=most_frequent_interval[0], color='red', linestyle='--', linewidth=2, label=f'Most Frequent Interval: {most_frequent_interval}')
    plt.text(most_frequent_interval[0], plt.gca().get_ylim()[1]*0.9, f'Frequency: {frequency}', ha='right', va='top', fontsize=12)
    
    # 设置图表的标题、x轴标签和y轴标签
    plt.title('Array Distribution and Most Frequent Interval')
    plt.xlabel('Value (0-255)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 显示图表
    plt.show()

def array_split(array_data):
    random.shuffle(array_data)

    train_size = int(len(array_data) * 5 / 6)

    train_set = array_data[:train_size]
    test_set = array_data[train_size:]

    return train_set,test_set


if __name__ == '__main__':
    directory_input = 'mydata_input'
    directory_output = 'mydata_output'
    pkl_path = 'mydata.pkl.gz'
    images_data, array_data, label_data,array_data_out, label_data_out = convert_images_to_mnist_format(directory_input)


    # print((array_data_out))
    # print((array_data))
    # print((array_data_out.shape))
    # print(label_data_out.shape)
    # save_as_pkl_gz([ array_data_out, label_data_out], pkl_path)
    # save_images(images_data, label_data, directory_output)
    