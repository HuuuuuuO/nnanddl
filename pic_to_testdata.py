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
                # img = contrast.enhance(2)  
                # enhanced_image.show()
                resized_img = img.resize((28, 28))
                arr = np.array(resized_img)
                arr_1 = arr.flatten()
                # hist=count_distribution(arr_1)
                # plot_distribution(hist)

                # 计算每个元素的出现次数
                counts = np.bincount(arr_1)
                # 找出出现次数最多的元素
                max_count = max(counts)
                most_frequent_elements = np.where(counts == max_count)[0][0]
                # print(most_frequent_elements)

                intervals=40
                threshold_1 = most_frequent_elements-int(intervals/2)
                threshold_2 = most_frequent_elements+int(intervals/2)

                arr[(arr > threshold_1) & (arr < threshold_2)] = 255
                arr[arr < threshold_1] = 0


                # threshold=127
                # arr[arr > threshold] = 255
                # arr[arr < threshold] = 0
              
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



def count_distribution(arr):
    hist = {}
    for num in arr:
        if num in hist:
            hist[num] += 1
        else:
            hist[num] = 1
    # print(hist)
    return hist

def plot_distribution(hist):
    values = list(hist.keys())
    # 存储对应的频率
    frequencies = [hist[value] for value in values]

    plt.title('Array Distribution')
    plt.xlabel('Value (0-255)')
    plt.ylabel('Frequency')

    plt.bar(values, frequencies)

    plt.show()


def array_split(array_data,label_data):
    combined = list(zip(array_data, label_data))
    random.shuffle(combined)
    array_data[:], label_data[:] = zip(*combined)

    train_size = int(len(array_data) * 5 / 6)
    test_size = len(array_data) - train_size

    # 提取训练集和测试集
    train_set_array, train_set_label = array_data[:train_size], label_data[:train_size]
    test_set_array, test_set_label = array_data[train_size:], label_data[train_size:]
    print(len(train_set_array))

    return train_set_array, train_set_label,test_set_array, test_set_label




if __name__ == '__main__':
    directory_input = 'mydata_input'
    directory_output = 'mydata_output'
    pkl_path_train = 'mydata_train.pkl.gz'
    pkl_path_test = 'mydata_test.pkl.gz'
    images_data, array_data, label_data,array_data_out, label_data_out = convert_images_to_mnist_format(directory_input)
    train_set_array, train_set_label,test_set_array, test_set_label=array_split(array_data_out,label_data_out)
    print((array_data_out))
    save_as_pkl_gz([train_set_array, train_set_label], pkl_path_train)
    save_as_pkl_gz([test_set_array, test_set_label], pkl_path_test)
    save_images(images_data, label_data, directory_output)
    