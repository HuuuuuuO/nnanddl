import os
import numpy as np
import pickle
import gzip
from PIL import Image
import json
import pylab

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
                resized_img = img.resize((28, 28))
                arr = np.array(resized_img)
                # if filename.endswith(('.jpg', '.jpeg')):
                arr_1 = arr.flatten()
                # 找到最小值和最大值
                min_val = int(min(arr_1))
                max_val = int(max(arr_1))
                middle_value = (min_val + max_val) *0.55
                print('======================',min_val,max_val,(middle_value))

                # threshold = middle_value
                # arr[arr < threshold] = 0
                # arr[arr >= threshold] = 255
                # else:
                #     threshold = 250
                #     arr[arr < threshold] = 0
                #     arr[arr >= threshold] = 255
                normalized_arr =1-arr.astype(np.float32) / 255
                
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


if __name__ == '__main__':
    directory_input = 'mydata_input'
    directory_output = 'mydata_output'
    pkl_path = 'mydata.pkl.gz'
    images_data, array_data, label_data,array_data_out, label_data_out = convert_images_to_mnist_format(directory_input)
    print((array_data_out))
    # print((array_data))
    print((array_data_out.shape))
    print(label_data_out.shape)
    save_as_pkl_gz([ array_data_out, label_data_out], pkl_path)
    save_images(images_data, label_data, directory_output)