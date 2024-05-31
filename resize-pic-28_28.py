from PIL import Image  
import pylab
import numpy as np

print('input the file you want to resize to 28*28:')
origfile = input()
destfile = origfile + '28_28.png'

def resize_image(input_image_path, output_image_path, size):  
    original_image = Image.open(input_image_path)  
    resized_image = original_image.resize(size)  
    resized_image.save(output_image_path)  

def convert_to_black_white_then_normalize(image_path, output_path):
    with Image.open(image_path) as image:
        # 灰度图像化
        gray_image = image.convert('L')
        # 将图像转换为NumPy数组
        gray_array = np.array(gray_image)
        # 将像素值从0到255归一化到0到1(chaonin:黑色作为背景)
        normalized_array = 1 - gray_array / 255.0
        # 将归一化的灰度数组转换为图像
        normalized_gray_image = Image.fromarray((normalized_array * 255).astype(np.uint8))
        # 保存图像
        normalized_gray_image.save(output_path)
def show_image(image_path):
    img_x = Image.open(image_path)
    pylab.imshow(img_x)
    pylab.gray()
    pylab.show()

def read_pic_pixel(image_path):
    image = Image.open(image_path)
    width, height = image.size
    print (width)
    print (height)
    i = 1 
    for h in range(height):
        for w in range(width):
            pixel_value = image.getpixel((w, h))
            print(str(i)+':'+str(pixel_value))  # 输出像素值，例如：(255, 0, 0)代表红色
            i = i+1
  
# 使用函数将图片压缩为28x28像素  
resize_image(origfile, destfile, (28, 28))
# 黑白化并归一化【RBG转换为0-1之间的灰度值】
convert_to_black_white_then_normalize(destfile, destfile) 
#read_pic_pixel(destfile)
show_image(destfile)