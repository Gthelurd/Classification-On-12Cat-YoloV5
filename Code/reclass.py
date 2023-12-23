# coding = utf-8
# 这段代码的主要作用就是转换为RGB格式
# reclass
from PIL import Image
import os

path = "F:\VSC-code\DeepLearning\FinalProj\cat_12_train_result\\"  # 最后要加双斜杠，不然会报错
save_path = "F:\VSC-code\DeepLearning\FinalProj\data_image\\"


def RGBconvert(path, save_path):
    filelist = os.listdir(path)

    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if img.mode == "P":
            img = img.convert("RGB")
        if img.mode == "L":
            img = img.convert("RGB")
        if img.mode == "CMYK":
            img = img.convert("RGB")
        if img.mode == "I":
            img = img.convert("RGB")
        if img.mode == "F":
            img = img.convert("RGB")
        img.save(save_path + file)
        print(save_path+file)


if __name__ == "__main__":
    RGBconvert(path, save_path)

# 参考链接：https://www.jianshu.com/p/e8d058767dfa
# 而labelme只认识RGB格式，不然会报:
# OSError: cannot write mode P as JPEG
# OSError: cannot write mode RGBA as JPEG
# 模式
# 1             1位像素，黑和白，存成8位的像素
# L             8位像素，黑白
# P             8位像素，使用调色板映射到任何其他模式
# RGB           3×8位像素，真彩
# RGBA          4×8位像素，真彩+透明通道
# CMYK          4×8位像素，颜色隔离
# YCbCr         3×8位像素，彩色视频格式
# I             32位整型像素
# F             32位浮点型像素
