# coding = utf-8
# 这段代码的主要作用是根据train_list.txt对数据集中的图片进行分类
import os
import shutil
import csv

# 获取图片路径与标签路径
label_file = "train_list.txt"
img_dir = "."
new_dir = "cat_12_train_resort"


def resort(img_dir, new_dir, label_file):
    # 创建index以及label和nums
    index = 0
    label_past = 0
    nums = 0
    # 获取父文件夹
    img_list = os.listdir(img_dir)
    # 创建新文件夹
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    # 重命名图片
    with open(label_file, "r") as f:
        for line in f.readlines():
            img_name, label = line.strip().split("\t")
            img_path = os.path.join(img_dir, img_name)
            nums += 1
            # print(img_name,label)
            # print(img_path)
            index += 1
            if label_past == label:
                pass
            else:
                label_past = label
                index = 0
            new_path = os.path.join(new_dir, "cat_{}_{}.jpg".format(label, index))
            shutil.copyfile(img_path, new_path)
            print(new_path)
    print("共有" + str(nums) + "张图片")


if __name__ == "__main__":
    resort(img_dir, new_dir, label_file)
