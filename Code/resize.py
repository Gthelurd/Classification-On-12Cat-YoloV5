# coding = utf-8
# 这个文件主要是对图片的大小进行重构为640x640
# resizes
import cv2
import os
import glob

new_width = 640
new_height = 640
save_path = "F:\VSC-code\DeepLearning\FinalProj\image_resize_cls\\"
path = r"F:\\VSC-code\DeepLearning\\FinalProj\\data_image\\*.jpg"

def resize(new_width, new_height, path, save_path):
    for i in glob.glob(path):
        im1 = cv2.imread(i)
        # print(im1)
        print(im1.shape,i)
        im2 = cv2.resize(im1, (224, 224))
        cv2.imwrite(os.path.join(save_path, os.path.basename(i)), im2)
    

if __name__ == "__main__":
    resize(new_width, new_height, path, save_path)
