# 这段代码的作用主要是将labelme的.json格式转换为YOLO的.txt格式
# coding = utf-8
# 这里的.json格式与.txt格式都是bbox(xywh)的格式
# labels
import json
import os

json_path = os.path.join("./data_json")
txt_path = os.path.join("./data_txt\\")

def read_json(json_file):
    with open(json_file, "r") as f:
        load_dict = json.load(f)
    f.close()
    return load_dict

def json2txt(json_path, txt_path):
    for json_file in os.listdir(json_path):
        txt_name = txt_path + json_file[0:-5] + ".txt"
        print(txt_name)
        # print(txt_name)
        txt_file = open(txt_name, "w")
        json_file_path = os.path.join(json_path, json_file)
        # print(json_file_path)
        json_data = read_json(json_file_path)
        # print(json_data)
        imageWidth = json_data["imageWidth"]
        imageHeight = json_data["imageHeight"]
        # print(imageWidth,imageHeight)
        for i in range(len(json_data["shapes"])):
            label = json_data["shapes"][i]["label"]
            # print(label)
            # if label=='Lesions':
            #     index=0
            # else:
            #     index=1
            x1 = json_data["shapes"][i]["points"][0][0]
            x2 = json_data["shapes"][i]["points"][1][0]
            y1 = json_data["shapes"][i]["points"][0][1]
            y2 = json_data["shapes"][i]["points"][1][1]
            # 将标注框按照图像大小压缩
            x_center = (x1 + x2) / 2 / imageWidth
            y_center = (y1 + y2) / 2 / imageHeight
            bbox_w = abs(x2 - x1) / imageWidth
            bbox_h = abs(y2 - y1) / imageHeight
            bbox = ((x_center), (y_center), abs(bbox_w), abs(bbox_h))
            # print(bbox)
            label = getname(label)
            txt_file.write(str(label) + "\t" + " ".join([str(a) for a in bbox]) + "\n")
            # print(label)

def getname(label):
    name = {
        "cat_0": 0,
        "cat_1": 1,
        "cat_2": 2,
        "cat_3": 3,
        "cat_4": 4,
        "cat_5": 5,
        "cat_6": 6,
        "cat_7": 7,
        "cat_8": 8,
        "cat_9": 9,
        "cat_10": 10,
        "cat_11": 11,
    }[label]
    return name


if __name__ == "__main__":
    json2txt(json_path, txt_path)
