import os


base_ = "datasets"
classname_ = [
    "cat_0",
    "cat_1",
    "cat_2",
    "cat_3",
    "cat_4",
    "cat_5",
    "cat_6",
    "cat_7",
    "cat_8",
    "cat_9",
    "cat_10",
    "cat_11",
]

imgpath_ = "image_resize_cls"


def splitclass(base_, imgpath_):
    class_name_past = 0
    index = 0
    # Iterate over the images in the dataset
    for image_path in os.listdir(imgpath_):
        # Get the class of the cat from the filename
        class_name = image_path.split(".")[0].split("_")[1]
        # print(class_name)
        if class_name_past == class_name:
            index += 1
        else:
            class_name_past = class_name
            index = 0
        class_ = f"cat_{class_name}"
        print(
            os.path.join(imgpath_, image_path), os.path.join(base_, class_, image_path)
        )
        # Move the image to the directory for its class
        os.rename(
            os.path.join(imgpath_, image_path), os.path.join(base_, class_, image_path)
        )


# Create a directory for each class of cat
def makefile(base_, classname_):
    for class_name in classname_:
        if not os.path.exists(os.path.join(base_, class_name)):
            os.makedirs(os.path.join(base_, class_name))


if __name__ == "__main__":
    splitclass(base_, imgpath_)
    makefile(base_, classname_)
