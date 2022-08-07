from PIL import Image
import os
import time

start = time.time()
# full_path = "/home/dastkd69/Minecraft/storage/train2017/"  #  640*640
full_path = "/home/dastkd69/Minecraft/storage/val2017/"  #  640*640
print("Collecting image paths...")
img_list = os.listdir(full_path)
img_paths = []
for img in img_list:
    img_paths.append(full_path+img)


def get_img_size(path):
    width, height = Image.open(path).size
    return width*height


def find_largest(img_paths):
    smallest = min(img_paths, key=get_img_size)
    Image.open(smallest).show
    print(Image.open(smallest).size)


def add_padding(img_path):
    image = Image.open(img_path)
    width, height = image.size
    new_width = width + (640 - width)
    new_height = height + (640 - height)
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 255))
    result.paste(image, (0, 0))
    result.save(img_path)


print("Finding minimum of images...")
# map(add_padding, img_paths)
find_largest(img_paths)
print(f"Completed! Took {time.time()-start}s!")

