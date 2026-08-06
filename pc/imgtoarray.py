import numpy as np
from PIL import Image
import os
from pathlib import Path

img_size = (64, 64)

def convert_bin(img):
    img_bin = Image.open(img).convert("1")
    img_arr2d = np.array(img_bin.resize(img_size))
    img_list = img_arr2d.flatten().tolist()
    return img_list

def convert_rgb(img):
    img_rgb = Image.open(img).resize(img_size).convert("RGB")
    pixel_rgb_values = list(img_rgb.getdata())
    img_flat = []
    for tup in pixel_rgb_values:
        img_flat.extend(tup)
    return ((np.array(img_flat)/255)).tolist()

def read_training_data_bin(paths):
    data = []
    for path in range(0,len(paths)):
        file_ct  = len([f for f in Path(paths[path]).iterdir() if f.is_file()])
        for i in range(0,file_ct):
            img = Image.open(f"{paths[path]}/img_{i}.png")
            img_bin = np.array(img.convert("1"))
            print(path, " ", i)
            img_list = ((img_bin.astype(np.float16))/255).flatten()
            data.append(img_list)
    r = np.array(data)
    print(r)
    return r

def read_training_data_rgb(paths):
    data = []
    for path in range(0,len(paths)):
        file_ct  = len([f for f in Path(paths[path]).iterdir() if f.is_file()])
        for i in range(0,file_ct):
            img = Image.open(f"{paths[path]}/img_{i}.png")
            img = convert_rgb(img)
            print(path, " ", i)
            img_list = (np.array(img).astype(np.float16)).flatten().tolist()
            data.append(img_list)
    r = np.array(data)
    print(r)
    return r


def name_training_data(path):
    file_ct = len([f for f in Path(path).iterdir() if f.is_file()])
    files = os.listdir(path)
    for i in range(0,file_ct):
        old_path = os.path.join(path, files[i])
        new_path = os.path.join(path, f"img_{i}.png")
        print(new_path)
        os.rename(old_path, new_path)
        img = (Image.open(new_path))
        if img.size != (img_size):
            img = img.resize(img_size)
            img.save(new_path)

