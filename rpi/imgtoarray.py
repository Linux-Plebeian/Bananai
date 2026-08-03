import numpy as np
from PIL import Image
import os

def convert(image):
    img = Image.open(f"{image}")
    img_gs = img.convert('L')
    #img_bin = img_grayscale.convert("1")
    img_arr2d = np.array(img_gs)
    img_list = img_arr2d.flatten().tolist()

    return img_list

def convert_oled(image):
    img = Image.open(f"{image}")
    img_bin =np.round((np.array(img.convert("L"))/255).flatten().tolist(),3)
    #print(img_bin)
    return img_bin
