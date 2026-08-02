import numpy as np
from PIL import Image
import os

def name_data(name, name2):
    for i in range(0,10):
        os.rename(f"./training_images/{name}_0"+str(i)+".png", "./training_images/"+str(i)+".png")
    for i in range(10,72):
        os.rename(f"./training_images/{name}_"+str(i)+".png", "./training_images/"+str(i)+".png")
    for i in range(0,10):
        os.rename(f"./training_images/{name2}_0"+str(i)+".png", "./training_images/"+str(i+72)+".png")
    for i in range(10,72):
        os.rename(f"./training_images/{name2}_"+str(i)+".png", "./training_images/"+str(i+72)+".png")


def read_training_data(elements): 
    data = []
    for i in range(0,elements):
        img = Image.open(f"training_images/{i}.png")
        img_gs = img.convert('L')
        #img_bin = img_gs.convert("1")
        img_arr2d = np.array(img_gs)
        img_list = (img_arr2d/255).flatten().tolist()
        data.append(img_list) 
    r = np.array(data)
    print(r)
    return r
    
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
