import numpy as np
from PIL import Image
import os
from pathlib import Path

def convert(image):
    img = Image.open(f"{image}")
    img_grayscale = img.convert('L')
    img_bin = img_grayscale.convert("1")
    img_arr2d = np.array(img_bin)
    img_list = img_arr2d.flatten().tolist()

    #print(matrix)
    return(img_list)
    
def name_training_data(path):
    file_ct = len([f for f in Path(path).iterdir() if f.is_file()])
    files = os.listdir(path)
    for i in range(0,file_ct):
        
        old_path = os.path.join(path, files[i])
        new_path = os.path.join(path, f"img_{i}.png")
        print(new_path)
        os.rename(old_path, new_path)
        img = Image.open(new_path)
        if img.size != (28, 28):
            img = img.resize((28,28))
            img.convert('L').save(new_path)

def read_training_data(paths):
    data = []
    for path in range(0,len(paths)):
        file_ct  = len([f for f in Path(paths[path]).iterdir() if f.is_file()])
        for i in range(0,file_ct):
            img = Image.open(f"{paths[path]}/img_{i}.png")
            #img_bin = img_gs.convert("1")
            img_arr2d = np.array(img)
            print(path, " ", i)
            img_list = ((img_arr2d.astype(np.float16))/255).flatten()
            data.append(img_list) 
    
        
    r = np.array(data)
    print(r)
    return r

