import numpy as np
from PIL import Image

def read_training_data(categories, images_per_category): # Name images <category>(<item>) starting with 1 for categories and 0 for item no. -> ex. 2(1)
    data = [[]]
    for i in range(1,categories+1):
        for j in range(0,images_per_category):
            img = Image.open(f"training_data/faces/{i}({j}).png")

            img_bin = img.convert("1")
            img_arr2d = np.array(img_bin).astype(int)
            img_list = img_arr2d.flatten().tolist()
            data.append(img_list) 
    del data[0] 
    r = np.array(data)
    print(r)
    return r
def test(image):
    img = Image.open(f"images/{image}")

    img_bin = img.convert("1")
    matrix = np.array(img_bin).astype(int)
    final_output = matrix.flatten()

    #print(matrix)
    return(final_output.tolist())

