import numpy as np
from PIL import Image

def export_all(categories, images_per_category): # Name images <category>(<item>) starting with 1 for categories and 0 for item no. -> ex. 2(1)
    for i in range(1,categories+1):
        for j in range(0,images_per_category+1):
            img = Image.open(f"training_data/{i}({j}).png")

            img_bin = img.convert("1")
            matrix = np.array(img_bin).astype(int)
            final_output = matrix.flatten()

            #print(matrix)
            print(final_output.tolist())
def test(image):
    img = Image.open(f"images/{image}")

    img_bin = img.convert("1")
    matrix = np.array(img_bin).astype(int)
    final_output = matrix.flatten()

    #print(matrix)
    return(final_output.tolist())

