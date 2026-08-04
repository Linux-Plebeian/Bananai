import cv2
import numpy as np
from PIL import Image
import oled
import imgtoarray as ita
import keyboard
from picamera2 import Picamera2
import time

breakage = 0

# Initialize PiCamera2 with desired resolution
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (128, 64)})
picam2.configure(config)
picam2.start()


def main():

    
    while True:
        for i in range(1, 4):
            vid = picam2.capture_array()
            cv2.imwrite(f"images/banana{i}.png", vid)
            img = Image.open(f"images/banana{i}.png")
            img.save(f"camera/banana{i}.png")
            img_gs = np.array(img.convert('L'))
            _, img_bin = cv2.threshold(img_gs, 128, 255, cv2.THRESH_BINARY)
            img = img.resize((28, 28))
            img.save(f"images/banana{i}.png")
            img_pil = Image.fromarray(img_bin).convert('1')
            oled.oled.fill(0)
            oled.oled.image(img_pil)
            oled.oled.show() 
            
            if keyboard.is_pressed('q'):
                break
                
        if keyboard.is_pressed('q'):
            break

    print("Processing . . .")
    oled.oled.fill(0)
    oled.oled.text("Processing",1,1,1)
    oled.oled.show()
    
#picam2.stop()
#cv2.destroyAllWindows()

