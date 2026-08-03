import cv2
from PIL import Image
import oled
import imgtoarray as ita
import keyboard
from picamera2 import Picamera2 
import time 

breakage = 0


picam2 = Picamera2() 
config = picam2.create_still_configuration() 
picam2.configure(config) 

picam2.start() 
picam2.resolution = (128, 64)
def deez():
    print("Processing...")
    breakage = 1
    
def main():
    
    test = 0
    breakage = 0
    while True:
        for i in range(1,4):
            #ret, frame = vid.read()
            #cv2.imshow('Camera Feed', frame)
            vid = picam2.capture_array()
            cv2.imwrite(f"images/banana{i}.png", vid)
            img = Image.open(f"images/banana{i}.png")
            img.save(f"camera/banana{i}.png")
            _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            img = img.resize((28, 28))
            img.save(f"images/banana{i}.png")
            img_pil = Image.fromarray(img_bin)
            oled.clear()
            oled.oled.image(img_pil)
            oled.show() 
            if keyboard.is_pressed('q'):
                break
        if keyboard.is_pressed('q'):
            break

    print("Processing . . .")

    

    vid.release()
    cv2.destroyAllWindows()
