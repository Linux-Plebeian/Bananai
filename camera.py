import cv2
from PIL import Image
import oled
import imgtoarray as ita
import keyboard
def main():
    vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
    test = 0
    while True:
        for i in range(1,4):
            ret, frame = vid.read()
            #cv2.imshow('Camera Feed', frame)
            cv2.imwrite(f"images/banana{i}.png", frame)
            img = Image.open(f"images/banana{i}.png")
            img = img.resize((128, 64))
            img.save(f"camera/banana{i}.png")
            img = img.resize((28, 28))
            img.save(f"images/banana{i}.png")
            print("frame",test)
            test+=1
            oled.draw_sprite(1,1,128,64,ita.convert("camera/banana1.png"))
        
        if keyboard.is_pressed('q'):
            break
            

    

    vid.release()
    cv2.destroyAllWindows()
