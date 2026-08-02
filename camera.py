import cv2
from PIL import Image
import oled
def main():
    vid = cv2.VideoCapture(1)

    while True:
        for i in range(1,4):
            ret, frame = vid.read()
            #cv2.imshow('Camera Feed', frame)
            cv2.imwrite(f"images/banana{i}.png", frame)
            img = Image.open(f"images/banana{i}.png")
            img = img.resize((28, 28))
            img.save(f"images/banana{i}.png")
            img = Image.open(f"images/banana{i}.png")
            img = img.resize((128, 64))
            img.save(f"camera/banana{i}.png")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    

    vid.release()
    cv2.destroyAllWindows()
