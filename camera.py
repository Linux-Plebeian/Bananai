import cv2
from PIL import Image
def main():
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        cv2.imshow('Camera Feed', frame)
        cv2.imwrite('images/banana.png', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    img = Image.open('images/banana.png')
    img = img.resize((28, 28))
    img.save('images/banana.png')

    vid.release()
    cv2.destroyAllWindows()
