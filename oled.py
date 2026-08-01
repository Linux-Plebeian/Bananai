import time
import RPi.GPIO as gpio
import Adafruit_SSD1306
import imgtoarray as ita
from  PIL import Image

oled_reset = None
oled = Adafruit_SSD1306.SSD1306_128_64(rst=oled_reset)

oled.begin()
oled.clear()
oled.display()

banana = Image.new('1', (128, 64))


def draw_bitmap(image, bitmap_data, width, height):
      image.putdata(bitmap_data)
      return image

def printoled(x,y,text):
    oled.draw.text((x, y), text, font=None, fill=255)
    oled.display()
    
def draw_sprite(image, bmp_data, w, h):
    image = draw_bitmap(image, bmp_data, bitmap_width, bitmap_height)
    oled.image(image)
    oled.display()
      
def test():
    oled.clear()
    printoled((0, 0), "test123!", font=None, fill=255)
    oled.display()
    time.sleep(1)


gpio.cleanup()
