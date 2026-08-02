import board
import busio
import adafruit_ssd1306
import time 

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
oled.text("BananAI v0.2", 0, 0, 1)
def draw_sprite(x, y, l, h, sprite_bin):
    oled.text("DRAW",0,0,1)
    for i in range(0,l):
        for j in range(0,h):
            if (sprite_bin[i + j*l]) >= 0.5:
                oled.pixel(i+x,j+y,1)
    oled.show()

def clear():
    oled.fill(0)
def test():
    oled.fill(1)
    time.sleep(.1)
    oled.fill(0)
    oled.show()
    oled.text("BananAI v0.2", 0, 0, 1)
    oled.show()
